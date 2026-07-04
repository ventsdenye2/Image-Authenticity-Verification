import argparse
import csv
import os
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


LABELS = {"REAL": 0, "FAKE": 1}
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def load_uint8_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def image_to_tensor(
    path: Path,
    train: bool,
    random_erasing: float = 0.0,
    cached_image: np.ndarray | None = None,
) -> torch.Tensor:
    if cached_image is None:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    else:
        arr = cached_image.astype(np.float32) / 255.0

    if train:
        if random.random() < 0.5:
            arr = np.ascontiguousarray(arr[:, ::-1, :])
        if random.random() < 0.5:
            # Small translations are helpful on 32x32 images without changing semantics.
            pad = 4
            padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
            top = random.randint(0, pad * 2)
            left = random.randint(0, pad * 2)
            arr = padded[top : top + 32, left : left + 32, :]

    tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
    if train and random.random() < random_erasing:
        # A small cutout discourages memorising isolated generator artifacts while
        # preserving the global colour/frequency cues that are useful here.
        size = random.randint(2, 6)
        top = random.randint(0, 32 - size)
        left = random.randint(0, 32 - size)
        tensor[:, top : top + size, left : left + size] = MEAN
    return (tensor - MEAN) / STD


class AIDetTrainDataset(Dataset):
    def __init__(self, data_dir: Path, train: bool = True, random_erasing: float = 0.0) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.train = train
        self.random_erasing = random_erasing
        self.image_cache: np.ndarray | None = None
        for class_name, label in LABELS.items():
            class_dir = data_dir / "train" / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing folder: {class_dir}")
            self.samples.extend((path, label) for path in sorted(class_dir.glob("*.jpg")))
        if not self.samples:
            raise RuntimeError(f"No training images found under {data_dir / 'train'}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        cached_image = None if self.image_cache is None else self.image_cache[index]
        image = image_to_tensor(path, self.train, self.random_erasing, cached_image)
        return image, torch.tensor(label, dtype=torch.long)

    def preload(self) -> None:
        cache = np.empty((len(self.samples), 32, 32, 3), dtype=np.uint8)
        paths = (path for path, _ in self.samples)
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as executor:
            for index, image in enumerate(executor.map(load_uint8_image, paths, chunksize=128)):
                cache[index] = image
        self.image_cache = cache


class AIDetTestDataset(Dataset):
    def __init__(self, test_dir: Path) -> None:
        self.paths = [test_dir / f"{idx}.jpg" for idx in range(1, 20001)]
        missing = [path.name for path in self.paths if not path.exists()]
        if missing:
            shown = ", ".join(missing[:10])
            raise FileNotFoundError(f"Missing test images: {shown}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.paths[index]
        return image_to_tensor(path, train=False), int(path.stem)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class SmallResNet(nn.Module):
    def __init__(self, num_classes: int = 2, width: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(width, width, blocks=2, stride=1)
        self.layer2 = self._make_layer(width, width * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(width * 2, width * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(width * 4, width * 4, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(width * 4, num_classes)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride)]
        layers.extend(BasicBlock(out_channels, out_channels, 1) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class SmallResNetV2(nn.Module):
    """A CIFAR-style ResNet-18 with a full 8x final stage.

    The original model kept its final two stages at the same width.  Expanding
    the last stage gives the classifier more capacity for subtle synthesis
    artifacts while retaining 32x32 feature maps and a modest parameter count.
    """

    def __init__(self, num_classes: int = 2, width: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = SmallResNet._make_layer(width, width, blocks=2, stride=1)
        self.layer2 = SmallResNet._make_layer(width, width * 2, blocks=2, stride=2)
        self.layer3 = SmallResNet._make_layer(width * 2, width * 4, blocks=2, stride=2)
        self.layer4 = SmallResNet._make_layer(width * 4, width * 8, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(width * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


class EfficientNetDetector(nn.Module):
    """ImageNet-pretrained EfficientNet with in-model colour conversion.

    Keeping resize and normalisation in the model makes training, validation,
    TTA and checkpoint inference use exactly the same preprocessing path.
    """

    def __init__(
        self,
        dropout: float = 0.2,
        pretrained: bool = False,
        color_space: str = "ycbcr",
        input_size: int = 224,
    ) -> None:
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights, dropout=dropout)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 2)
        self.color_space = color_space
        self.input_size = input_size
        self.register_buffer("cifar_mean", MEAN.unsqueeze(0), persistent=False)
        self.register_buffer("cifar_std", STD.unsqueeze(0), persistent=False)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x * self.cifar_std + self.cifar_mean).clamp(0, 1)
        if self.color_space == "ycbcr":
            r, g, b = x.unbind(1)
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
            cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
            x = torch.stack((y, cb, cr), dim=1)
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            # Antialiasing is unnecessary when only upsampling and is very slow
            # for batched CUDA interpolation on Windows.
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.imagenet_mean) / self.imagenet_std
        return self.backbone(x)


def make_model(
    version: str,
    width: int,
    dropout: float,
    pretrained: bool = False,
    color_space: str = "ycbcr",
    input_size: int = 224,
) -> nn.Module:
    if version == "v1":
        return SmallResNet(width=width)
    if version == "v2":
        return SmallResNetV2(width=width, dropout=dropout)
    if version == "efficientnet-b0":
        return EfficientNetDetector(dropout, pretrained, color_space, input_size)
    raise ValueError(f"Unknown model version: {version}")


@dataclass
class Metrics:
    loss: float
    accuracy: float


def stratified_split(samples: list[tuple[Path, int]], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for index, (_, label) in enumerate(samples):
        by_label[label].append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in by_label.values():
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_ratio))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    ema_model: AveragedModel | None = None,
) -> Metrics:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if ema_model is not None:
                    ema_model.update_parameters(model)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return Metrics(total_loss / total, correct / total)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, AveragedModel) else model


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    full_train = AIDetTrainDataset(args.data_dir, train=True, random_erasing=args.random_erasing)
    full_val = AIDetTrainDataset(args.data_dir, train=False)
    if args.cache_images:
        start = time.time()
        full_train.preload()
        # Both datasets have identical sorted samples, so one 293 MiB cache is enough.
        full_val.image_cache = full_train.image_cache
        print(f"Cached {len(full_train)} decoded images in {time.time() - start:.1f}s")
    train_indices, val_indices = stratified_split(full_train.samples, args.val_ratio, args.seed)

    train_loader = DataLoader(
        Subset(full_train, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        Subset(full_val, val_indices),
        batch_size=args.batch_size * args.val_batch_multiplier,
        shuffle=False,
        # On Windows each worker receives its own copy of the decoded cache.
        # Validation has no augmentation, so a second worker pool costs memory
        # without helping throughput meaningfully.
        num_workers=0 if args.cache_images else args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    print(f"Train images: {len(train_indices)} | Val images: {len(val_indices)}")
    return train_loader, val_loader


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = make_loaders(args)
    model = make_model(
        args.model_version,
        args.width,
        args.dropout,
        args.pretrained,
        args.color_space,
        args.input_size,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema_model = None

    best_acc = -1.0
    best_path = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            if ema_model is None and args.ema_decay > 0 and epoch >= args.ema_start:
                # Starting EMA from a partially trained model avoids averaging
                # the unstable high-learning-rate weights from the first epochs.
                ema_model = AveragedModel(
                    model,
                    multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
                    use_buffers=True,
                )
            train_metrics = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler, ema_model
            )
            eval_model = ema_model if ema_model is not None else model
            val_metrics = run_epoch(eval_model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - start
            lr = scheduler.get_last_lr()[0]

            writer.writerow(
                [
                    epoch,
                    f"{train_metrics.loss:.6f}",
                    f"{train_metrics.accuracy:.6f}",
                    f"{val_metrics.loss:.6f}",
                    f"{val_metrics.accuracy:.6f}",
                    f"{lr:.8f}",
                    f"{elapsed:.2f}",
                ]
            )
            f.flush()

            print(
                f"Epoch {epoch:02d}/{args.epochs} "
                f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} "
                f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} "
                f"lr={lr:.6f} time={elapsed:.1f}s"
            )

            if val_metrics.accuracy > best_acc:
                best_acc = val_metrics.accuracy
                torch.save(
                    {
                        "model": unwrap_model(eval_model).state_dict(),
                        "args": serializable_args(args),
                        "best_val_acc": best_acc,
                        "epoch": epoch,
                    },
                    best_path,
                )
                print(f"Saved new best model to {best_path} (val_acc={best_acc:.4f})")

    return best_path


@torch.no_grad()
def predict(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    device = select_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    width = int(checkpoint.get("args", {}).get("width", args.width))
    checkpoint_args = checkpoint.get("args", {})
    version = str(checkpoint_args.get("model_version", "v1"))
    dropout = float(checkpoint_args.get("dropout", args.dropout))
    color_space = str(checkpoint_args.get("color_space", args.color_space))
    input_size = int(checkpoint_args.get("input_size", args.input_size))
    model = make_model(version, width, dropout, False, color_space, input_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = AIDetTestDataset(args.data_dir / "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )

    rows: list[tuple[int, int]] = []
    for images, ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            if args.tta_flip:
                logits = (logits + model(images.flip(-1))) * 0.5
        preds = logits.argmax(dim=1).cpu().tolist()
        rows.extend((int(image_id), int(label)) for image_id, label in zip(ids.tolist(), preds))

    rows.sort(key=lambda item: item[0])
    validate_rows(rows)
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    with args.submission_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {args.submission_path}")
    return args.submission_path


def validate_rows(rows: list[tuple[int, int]]) -> None:
    if len(rows) != 20000:
        raise ValueError(f"Expected 20000 predictions, got {len(rows)}")
    ids = [row[0] for row in rows]
    labels = [row[1] for row in rows]
    if ids != list(range(1, 20001)):
        raise ValueError("Submission IDs must be exactly 1..20000")
    bad_labels = sorted(set(labels) - {0, 1})
    if bad_labels:
        raise ValueError(f"Labels must be 0 or 1, got {bad_labels}")


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return torch.load(checkpoint_path, map_location=device, weights_only=False)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build cannot access CUDA.")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AI-generated image detector.")
    parser.add_argument("--data-dir", type=Path, default=Path("img_AIdet/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    parser.add_argument("--submission-path", type=Path, default=Path("submission_v2.csv"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-batch-multiplier", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--model-version", choices=["v1", "v2", "efficientnet-b0"], default="v2"
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--color-space", choices=["rgb", "ycbcr"], default="ycbcr")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--ema-start", type=int, default=5)
    parser.add_argument("--random-erasing", type=float, default=0.15)
    parser.add_argument("--no-cache-images", dest="cache_images", action="store_false")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--no-tta-flip", dest="tta_flip", action="store_false")
    parser.set_defaults(tta_flip=True, cache_images=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predict_only:
        if args.checkpoint is None:
            args.checkpoint = args.output_dir / "best_model.pt"
        predict(args, args.checkpoint)
        return

    checkpoint = train(args)
    if not args.skip_predict:
        predict(args, checkpoint)


if __name__ == "__main__":
    main()
