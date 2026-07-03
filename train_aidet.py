import argparse
import csv
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


LABELS = {"REAL": 0, "FAKE": 1}
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def image_to_tensor(path: Path, train: bool) -> torch.Tensor:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

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
    return (tensor - MEAN) / STD


class AIDetTrainDataset(Dataset):
    def __init__(self, data_dir: Path, train: bool = True) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.train = train
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
        return image_to_tensor(path, self.train), torch.tensor(label, dtype=torch.long)


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
) -> Metrics:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
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

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return Metrics(total_loss / total, correct / total)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    full_train = AIDetTrainDataset(args.data_dir, train=True)
    full_val = AIDetTrainDataset(args.data_dir, train=False)
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
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    print(f"Train images: {len(train_indices)} | Val images: {len(val_indices)}")
    return train_loader, val_loader


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = make_loaders(args)
    model = SmallResNet(width=args.width).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc = -1.0
    best_path = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            val_metrics = run_epoch(model, val_loader, criterion, device)
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
                        "model": model.state_dict(),
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
    model = SmallResNet(width=width).to(device)
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--submission-path", type=Path, default=Path("submission.csv"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
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
