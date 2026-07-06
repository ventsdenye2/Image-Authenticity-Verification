import argparse
import csv
import os
import random
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Direct-run defaults for OpenL. If the platform starts this file without
# command-line arguments, these are the training settings that will be used.
DEFAULT_DATASET_NAME = "ML-img"
DEFAULT_MODELS = "convnext_tiny,efficientnet_b0,resnet34"
DEFAULT_SEEDS = "42"
DEFAULT_EPOCHS = 24
DEFAULT_BATCH_SIZE = 512
DEFAULT_IMG_SIZE = 224
DEFAULT_WORKERS = 8
DEFAULT_VAL_RATIO = 0.08
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_MIXUP_ALPHA = 0.15
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_THRESHOLD = 0.5
DEFAULT_EXPECTED_TEST_COUNT = 20000


@dataclass
class Sample:
    path: Path
    label: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_files(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def is_data_root(path: Path) -> bool:
    return (
        (path / "train" / "REAL").is_dir()
        and (path / "train" / "FAKE").is_dir()
        and (path / "test").is_dir()
    )


def find_existing_data_root(candidates: Sequence[Path]) -> Optional[Path]:
    checked = []
    for base in candidates:
        if not base or not base.exists():
            continue
        base = base.resolve()
        checked.append(base)
        direct_candidates = [
            base,
            base / "data",
            base / "img_AIdet" / "data",
        ]
        for candidate in direct_candidates:
            if is_data_root(candidate):
                return candidate
        for root, dirs, _ in os.walk(base):
            root_path = Path(root)
            if is_data_root(root_path):
                return root_path
            if len(root_path.relative_to(base).parts) > 4:
                dirs[:] = []
    print("Checked data candidates:", ", ".join(str(p) for p in checked))
    return None


def extract_zip_archives(base: Path, extract_dir: Path) -> List[Path]:
    extracted_roots = []
    if not base.exists():
        return extracted_roots
    zips = sorted(p for p in base.rglob("*.zip") if p.is_file())
    for archive in zips:
        target = extract_dir / archive.stem
        marker = target / ".extract_done"
        if marker.exists():
            extracted_roots.append(target)
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"Extracting dataset archive: {archive} -> {target}")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target)
        marker.write_text("ok", encoding="utf-8")
        extracted_roots.append(target)
    return extracted_roots


def prepare_openl_context():
    try:
        from c2net.context import prepare

        return prepare()
    except Exception as exc:
        print(f"c2net prepare() is unavailable, falling back to local paths: {exc}")
        return None


def upload_openl_output() -> None:
    try:
        from c2net.context import upload_output

        upload_output()
    except Exception as exc:
        print(f"upload_output() skipped: {exc}")


def resolve_paths(args) -> Tuple[Path, Path]:
    c2net_context = None if args.no_c2net else prepare_openl_context()
    output_dir = Path(args.output_dir) if args.output_dir else None
    candidates: List[Path] = []

    if args.data_root:
        candidates.append(Path(args.data_root))

    if c2net_context is not None:
        dataset_base = Path(c2net_context.dataset_path)
        output_dir = output_dir or Path(c2net_context.output_path)
        candidates.append(dataset_base / args.dataset_name)
        candidates.append(dataset_base)

    project_local = Path.cwd()
    candidates.extend(
        [
            project_local / "img_AIdet" / "data",
            project_local / "data",
            project_local,
        ]
    )
    output_dir = output_dir or (project_local / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = find_existing_data_root(candidates)
    if data_root is None:
        extract_dir = output_dir / "extracted_dataset"
        extracted_candidates = []
        for candidate in candidates:
            if candidate.exists():
                extracted_candidates.extend(extract_zip_archives(candidate, extract_dir))
        data_root = find_existing_data_root(extracted_candidates)

    if data_root is None:
        raise FileNotFoundError(
            "Could not locate dataset root. Expected train/REAL, train/FAKE and test folders, "
            "or a zip archive containing them."
        )

    print(f"Using data root: {data_root}")
    print(f"Using output dir: {output_dir}")
    return data_root, output_dir


class TrainDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample.path) as img:
            img = img.convert("RGB")
            return self.transform(img), torch.tensor(sample.label, dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, test_dir: Path, transform):
        self.items = []
        for path in image_files(test_dir):
            try:
                image_id = int(path.stem)
            except ValueError:
                continue
            self.items.append((image_id, path))
        self.items.sort(key=lambda x: x[0])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        image_id, path = self.items[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return image_id, self.transform(img)


def build_samples(data_root: Path) -> List[Sample]:
    real = [Sample(p, 0) for p in image_files(data_root / "train" / "REAL")]
    fake = [Sample(p, 1) for p in image_files(data_root / "train" / "FAKE")]
    print(f"Train REAL={len(real)}, FAKE={len(fake)}")
    return real + fake


def stratified_split(samples: Sequence[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)

    train, val = [], []
    for label, label_samples in by_label.items():
        rng.shuffle(label_samples)
        val_count = max(1, int(len(label_samples) * val_ratio))
        val.extend(label_samples[:val_count])
        train.extend(label_samples[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    print(f"Split train={len(train)}, val={len(val)}")
    return train, val


def make_transforms(img_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.12, scale=(0.01, 0.04), ratio=(0.5, 2.0), value="random"),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_tf, eval_tf


def replace_classifier(model: nn.Module, name: str) -> nn.Module:
    if name.startswith("convnext"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
    elif name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
    elif name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def build_model(name: str, pretrained: bool) -> nn.Module:
    name = name.lower()
    constructors = {
        "convnext_tiny": (models.convnext_tiny, getattr(models, "ConvNeXt_Tiny_Weights", None)),
        "efficientnet_b0": (models.efficientnet_b0, getattr(models, "EfficientNet_B0_Weights", None)),
        "resnet34": (models.resnet34, getattr(models, "ResNet34_Weights", None)),
        "resnet18": (models.resnet18, getattr(models, "ResNet18_Weights", None)),
    }
    if name not in constructors:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(constructors)}")

    ctor, weights_enum = constructors[name]
    weights = None
    if pretrained and weights_enum is not None:
        weights = weights_enum.DEFAULT
    try:
        model = ctor(weights=weights)
    except Exception as exc:
        print(f"Failed to load pretrained weights for {name}: {exc}. Retrying without pretrained weights.")
        model = ctor(weights=None)
    return replace_classifier(model, name)


def mixup_batch(images, labels, alpha: float):
    if alpha <= 0:
        return images, labels
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[perm]
    mixed_labels = lam * labels + (1.0 - lam) * labels[perm]
    return mixed_images, mixed_labels


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == labels).float().mean().item()


def train_one_model(
    model_name: str,
    seed: int,
    data_root: Path,
    output_dir: Path,
    args,
    device: torch.device,
) -> Path:
    seed_everything(seed)
    train_tf, eval_tf = make_transforms(args.img_size)
    samples = build_samples(data_root)
    train_samples, val_samples = stratified_split(samples, args.val_ratio, seed)

    train_loader = DataLoader(
        TrainDataset(train_samples, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        TrainDataset(val_samples, eval_tf),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = build_model(model_name, pretrained=not args.no_pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_acc = -1.0
    best_path = output_dir / f"best_{model_name}_seed{seed}.pt"
    log_path = output_dir / "train_log.csv"
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "seed", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss = total_acc = total_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1)
            mixed_images, mixed_labels = mixup_batch(images, labels, args.mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(mixed_images)
                loss = criterion(logits, mixed_labels)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy_from_logits(logits.detach(), labels) * batch_size
            total_count += batch_size

        scheduler.step()
        train_loss = total_loss / total_count
        train_acc = total_acc / total_count
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args.amp)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "seed": seed,
                    "img_size": args.img_size,
                    "val_acc": val_acc,
                },
                best_path,
            )

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start
        print(
            f"[{model_name} seed={seed}] epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} best={best_acc:.4f} "
            f"lr={lr:.2e} time={elapsed:.1f}s"
        )
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [model_name, seed, epoch, train_loss, train_acc, val_loss, val_acc, lr]
            )

    return best_path


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool) -> Tuple[float, float]:
    model.eval()
    total_loss = total_acc = total_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy_from_logits(logits, labels) * batch_size
        total_count += batch_size
    return total_loss / total_count, total_acc / total_count


@torch.no_grad()
def predict_ensemble(
    checkpoint_paths: Sequence[Path],
    data_root: Path,
    output_dir: Path,
    args,
    device: torch.device,
) -> Path:
    _, eval_tf = make_transforms(args.img_size)
    test_dataset = TestDataset(data_root / "test", eval_tf)
    if len(test_dataset) != args.expected_test_count:
        print(f"Warning: expected {args.expected_test_count} test images, found {len(test_dataset)}.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    predictions = {}
    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_name = ckpt["model_name"]
        model = build_model(model_name, pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Predicting with {ckpt_path.name}, val_acc={ckpt.get('val_acc', -1):.4f}")

        for ids, images in test_loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                probs = torch.sigmoid(model(images))
                if args.tta:
                    flipped = torch.flip(images, dims=[3])
                    probs = 0.5 * (probs + torch.sigmoid(model(flipped)))
            for image_id, prob in zip(ids.tolist(), probs.view(-1).detach().cpu().tolist()):
                predictions.setdefault(image_id, []).append(prob)

    rows = []
    missing = []
    for image_id in range(1, args.expected_test_count + 1):
        probs = predictions.get(image_id)
        if not probs:
            missing.append(image_id)
            label = 0
        else:
            label = int((sum(probs) / len(probs)) >= args.threshold)
        rows.append((image_id, label))

    if missing:
        raise RuntimeError(f"Missing predictions for IDs: {missing[:20]} ... total={len(missing)}")

    submission_path = output_dir / "submission.csv"
    with submission_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "label"])
        writer.writerows(rows)
    print(f"Saved submission: {submission_path}")

    latest_path = output_dir / "checkpoints"
    latest_path.mkdir(exist_ok=True)
    for ckpt in checkpoint_paths:
        shutil.copy2(ckpt, latest_path / ckpt.name)
    return submission_path


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in parse_csv_list(text)]


def parse_args():
    parser = argparse.ArgumentParser(description="Train AI-generated image detector and create submission.csv")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="OpenL/OpenI dataset name under c2net dataset_path.")
    parser.add_argument("--data-root", default="", help="Optional direct path containing train/ and test/.")
    parser.add_argument("--output-dir", default="", help="Optional output path. Defaults to c2net output_path on OpenL.")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated model list.")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated seed list.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--mixup-alpha", type=float, default=DEFAULT_MIXUP_ALPHA)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--expected-test-count", type=int, default=DEFAULT_EXPECTED_TEST_COUNT)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-c2net", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.set_defaults(amp=True, tta=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root, output_dir = resolve_paths(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"CUDA: {torch.cuda.get_device_name(0)}")

    checkpoint_paths = []
    for model_name in parse_csv_list(args.models):
        for seed in parse_int_list(args.seeds):
            checkpoint_paths.append(train_one_model(model_name, seed, data_root, output_dir, args, device))

    predict_ensemble(checkpoint_paths, data_root, output_dir, args, device)
    if not args.no_c2net:
        upload_openl_output()


if __name__ == "__main__":
    main()
