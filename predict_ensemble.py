import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_aidet import AIDetTestDataset, load_checkpoint, make_model, select_device, validate_rows


def load_model(path: Path, device: torch.device) -> nn.Module:
    checkpoint = load_checkpoint(path, device)
    args = checkpoint.get("args", {})
    model = make_model(
        str(args.get("model_version", "v1")),
        int(args.get("width", 48)),
        float(args.get("dropout", 0.2)),
        False,
        str(args.get("color_space", "ycbcr")),
        int(args.get("input_size", 224)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def tta_margin(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    logits = (model(images) + model(images.flip(-1))) * 0.5
    return logits[:, 1] - logits[:, 0]


@torch.no_grad()
def predict(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    ycbcr_model = load_model(args.ycbcr_checkpoint, device)
    rgb_model = load_model(args.rgb_checkpoint, device)
    resnet_model = load_model(args.resnet_checkpoint, device)

    dataset = AIDetTestDataset(args.data_dir / "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    rows: list[tuple[int, int]] = []
    for images, ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            score = (
                tta_margin(ycbcr_model, images)
                + args.rgb_weight * tta_margin(rgb_model, images)
                + args.resnet_weight * tta_margin(resnet_model, images)
            )
        predictions = (score > args.bias).long().cpu().tolist()
        rows.extend((int(image_id), label) for image_id, label in zip(ids.tolist(), predictions))

    rows.sort(key=lambda item: item[0])
    validate_rows(rows)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ensemble predictions to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with the calibrated three-model ensemble.")
    parser.add_argument("--data-dir", type=Path, default=Path("img_AIdet/data"))
    parser.add_argument("--ycbcr-checkpoint", type=Path, default=Path("outputs_effnet_ycbcr/best_model.pt"))
    parser.add_argument("--rgb-checkpoint", type=Path, default=Path("outputs_effnet_rgb_seed42/best_model.pt"))
    parser.add_argument("--resnet-checkpoint", type=Path, default=Path("outputs_v2/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("submission_ensemble.csv"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rgb-weight", type=float, default=1.94)
    parser.add_argument("--resnet-weight", type=float, default=0.64)
    parser.add_argument("--bias", type=float, default=0.194)
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
