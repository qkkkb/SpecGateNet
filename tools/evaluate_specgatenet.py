#!/usr/bin/env python3
"""Evaluate a SpecGateNet checkpoint on a prepared held-out split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import HyDataset  # noqa: E402
from models import SpecGateNet  # noqa: E402


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)


def update_confusion(
    confusion: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> None:
    mask = (target >= 0) & (target < num_classes)
    values = num_classes * target[mask] + prediction[mask]
    confusion += torch.bincount(
        values,
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--data-name", default="SPARCS")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    dataset_root = args.data_root / args.data_name
    dataset = HyDataset(
        data_root=str(dataset_root / args.split),
        csv_path=str(dataset_root / "class_dict.csv"),
        scale=(args.size, args.size),
        mode="val",
        aux_bands=[],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    model = SpecGateNet(
        size=args.size,
        n_classes=args.num_classes,
        cnn_pretrained=False,
        aux_loss=False,
    ).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)

    confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
    with torch.inference_mode():
        for image, label, _ in loader:
            logits = model(image.to(device, non_blocking=True))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            prediction = logits.argmax(dim=1).cpu()
            target = label.argmax(dim=1).to(dtype=torch.int64)
            update_confusion(confusion, prediction, target, args.num_classes)

    true_positive = confusion.diag().to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    actual = confusion.sum(dim=1).to(torch.float64)
    union = actual + predicted - true_positive
    iou = true_positive / union.clamp_min(1)
    recall = true_positive / actual.clamp_min(1)
    pixel_accuracy = true_positive.sum() / confusion.sum().clamp_min(1)

    print("model=SpecGateNet")
    print(f"dataset={args.data_name}/{args.split}")
    print(f"samples={len(dataset)}")
    print(f"PA={pixel_accuracy.item() * 100:.4f}")
    print(f"MPA={recall.mean().item() * 100:.4f}")
    print(f"mIoU={iou.mean().item() * 100:.4f}")
    print("per_class_iou=" + ",".join(f"{value.item() * 100:.4f}" for value in iou))


if __name__ == "__main__":
    main()
