#!/usr/bin/env python3
"""Train the checkpoint-compatible SpecGateNet configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import HyDataset  # noqa: E402
from models import SpecGateNet  # noqa: E402
from train import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_workers = 2 if os.name == "nt" else 8
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--data-name", default="SPARCS")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--crop-height", type=int, default=224)
    parser.add_argument("--crop-width", type=int, default=224)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epoch-start-i", type=int, default=0)
    parser.add_argument("--checkpoint-step", type=int, default=30)
    parser.add_argument("--validation-step", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=default_workers)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--pretrained-model-path", type=Path, default=None)
    parser.add_argument("--model-name", default="SpecGateNet")
    parser.add_argument("--no-cnn-pretrained", action="store_true")
    parser.add_argument("--enable-demo", action="store_true")
    parser.add_argument("--demo-root", default="demo")
    parser.add_argument("--demo-name", default="demo")
    parser.add_argument("--use-wandb", action="store_true")
    return parser.parse_args()


def make_loader(
    dataset: HyDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4 if (os.cpu_count() or 0) > 4 else 2
    return DataLoader(dataset, **kwargs)


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


def main() -> None:
    args = parse_args()
    if args.crop_height != args.crop_width:
        raise ValueError("SpecGateNet reproduction expects square inputs.")

    args.aux_bands = []
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if not torch.cuda.is_available():
        raise RuntimeError("The provided training loop requires a CUDA-capable GPU.")

    dataset_root = args.data_root / args.data_name
    csv_path = dataset_root / "class_dict.csv"
    scale = (args.crop_height, args.crop_width)
    dataset_train = HyDataset(
        data_root=str(dataset_root / "train"),
        csv_path=str(csv_path),
        scale=scale,
        mode="train",
        aux_bands=[],
    )
    dataset_val = HyDataset(
        data_root=str(dataset_root / "val"),
        csv_path=str(csv_path),
        scale=scale,
        mode="val",
        aux_bands=[],
    )
    dataloader_train = make_loader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_batch_size = args.val_batch_size if args.num_classes == 3 else 1
    dataloader_val = make_loader(
        dataset_val,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SpecGateNet(
        size=args.crop_height,
        n_classes=args.num_classes,
        cnn_pretrained=not args.no_cnn_pretrained,
        aux_loss=False,
    ).cuda()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if args.pretrained_model_path is not None:
        checkpoint = torch.load(
            args.pretrained_model_path,
            map_location="cuda",
            weights_only=True,
        )
        model.load_state_dict(extract_state_dict(checkpoint), strict=True)
        print(f"Loaded checkpoint: {args.pretrained_model_path}")

    run_time = time.strftime("%Y_%m_%d_%H_%M_", time.localtime())
    (Path("checkpoints") / args.data_name / args.model_name / run_time).mkdir(
        parents=True,
        exist_ok=True,
    )
    Path("result").mkdir(exist_ok=True)
    if args.enable_demo:
        (Path(args.demo_root) / args.model_name / run_time).mkdir(parents=True, exist_ok=True)

    train(
        args,
        model,
        dataloader_train,
        dataloader_val,
        str(csv_path),
        run_time,
        aux_loss=False,
    )


if __name__ == "__main__":
    main()
