#!/usr/bin/env python3
"""Utility helpers for preparing patch datasets used by SpecGateNet.

This script is intentionally generic and path-parameterized. It does not depend
on local lab paths and does not include raw imagery.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_images(path: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def crop_grid(args: argparse.Namespace) -> None:
    ensure_dir(args.output_image_dir)
    ensure_dir(args.output_label_dir)
    label_by_stem = {item.stem: item for item in iter_images(args.label_dir)}
    step = args.crop_size - args.overlap
    if step <= 0:
        raise ValueError("--overlap must be smaller than --crop-size")

    for image_path in iter_images(args.image_dir):
        label_path = label_by_stem.get(image_path.stem)
        if label_path is None:
            raise FileNotFoundError(f"Missing label for {image_path.name}")
        image = Image.open(image_path)
        label = Image.open(label_path)
        for row in range(args.rows):
            for col in range(args.cols):
                left = col * step
                top = row * step
                box = (left, top, left + args.crop_size, top + args.crop_size)
                out_name = f"{image_path.stem}_{row}_{col}.png"
                image.crop(box).save(args.output_image_dir / out_name)
                label.crop(box).save(args.output_label_dir / out_name)


def apply_manifest(args: argparse.Namespace) -> None:
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("used_for_reported_metrics") != "true" and not args.include_unused:
                continue
            split = row["split"]
            image_src = args.source_root / row["image_relpath"]
            label_src = args.source_root / row["label_relpath"]
            image_dst = args.output_root / row["dataset"] / split / "img" / row["image_filename"]
            label_dst = args.output_root / row["dataset"] / split / "label" / row["label_filename"]
            ensure_dir(image_dst.parent)
            ensure_dir(label_dst.parent)
            shutil.copy2(image_src, image_dst)
            shutil.copy2(label_src, label_dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    crop = subparsers.add_parser("crop-grid", help="Crop image/label pairs with a regular grid.")
    crop.add_argument("--image-dir", type=Path, required=True)
    crop.add_argument("--label-dir", type=Path, required=True)
    crop.add_argument("--output-image-dir", type=Path, required=True)
    crop.add_argument("--output-label-dir", type=Path, required=True)
    crop.add_argument("--crop-size", type=int, default=224)
    crop.add_argument("--overlap", type=int, default=0)
    crop.add_argument("--rows", type=int, default=3)
    crop.add_argument("--cols", type=int, default=3)
    crop.set_defaults(func=crop_grid)

    manifest = subparsers.add_parser("apply-manifest", help="Copy samples listed in a split manifest.")
    manifest.add_argument("--manifest", type=Path, required=True)
    manifest.add_argument("--source-root", type=Path, required=True)
    manifest.add_argument("--output-root", type=Path, required=True)
    manifest.add_argument("--include-unused", action="store_true")
    manifest.set_defaults(func=apply_manifest)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
