#!/usr/bin/env python3
"""Build filename-level split manifests for the SpecGateNet datasets.

The script reads already prepared ``train``/``val``/``test`` folders and writes
CSV manifests that make the exact experimental split auditable without
shipping raw imagery.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ParsedName:
    source_scene_id: str
    original_patch_id: str
    subpatch_row: str = ""
    subpatch_col: str = ""
    notes: str = ""


@dataclass(frozen=True)
class DatasetConfig:
    public_name: str
    default_local_name: str
    parser: Callable[[str], ParsedName]
    label_key: Callable[[str], str]


def stem(path_or_name: str) -> str:
    return Path(path_or_name).stem


def default_label_key(name: str) -> str:
    return stem(name)


def cloudsen_label_key(name: str) -> str:
    return stem(name).replace("_label_", "_")


def parse_cloudsen12(name: str) -> ParsedName:
    base = stem(name)
    match = re.match(r"(.+)_([0-2])_([0-2])$", base)
    if not match:
        return ParsedName(source_scene_id=base, original_patch_id=base)
    original_patch_id, row, col = match.groups()
    return ParsedName(
        source_scene_id=original_patch_id,
        original_patch_id=original_patch_id,
        subpatch_row=row,
        subpatch_col=col,
    )


def parse_sparcs(name: str) -> ParsedName:
    base = stem(name)
    match = re.match(r"(.+)_([0-9]+)_([0-9]+)$", base)
    if not match:
        return ParsedName(source_scene_id=base.split("_")[0], original_patch_id=base)
    original_patch_id, row, col = match.groups()
    scene_id = original_patch_id.split("_")[0]
    return ParsedName(
        source_scene_id=scene_id,
        original_patch_id=original_patch_id,
        subpatch_row=row,
        subpatch_col=col,
    )


def parse_38cloud(name: str) -> ParsedName:
    base = stem(name)
    parts = base.split("_")
    scene_id = ""
    for index, part in enumerate(parts):
        if part.startswith("LC08") or part.startswith("LC8"):
            scene_id = "_".join(parts[index:])
            break
    return ParsedName(source_scene_id=scene_id, original_patch_id=base)


CONFIGS = {
    "CloudSEN12": DatasetConfig(
        public_name="CloudSEN12",
        default_local_name="CloudSEN12",
        parser=parse_cloudsen12,
        label_key=cloudsen_label_key,
    ),
    "SPARCS": DatasetConfig(
        public_name="SPARCS",
        default_local_name="SPARCS",
        parser=parse_sparcs,
        label_key=default_label_key,
    ),
    "38-Cloud": DatasetConfig(
        public_name="38-Cloud",
        default_local_name="38-Cloud",
        parser=parse_38cloud,
        label_key=default_label_key,
    ),
}


def split_role(split: str) -> tuple[str, bool]:
    if split == "train":
        return "train", True
    if split == "val":
        return "heldout_evaluation", True
    return "unused_rounding", False


def list_pngs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".png")


def pair_images_and_labels(dataset_root: Path, split: str, config: DatasetConfig) -> list[tuple[Path, Path]]:
    image_dir = dataset_root / split / "img"
    label_dir = dataset_root / split / "label"
    images = list_pngs(image_dir)
    labels = list_pngs(label_dir)
    label_by_key = {config.label_key(label.name): label for label in labels}
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for image in images:
        key = default_label_key(image.name)
        label = label_by_key.get(key)
        if label is None:
            missing.append(image.name)
        else:
            pairs.append((image, label))
    extra = sorted(set(label_by_key) - {default_label_key(image.name) for image in images})
    if missing or extra:
        message = [f"Image/label mismatch in {dataset_root.name}/{split}"]
        if missing:
            message.append(f"missing labels for {len(missing)} images, e.g. {missing[:3]}")
        if extra:
            message.append(f"extra labels for {len(extra)} keys, e.g. {extra[:3]}")
        raise RuntimeError("; ".join(message))
    return pairs


def build_rows(dataset_root: Path, config: DatasetConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in SPLITS:
        pairs = pair_images_and_labels(dataset_root, split, config)
        role, used = split_role(split)
        for image, label in pairs:
            parsed = config.parser(image.name)
            rows.append(
                {
                    "dataset": config.public_name,
                    "split": split,
                    "reported_subset": role,
                    "used_for_reported_metrics": str(used).lower(),
                    "image_filename": image.name,
                    "label_filename": label.name,
                    "source_scene_id": parsed.source_scene_id,
                    "original_patch_id": parsed.original_patch_id,
                    "subpatch_row": parsed.subpatch_row,
                    "subpatch_col": parsed.subpatch_col,
                    "image_relpath": f"{config.public_name}/{split}/img/{image.name}",
                    "label_relpath": f"{config.public_name}/{split}/label/{label.name}",
                    "notes": parsed.notes,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary_rows: list[dict[str, str]] = []
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for split in SPLITS:
            split_rows = [row for row in dataset_rows if row["split"] == split]
            if not split_rows:
                continue
            summary_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "reported_subset": split_rows[0]["reported_subset"],
                    "used_for_reported_metrics": split_rows[0]["used_for_reported_metrics"],
                    "samples": str(len(split_rows)),
                    "original_patches": str(len({row["original_patch_id"] for row in split_rows})),
                    "source_scenes": str(len({row["source_scene_id"] for row in split_rows if row["source_scene_id"]})),
                }
            )
    return summary_rows


def check_overlaps(rows: list[dict[str, str]]) -> list[str]:
    messages: list[str] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        by_split = {split: [row for row in dataset_rows if row["split"] == split] for split in SPLITS}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if not by_split[left] or not by_split[right]:
                continue
            left_images = {row["image_filename"] for row in by_split[left]}
            right_images = {row["image_filename"] for row in by_split[right]}
            exact = left_images & right_images
            if exact:
                raise RuntimeError(f"{dataset}: exact image overlap between {left} and {right}: {sorted(exact)[:5]}")
            original_overlap = {row["original_patch_id"] for row in by_split[left]} & {
                row["original_patch_id"] for row in by_split[right]
            }
            scene_overlap = {row["source_scene_id"] for row in by_split[left] if row["source_scene_id"]} & {
                row["source_scene_id"] for row in by_split[right] if row["source_scene_id"]
            }
            messages.append(
                f"{dataset} {left}-{right}: exact=0, original_patch_overlap={len(original_overlap)}, "
                f"source_scene_overlap={len(scene_overlap)}"
            )
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"), help="Root containing dataset folders.")
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/splits"), help="Output directory for CSV files.")
    parser.add_argument("--cloudsen12-name", default="CloudSEN12", help="Local folder name for CloudSEN12.")
    parser.add_argument("--sparcs-name", default="SPARCS", help="Local folder name for SPARCS.")
    parser.add_argument("--cloud38-name", default="38-Cloud", help="Local folder name for 38-Cloud.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_names = {
        "CloudSEN12": args.cloudsen12_name,
        "SPARCS": args.sparcs_name,
        "38-Cloud": args.cloud38_name,
    }
    fieldnames = [
        "dataset",
        "split",
        "reported_subset",
        "used_for_reported_metrics",
        "image_filename",
        "label_filename",
        "source_scene_id",
        "original_patch_id",
        "subpatch_row",
        "subpatch_col",
        "image_relpath",
        "label_relpath",
        "notes",
    ]
    all_rows: list[dict[str, str]] = []
    for dataset_key, config in CONFIGS.items():
        dataset_root = args.datasets_root / local_names[dataset_key]
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")
        rows = build_rows(dataset_root, config)
        if not rows:
            raise RuntimeError(f"No PNG samples found for {config.public_name} under {dataset_root}")
        all_rows.extend(rows)
        write_csv(args.out_dir / f"{config.public_name}_split_manifest.csv", rows, fieldnames)

    summary_rows = summarize(all_rows)
    write_csv(
        args.out_dir / "split_summary.csv",
        summary_rows,
        [
            "dataset",
            "split",
            "reported_subset",
            "used_for_reported_metrics",
            "samples",
            "original_patches",
            "source_scenes",
        ],
    )
    for message in check_overlaps(all_rows):
        print(message)
    print(f"Wrote manifests to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
