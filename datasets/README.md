# Datasets

This repository does **not** ship raw imagery. Place each dataset under this
folder using the layout below, then point `--data_root datasets --data_name <NAME>`
at it (see the project [README](../README.md)).

## Expected layout

```
datasets/
└── <NAME>/                 # e.g. SPARCS, CloudSEN12, 38-Cloud
    ├── class_dict.csv      # provided here (RGB color -> class)
    ├── train/
    │   ├── img/            # *.png RGB images
    │   └── label/          # *.png RGB masks colored per class_dict.csv
    └── val/
        ├── img/
        └── label/
```

* Images and labels are matched by **sorted filename order** (`img/*.png` ↔ `label/*.png`).
* Labels are RGB PNGs; each pixel color must match a row in `class_dict.csv`.
  They are converted to one-hot internally (`utils.one_hot_it`).
* `--num_classes` in training **must equal** the number of rows in `class_dict.csv`.

## class_dict.csv (already included)

| Dataset    | #Classes | Notes                                              |
|------------|----------|----------------------------------------------------|
| `SPARCS`   | 7        | cloud / shadow / water / ice-snow / land / flooded |
| `CloudSEN12` | 3      | cloud / shadow / void                              |
| `38-Cloud` | 2        | cloud / void                                       |

## Where to obtain the data

* **SPARCS** — USGS Spatial Procedures for Automated Removal of Cloud and Shadow.
* **CloudSEN12** — Sentinel-2 cloud/cloud-shadow benchmark.
* **38-Cloud** — Landsat 8 cloud segmentation dataset.

Download from the respective official sources, convert the masks to the RGB
color scheme in the matching `class_dict.csv`, and arrange them as shown above.

## Split manifests

Exact filename-level split manifests are provided under
[`datasets/splits/`](splits/). They are CSV files that list the image filename,
label filename, split assignment, source scene ID, original patch ID, and
subpatch indices where applicable.

The reported results use the rows where `used_for_reported_metrics=true`, i.e.
the `train` and `val` folders. The `val` split is the held-out evaluation subset
used for checkpoint selection and metric reporting. Rows marked as
`unused_rounding` are leftovers from the integer split operation and are not
used in the reported metrics.

Protocol details:

| Dataset | Reported train samples | Reported held-out samples | Split unit |
| --- | ---: | ---: | --- |
| `CloudSEN12` | 45,621 | 5,067 | 509x509 original patch before 3x3 subpatch cropping |
| `SPARCS` | 1,800 | 200 | Fixed 224x224 patch-level split |
| `38-Cloud` | 4,639 | 515 | Fixed 224x224 patch-level split |

For CloudSEN-12, all nine 224x224 subpatches from a 509x509 original patch stay
within the same subset. SPARCS-Val and 38-Cloud use fixed patch-level controlled
splits for fair method comparison under the same protocol; they should not be
interpreted as scene-level generalization splits.

To regenerate the manifests from prepared local folders:

```bash
python tools/build_split_manifests.py \
  --datasets-root /path/to/prepared/datasets \
  --sparcs-name SPARCS-val \
  --out-dir datasets/splits
```
