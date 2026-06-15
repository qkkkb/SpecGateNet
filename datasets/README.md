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
