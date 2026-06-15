# SpecGateNet

**SpecGateNet** (Spectrum-Guided Fusion Network, 频谱引导融合网络) is a dual-branch
CNN–Transformer network for **cloud and cloud-shadow segmentation** in optical
remote-sensing imagery, designed from a **frequency-domain perspective**.

The Fourier amplitude spectrum of deep features encodes how energy is distributed across
spatial frequencies: high-frequency components correspond to boundaries and texture (thin-cloud
edges, cloud-shadow contours), while low-frequency components correspond to smooth regions
(thick-cloud interiors, clear sky). SpecGateNet turns this amplitude-spectrum response into a
**structural prior** and couples it with a learnable gate to produce a spatially adaptive
fusion signal — addressing two weaknesses of prior dual-branch methods: insufficient
cross-branch interaction in the encoder, and decoder fusion that treats every spatial location
identically.

## Highlights

- **Dual-branch encoder** — ResNet-50 (stride-1 stem, higher-resolution local detail) +
  Swin Transformer (long-range context).
- **Cross-Feature Fusion (CFF)** — explicit bidirectional CNN ↔ Transformer interaction across
  the first three encoder stages (CBAM-enhanced).
- **Frequency-domain DynamicFilter bottleneck** — input-adaptive FFT filtering that gives the
  bottleneck a global receptive field.
- **Spectrum-Guided Fusion decoder (SGF)** — the "SpecGate": the amplitude-spectrum energy map
  generates a spatial gate that preserves low-level detail in high-frequency regions and relies
  on semantics in low-frequency regions, applied across three cascaded decoder levels.
- **PooledAFT** — Attention-Free-Transformer attention after the first SGF level for long-range
  spatial dependency.
- **RGB-only** — no infrared / auxiliary bands required.

The default configuration takes a `3×224×224` RGB image and outputs an `n_classes×224×224`
segmentation map (~40.8 M parameters with a ResNet-50 backbone).

## Repository structure

```
SpecGateNet/
├── main.py                 # entry point: build model + train (with per-epoch validation)
├── train.py                # training loop (AMP, poly LR, optional aux loss)
├── val.py                  # validation metrics (PA / MPA / mIoU / per-class P,R,F1)
├── predict.py              # inference / visualization on a folder of images
├── dataset.py              # HyDataset: RGB image + RGB mask loader
├── utils.py                # label parsing, one-hot, metrics, LR scheduler
├── models/
│   └── specgatenet.py      # the SpecGateNet model (self-contained)
├── datasets/               # class_dict.csv per dataset + layout docs (no raw data)
├── requirements.txt
└── LICENSE                 # MIT
```

## Installation

```bash
conda create -n specgatenet python=3.10
conda activate specgatenet
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. The code was developed with
`torch==2.5.0` / `torchvision==0.20.0` (CUDA 11.8).

## Data preparation

Raw imagery is **not** included. Arrange each dataset as below and see
[`datasets/README.md`](datasets/README.md) for details and download pointers.

```
datasets/<NAME>/
├── class_dict.csv
├── train/{img,label}/*.png
└── val/{img,label}/*.png
```

`class_dict.csv` files for `SPARCS` (7 classes), `CloudSEN12` (3 classes) and
`38-Cloud` (2 classes) are provided.

## Training

Edit the `params` block at the bottom of `main.py` (dataset name, number of classes,
batch size, learning rate, …), then run:

```bash
python main.py
```

Key settings (`--data_name` must match a folder in `datasets/`, and `--num_classes`
must match its `class_dict.csv`):

| Argument          | Default        | Description                              |
|-------------------|----------------|------------------------------------------|
| `--data_root`     | `datasets`     | root containing the dataset folders      |
| `--data_name`     | `SPARCS`       | dataset folder name                      |
| `--num_classes`   | `7`            | number of classes (rows in class_dict)   |
| `--crop_height/width` | `224`      | input resolution                         |
| `--batch_size`    | `16`           | training batch size                      |
| `--learning_rate` | `5e-5`         | base LR (poly decay)                     |
| `--num_epochs`    | `300`          | total epochs                             |

Validation runs every epoch; checkpoints are saved to
`checkpoints/<data_name>/<model_name>/<timestamp>/` and metric logs to `result/`.
Training curves are written for TensorBoard (`runs/`).

## Inference

```bash
python predict.py --checkpoint_path <path/to/checkpoint.pth> \
                  --demo_root demo --demo_name <set> \
                  --num_classes 7 --csv_path datasets/SPARCS/class_dict.csv
```

Color-coded prediction masks are written under `<demo_root>/<demo_name>/<model_name>/`.

## Use the model directly

```python
import torch
from models.specgatenet import SpecGateNet

model = SpecGateNet(n_classes=7, size=224, cnn_pretrained=True)
y = model(torch.randn(1, 3, 224, 224))   # -> [1, 7, 224, 224]
```

## Results

All results use **RGB-only** input (no auxiliary bands). SpecGateNet reaches the best mIoU on
three public benchmarks, surpassing CNN, Transformer and hybrid baselines:

| Dataset      | Task                         | mIoU    | Δ vs. 2nd-best |
|--------------|------------------------------|---------|----------------|
| CloudSEN-12  | cloud / shadow (3-class)     | 77.80%  | +1.33          |
| SPARCS-Val   | 7-class cloud & shadow       | 76.75%  | +5.26          |
| 38-Cloud     | cloud / background (2-class) | 93.30%  | +1.44          |

Ablations attribute **70.1%** of the total gain to the spectral modules, with SGF contributing
the largest single-module improvement (**+1.22%** mIoU).

## Citation

If you find this project useful, please consider citing it:

```bibtex
@misc{specgatenet,
  title  = {SpecGateNet: A Spectrum-Guided Fusion Network for Cloud and Cloud-Shadow Segmentation},
  author = {Kingsley},
  year   = {2026},
  howpublished = {\url{https://github.com/<your-account>/SpecGateNet}}
}
```

## Author

- **Kingsley** — 940714688@qq.com

## License

Released under the [MIT License](LICENSE).
