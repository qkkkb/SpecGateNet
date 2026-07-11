# SpecGateNet

SpecGateNet is a dual-branch CNN-Transformer network for cloud and
cloud-shadow segmentation in optical remote-sensing imagery. The repository
contains the checkpoint-compatible model used for the manuscript experiments,
the filename-level data split manifests, and the profiling, training, and
evaluation utilities needed to audit the reported configuration.

## Architecture

The model uses RGB input and combines:

- a ResNet-50 branch for local spatial features;
- a hierarchical Swin Transformer branch for contextual features;
- three Cross-Feature Fusion (CFF) modules for bidirectional encoder interaction;
- an input-adaptive frequency-domain DynamicFilter at the bottleneck;
- PooledAFT for pooled long-range dependency modeling; and
- three Spectrum-Guided Fusion (SGF) stages in the decoder.

SGF computes an amplitude-derived, phase-free spectral-energy cue from the
aligned high-level feature. This cue is not treated as a location-aligned
boundary map. A lightweight convolutional gate is learned end to end from the
cue and is used to modulate high- and low-level feature fusion.

`models/specgatenet.py` is the single canonical model implementation. It
strictly loads the original experiment checkpoints for SPARCS-Val,
CloudSEN-12, and 38-Cloud.

## Verified Complexity

The following values were measured with `tools/profile_model.py` using one
224x224 RGB input. PyTorch parameter counts include all model parameters;
THOP parameter and operation counts are reported separately because THOP does
not count every parameter used by custom operations.

| Classes | Dataset setting | PyTorch params | THOP params | THOP FLOPs |
| ---: | --- | ---: | ---: | ---: |
| 7 | SPARCS-Val | 40.818324 M | 38.884429 M | 27.051236 G |
| 3 | CloudSEN-12 | 40.818064 M | 38.884169 M | 27.038391 G |
| 2 | 38-Cloud | 40.817999 M | 38.884104 M | 27.035179 G |

Reproduce a profile with:

```bash
python tools/profile_model.py --num-classes 7 --size 224 --device cpu
```

## Repository Structure

```text
SpecGateNet/
|-- models/
|   |-- __init__.py
|   `-- specgatenet.py
|-- datasets/
|   |-- README.md
|   `-- splits/
|-- tools/
|   |-- build_split_manifests.py
|   |-- prepare_patch_dataset.py
|   |-- profile_model.py
|   |-- train_specgatenet.py
|   `-- evaluate_specgatenet.py
|-- dataset.py
|-- train.py
|-- val.py
|-- predict.py
|-- requirements.txt
`-- LICENSE
```

## Installation

```bash
conda create -n specgatenet python=3.10
conda activate specgatenet
pip install -r requirements.txt
```

The code was verified with `torch==2.5.0` and `torchvision==0.20.0`.
A CUDA-capable GPU is required by the provided training loop.

## Data Preparation

Raw imagery is not redistributed. Arrange each prepared dataset as follows:

```text
datasets/<NAME>/
|-- class_dict.csv
|-- train/
|   |-- img/*.png
|   `-- label/*.png
`-- val/
    |-- img/*.png
    `-- label/*.png
```

The exact filename-level split manifests used by the reported protocol are in
`datasets/splits/`. Dataset download pointers and class-label details are in
`datasets/README.md`.

If a prepared source directory contains the relative files listed by a
manifest, reconstruct the corresponding layout with:

```bash
python tools/prepare_patch_dataset.py apply-manifest \
  --manifest datasets/splits/SPARCS_split_manifest.csv \
  --source-root /path/to/prepared/source \
  --output-root datasets
```

The manifests can also be regenerated and audited with:

```bash
python tools/build_split_manifests.py \
  --datasets-root /path/to/prepared/datasets \
  --sparcs-name SPARCS-val \
  --out-dir datasets/splits
```

## Training

The manuscript configuration uses 224x224 RGB inputs, batch size 16, AdamW,
an initial learning rate of `5e-5`, poly decay with power 2, 300 epochs,
ImageNet-pretrained ResNet-50 residual stages, no auxiliary loss, and no TTA.

```bash
python tools/train_specgatenet.py \
  --data-root datasets \
  --data-name SPARCS \
  --num-classes 7 \
  --batch-size 16 \
  --learning-rate 5e-5 \
  --num-epochs 300
```

Use `--no-cnn-pretrained` only when a randomly initialized ResNet branch is
intended. When resuming, `--pretrained-model-path` loads the checkpoint with
strict state-dict validation.

## Evaluation

```bash
python tools/evaluate_specgatenet.py \
  --checkpoint /path/to/checkpoint.pth \
  --data-root datasets \
  --data-name SPARCS \
  --num-classes 7 \
  --device cuda
```

The evaluation command strictly validates checkpoint compatibility before
computing pixel accuracy, mean pixel accuracy, mIoU, and per-class IoU on the
prepared held-out split.

## Direct Use

```python
import torch
from models import SpecGateNet

model = SpecGateNet(
    n_classes=7,
    size=224,
    cnn_pretrained=False,
    aux_loss=False,
)
output = model(torch.randn(1, 3, 224, 224))
print(output.shape)  # torch.Size([1, 7, 224, 224])
```

## Results

All reported experiments use three-channel RGB input.

| Dataset | Task | mIoU |
| --- | --- | ---: |
| CloudSEN-12 | Cloud / cloud shadow / background | 77.80% |
| SPARCS-Val | Seven-class cloud and cloud-shadow segmentation | 76.75% |
| 38-Cloud | Cloud / background | 93.30% |

## License

Released under the [MIT License](LICENSE).
