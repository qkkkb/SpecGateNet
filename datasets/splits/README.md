# Split Manifests

This directory contains the filename-level split manifests used for the reported
SpecGateNet experiments. The CSV files list image and label filenames only; raw
remote-sensing imagery is not included in this repository.

## Files

| File | Content |
| --- | --- |
| `CloudSEN12_split_manifest.csv` | CloudSEN-12 train, held-out evaluation, and unused rounding samples. |
| `SPARCS_split_manifest.csv` | SPARCS-Val fixed patch-level train and held-out evaluation samples. |
| `38-Cloud_split_manifest.csv` | 38-Cloud fixed patch-level train, held-out evaluation, and unused rounding samples. |
| `split_summary.csv` | Per-dataset sample and source-group counts. |

## Reported Evaluation Protocol

The metrics reported in the paper use the rows where
`used_for_reported_metrics=true`, namely the `train` and `val` folders. The
`val` split is the held-out evaluation subset used for checkpoint selection and
metric reporting. Rows marked as `unused_rounding` are leftover samples from the
integer split operation and are not used for the reported metrics.

CloudSEN-12 is split before the 3x3 subpatch crop: all nine 224x224 subpatches
from the same 509x509 original patch remain in the same subset. SPARCS-Val and
38-Cloud use fixed patch-level controlled splits; they are intended for
method-to-method comparison under the same protocol rather than scene-level
generalization evaluation.

For CloudSEN-12, `source_scene_id` is set to the full 509x509 original-patch
identifier, which is also the split unit. For SPARCS-Val and 38-Cloud,
`source_scene_id` denotes the Landsat scene identifier parsed from the filename.

## Regeneration

If the prepared datasets are arranged locally as described in
`datasets/README.md`, the manifests can be regenerated with:

```bash
python tools/build_split_manifests.py \
  --datasets-root /path/to/prepared/datasets \
  --sparcs-name SPARCS-val \
  --out-dir datasets/splits
```
