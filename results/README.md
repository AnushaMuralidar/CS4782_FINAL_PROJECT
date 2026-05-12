# Results Directory

## Overview

This directory contains training outputs, evaluation metrics, and generated figures for all model variants evaluated in this project.

Each subfolder corresponds to:
- a specific model architecture
- a specific dataset configuration

---

## Folder Naming Convention

## Model Result Folders

| Folder | Description |
|---|---|
| `baseline_s1` | Baseline DLoc model trained on single-scenario dataset |
| `baseline_s2` | Baseline DLoc model trained on multi-scenario dataset |
| `unet_s1` | UNet model trained on single-scenario dataset |
| `unet_s2` | UNet model trained on multi-scenario dataset |
| `transformer_s1` | UNet + Transformer model trained on single-scenario dataset |
| `transformer_s2` | UNet + Transformer model trained on multi-scenario dataset |
| `crossap_s1` | Cross-AP Transformer model trained on single-scenario dataset |
| `crossap_s2` | Cross-AP Transformer model trained on multi-scenario dataset |

Where:
- `s1` = single-scenario dataset (~11k samples)
- `s2` = mixed multi-scenario dataset (~9k samples)

---

## Files Inside Each Results Folder

Each folder contains:

| File | Description |
|---|---|
| `*_training_metrics.csv` | Per-epoch training and evaluation metrics |
| `*_final_errors.npy` | Final localization error values used for CDF analysis |

Example:

```text
unet_s1/
 ├── unet_s1_training_metrics.csv
 └── unet_s1_final_errors.npy
```

---

## Consistency-Loss Ablation

The `lambda_sweep/` directory contains localization error outputs for different consistency-loss weights:

```text
lambda_sweep/
 ├── lambda_0_final_errors.npy
 ├── lambda_001_final_errors.npy
 ├── lambda_01_final_errors.npy
 └── lambda_05_final_errors.npy
```

These files are used to generate the consistency decoder ablation plot shown in the report.

## Figures

The `figures/` directory contains:
- CDF plot
- consistency decoder ablation
- qualitative prediction examples


---

## Evaluation Metrics

Performance is reported using:
- Median localization error (cm)
- 90th percentile localization error (P90)

Lower values indicate better localization accuracy.