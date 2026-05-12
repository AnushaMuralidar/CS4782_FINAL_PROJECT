# Code Directory

## Overview

This directory contains the model implementations, training scripts, preprocessing utilities, evaluation code, and plotting scripts used for the DLoc re-implementation project.

---

## Directory Structure

```text
code/
 ├── models/
 ├── preprocessing/
 ├── plots/
 ├── dataset.py
 ├── losses.py
 ├── evaluate.py
 ├── train.py
 └── README.md
```

---

## Models

```text
models/
 ├── baseline.py
 ├── unet.py
 ├── unet_transformer.py
 └── unet_crossap_transformer.py
```

| File | Description |
|---|---|
| `baseline.py` | ResNet encoder-decoder baseline from the DLoc paper |
| `unet.py` | UNet extension with skip connections |
| `unet_transformer.py` | SE-UNet with Transformer bottleneck |
| `unet_crossap_transformer.py` | Proposed Cross-AP Attention + Transformer model |

---

## Preprocessing

```text
preprocessing/
 ├── mat_to_npz.py
 └── resize_grid.py
```

| Script | Purpose |
|---|---|
| `mat_to_npz.py` | Converts raw `.mat` files into compressed `.npz` chunks |
| `resize_grid.py` | Resizes heatmaps to a common spatial resolution |

Example:

```bash
python preprocessing/mat_to_npz.py --input <file>.mat --output <scenario_folder>
```

---

## Training Pipeline

| File | Description |
|---|---|
| `dataset.py` | Loads `.npz` chunks and creates train/test splits |
| `losses.py` | Defines localization and consistency losses |
| `evaluate.py` | Computes localization metrics |
| `train.py` | Main training script |

Example:

```bash
python train.py --model unet --data_path ../data/scenario_1
```

---

## Plotting Scripts

```text
plots/
 ├── plot_cdf.py
 └── plot_lambda_sweep.py
```

| Script | Purpose |
|---|---|
| `plot_cdf.py` | Generates localization error CDF plots |
| `plot_lambda_sweep.py` | Generates consistency-loss ablation plots |

Generated figures are saved to:

```text
results/figures/
```

---

## Training Outputs

Training automatically saves:
- model checkpoints
- localization errors
- training metrics

Outputs are stored inside:

```text
results/<model_name>/
```