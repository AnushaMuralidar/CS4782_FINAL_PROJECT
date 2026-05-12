# Dataset description
## Overview

This project uses a preprocessed Angle of Arrival (AoA) and Time of Flight (ToF) heatmap features derived from wireless Channel State Information (CSI) measurements for indoor locaization. Each data sample is represented as a multi channel heatmap corresponding to multiple access points (APs) and is used as input to deep learning models for predicting spatial location. 

## Data preparation pipeline

The dataset was originally provided in `.mat` format and we did the following preprocessing steps: 

- Converted `.mat` files to `.npz` format (primarily due to storage constraints and to improve data loading efficiency)
- Extracted AoA-ToF heatmap features from the raw data
- Organized data into fixed-size chunks 
- Reduced storage size significantly(~ 15-16 GB to ~2.5GB per scenario)

Each chunk contains approximately 1000 samples enabling modular loading during training. 
Preprocessing scripts are available in:

```text
code/preprocessing/
 ├── mat_to_npz.py
 └── resize_grid.py
```

## Dataset Organization

### Single-scenario dataset

Each scenario corresponds to a distinct indoor environment/setup.

```text
scenario_1/
 ├── chunk_00.npz
 ├── chunk_01.npz
 └── ...
```

- ~10–11 chunks per scenario
- ~1000 samples per chunk
- ~11k total samples

---

### Multi-scenario dataset

The multi-scenario dataset combines subsets from three environments:

- `dataset_jacobs_July28`
- `dataset_jacobs_July28_2`
- `dataset_aug16_4_ref`

```text
multi_scenario/
 ├── chunk_00.npz
 ├── chunk_01.npz
 └── ...
```

This setup enables evaluation of generalization under diverse NLOS and multipath conditions.


## Data Format

Each `.npz` chunk contains:

| Key | Description |
|---|---|
| `A` | Input AoA-ToF heatmaps |
| `B` | Consistency targets |
| `L` | 2D Gaussian location heatmaps |
| `G` | Ground-truth `(x,y)` coordinates |

### Example Shapes

| Tensor | Shape |
|---|---|
| `A` | `[4, 161, 101]` |
| `B` | `[4, 161, 101]` |
| `L` | `[1, 161, 101]` |
| `G` | `[2]` |

## Storage and access

Due to size constraints the dataset is not included in the Github repository. Preprocessed `.npz` data can be accessed [here](https://drive.google.com/drive/folders/1Yu8NlegvyCDvzDedaokEcslB0ep9DOg0?usp=sharing). If access is unavailable, follow the preprocessing steps in the main README to regenerate the dataset locally.

