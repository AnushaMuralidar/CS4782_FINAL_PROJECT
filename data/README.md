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

# Dataset organization 

### Single - scenario dataset

- Each scenario corresponds to a distinct environment/setup
- Data is divided into 10-11 chunks per scenario
- Each scenario contains: input features (AoA-ToF heatmaps) and labels ( ground truth location information)

```bash
scenario_1/
 ├── chunk_0.npz
 ├── chunk_1.npz
 └── ...
```

### Multi scenario dataset

- Combines data from 3 different scenarios
- each scenario contributes ~3-4 chunks
- Final dataset contains ~ 11 chunks total

```bash
multi_scenario/
 ├── chunk_0.npz
 ├── chunk_1.npz
 └── ...
```

This setup enables evaluation of model generalization across environments.

## Data format

Each `.npz` file contains: 

- `features_with_offset` -> Input AoA-ToF heatmaps
- `features_without_offset` -> Consistency targets
- `labels_gaussian_2d` -> 2D Gaussian location maps
- `lables` -> Ground - truth (x,y) coordinates

### Example shapes

- Input: [4, 161, 101] ( 4 AP heatmaps per sample)
- Location map : [1, 161, 101]
- Coordinates : [2]

## Storage and access

Due to size constraints the dataset is not stored in the Github repository. Data can be made available upon request. 

