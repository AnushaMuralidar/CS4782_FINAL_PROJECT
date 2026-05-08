# Dataset description
## Overview

This project uses a preprocessed Angle of Arrival (AoA) and Time of Flight (ToF) heatmap features derived from wireless Channel State Information (CSI) measurements for indoor locaization. Each data sample is represented as a multi channel heatmap corresponding to multiple access points (APs) and is used as input to deep learnign models for predictogn spatial location. 

# Data preparation pipeline

The dataset was origiannly provided in `.mat` format and we did the following preprocessing steps: 

- Convereted `.mat` files to `.npz` format (primarily due to storage constraints and to improve data loading efficieny)
- Extracted AoA-ToF heatmap features from the raw data
- Organized data into fixed-size chunks 
- Reduced storage size significantly(~ 15-16 GB to ~2.5GB per scenario)

Each chunk contains approximately 1000 samples enabling modular loading during training. 

# Dataset organization 

## Single - scenario dataset

- Each scenario corresponds to a distinct environment/setup
- Data is divided into 10-11 chunks per scenrio
- Each scenario contains: input features (AoA-ToF heatmaps) and labels ( ground truth location information)

```bash
scenario_1/
 ├── chunk_0.npz
 ├── chunk_1.npz
 └── ...
```
