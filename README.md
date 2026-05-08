# CS4782_FINAL_PROJECT
# DLoc: Deep learning based Indoor Localization

## Overview 
This project implements deep learning models for indoor localization using **Angle of Arrival (AoA)** and **Time of Flight (ToF)** heatmap features derived from wireless Channel State Information (CSI). The goal is to predict spatial location from multi-AP RF measurements. 

We evaluate three models: 
- **Baseline model** (ResNet-style encoder–decoder)
- **UNet model**
- **UNet + Transformer model**

## Project Structure

```text
CS4782_FINAL_PROJECT/
 ├── code/
 │    ├── train.py
 │    ├── dataset.py
 │    ├── losses.py
 │    ├── evaluate.py
 │    └── models/
 │         ├── baseline.py
 │         ├── unet.py
 │         └── unet_transformer.py
 ├── data/
 │    └── README.md
 ├── poster/
 ├── report/
 ├── results/
 ├── requirements.txt
 └── README.md
 ```

 ## Dataset

The dataset is not included in this repository due to size constraints.

Expected structure:

```text
data/
 ├── scenario_1/
 ├── scenario_2/
```
Each folder contains `.npz` chunk files.

## How to run

### Install dependencies
```bash
pip install -r requirements.txt
```
### Run training on models
```bash
cd code

python train.py --model unet --data_path ../data/scenario_1
python train.py --model baseline --data_path ../data/scenario_1
python train.py --model unet_transformer --data_path ../data/scenario_1

```

## Running on Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/CS4782_project

!pip install -r requirements.txt

!python code/train.py --model unet --data_path data/scenario_1 --epochs 5 --batch_size 32 --save_dir results/unet_s1

```
### Try other models

```python
# Baseline
!python code/train.py --model baseline --data_path data/scenario_1 --save_dir results/baseline_s1

# UNet + Transformer (use smaller batch size)
!python code/train.py --model unet_transformer --data_path data/scenario_1 --batch_size 8 --save_dir results/transformer_s1
```