# PortionNet: Distilling 3D Geometric Knowledge for Food Nutrition Estimation

[![arXiv](https://img.shields.io/badge/arXiv-2512.22304-b31b1b.svg)](https://arxiv.org/abs/2512.22304)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of **PortionNet**, a novel cross-modal knowledge distillation framework for accurate food nutrition estimation from single RGB images.

> **PortionNet: Distilling 3D Geometric Knowledge for Food Nutrition Estimation**  
> Darrin Bright, Rakshith Raj, Kanchan Keisham  
> Vellore Institute of Technology  
> *arXiv preprint arXiv:2512.22304, 2025*

## Highlights

- State-of-the-art RGB-only performance on MetaFood3D:
  - 17.43% volume MAPE (57.9% improvement over MFP3D)
  - 15.36% energy MAPE (77.4% improvement over MFP3D)
  
- Exceptional cross-dataset generalization on SimpleFood45:
  - 12.17% energy MAPE (49.4% improvement over MFP3D)
  - 100% classification accuracy

- No depth sensors required at inference - works with standard smartphone cameras

- Novel dual-mode training strategy with cross-modal knowledge distillation

## Overview

PortionNet addresses the challenge of accurate food nutrition estimation from single RGB images by learning 3D geometric features from point clouds during training while requiring only RGB images at inference.

### Key Components

1. **Dual RGB Encoders**: ViT-B/16 + ResNet-18 for robust visual feature extraction
2. **PointNet Geometry Encoder**: Extracts 3D spatial features from point clouds (training only)
3. **RGB-to-Point-Cloud Adapter**: Lightweight network that mimics point cloud features
4. **Cross-Modal Fusion**: Bidirectional attention for feature integration
5. **Multi-Task Heads**: Simultaneous food classification, volume, and energy estimation

### Architecture

![PortionNet Architecture](assets/architecture.png)

*Figure 1: Overview of the PortionNet framework. During training, the model uses both RGB images and point clouds with cross-modal knowledge distillation. At inference, only RGB images are required.*

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU training)

### Setup

```bash
git clone https://github.com/darrinbright/PortionNet.git
cd PortionNet

conda create -n portionnet python=3.8
conda activate portionnet

pip install -r requirements.txt
pip install open3d
```

## Dataset Preparation

### MetaFood3D

1. Download the MetaFood3D dataset from [here](https://github.com/GCVCG/MetaFood3D)
2. Extract and organize as follows:

```
MetaFood3D/
├── RGB/
│   ├── apple/
│   │   ├── apple_1.png
│   │   └── ...
│   └── ...
├── Point_Cloud/
│   ├── apple/
│   │   ├── apple_1.ply
│   │   └── ...
│   └── ...
└── _MetaFood3D_new_complete_dataset_nutrition_v2.xlsx
```

### SimpleFood45

Download SimpleFood45 from [here](https://github.com/GCVCG/SimpleFood45) and organize similarly.

## Training

### Basic Training

```bash
python src/train.py \
  --data_dir /path/to/MetaFood3D \
  --excel_path /path/to/MetaFood3D/_MetaFood3D_new_complete_dataset_nutrition_v2.xlsx \
  --output_dir ./outputs \
  --epochs 25 \
  --batch_size 16 \
  --seed 7
```

### Multi-Seed Training

For reproducible results as reported in the paper:

```bash
# Seed 7
python src/train.py --data_dir /path/to/MetaFood3D --excel_path /path/to/excel --seed 7 --output_dir ./outputs/seed7

# Seed 13
python src/train.py --data_dir /path/to/MetaFood3D --excel_path /path/to/excel --seed 13 --output_dir ./outputs/seed13

# Seed 2023
python src/train.py --data_dir /path/to/MetaFood3D --excel_path /path/to/excel --seed 2023 --output_dir ./outputs/seed2023
```

### Key Hyperparameters

- `--rgb_only_ratio`: Proportion of batches trained in RGB-only mode (default: 0.3)
- `--lambda_distill`: Weight for distillation loss (default: 0.5)
- `--lambda_reg`: Weight for regression loss (default: 0.1)

## Evaluation

### Evaluate on MetaFood3D

```bash
# RGB-only mode (inference mode)
python src/evaluate.py \
  --data_dir /path/to/MetaFood3D \
  --excel_path /path/to/excel \
  --checkpoint ./outputs/best_model_seed7.pt \
  --mode rgb_only \
  --output_file results_rgb.json

# Multimodal mode (with point clouds)
python src/evaluate.py \
  --data_dir /path/to/MetaFood3D \
  --excel_path /path/to/excel \
  --checkpoint ./outputs/best_model_seed7.pt \
  --mode multimodal \
  --output_file results_multimodal.json
```

### Cross-Dataset Evaluation on SimpleFood45

```bash
python src/evaluate.py \
  --data_dir /path/to/SimpleFood45 \
  --excel_path /path/to/SimpleFood45/labels.xlsx \
  --checkpoint ./outputs/best_model_seed7.pt \
  --mode rgb_only \
  --num_classes 12 \
  --output_file results_simplefood45.json
```

## Results

### MetaFood3D Performance (RGB-only Mode)

| Metric | Mean ± Std | Individual Seeds (7, 13, 2023) |
|--------|------------|--------------------------------|
| Accuracy (%) | 98.34 ± 0.58 | 98.12, 97.88, 98.16 |
| Volume MAE (mL) | 25.39 ± 2.59 | 27.51, 26.13, 27.27 |
| Volume MAPE (%) | 17.43 ± 0.81 | 28.34, 24.81, 22.73 |
| Energy MAE (kcal) | 32.26 ± 0.43 | 38.09, 32.13, 30.88 |
| Energy MAPE (%) | 15.36 ± 1.33 | 41.91, 37.52, 31.94 |
| R² Score | 0.926 ± 0.004 | 0.9216, 0.9271, 0.9110 |

### Comparison with State-of-the-Art

**MetaFood3D**

| Method | Vol MAPE (%) | Eng MAPE (%) |
|--------|--------------|--------------|
| Density Map† | - | 663.43 |
| Stereo† | 210.90 | - |
| Voxel† | 104.07 | - |
| 3D Assisted† | 79.33 | 102.25 |
| MFP3D† | 41.43 | 68.05 |
| **PortionNet (Ours)** | **17.43 ± 0.81** | **15.36 ± 1.33** |
| Improvement | -57.9% | -77.4% |

†Results from MFP3D paper

**SimpleFood45 (Cross-Dataset)**

| Method | Vol MAPE (%) | Eng MAPE (%) |
|--------|--------------|--------------|
| 3D Assisted† | 14.01 | 25.13 |
| MFP3D† | 16.15 | 24.03 |
| **PortionNet (Ours)** | **23.51 ± 0.92** | **12.17 ± 1.36** |
| Improvement | - | -49.4% |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{bright2025portionnet,
  title={PortionNet: Distilling 3D Geometric Knowledge for Food Nutrition Estimation},
  author={Bright, Darrin and Raj, Rakshith and Keisham, Kanchan},
  journal={arXiv preprint arXiv:2512.22304},
  year={2025}
}
```

## Acknowledgments

- MetaFood3D dataset: [Chen et al., 2024](https://github.com/GCVCG/MetaFood3D)
- SimpleFood45 dataset: [Vinod et al., 2024](https://github.com/GCVCG/SimpleFood45)
- MFP3D baseline: [Ma et al., 2024](https://arxiv.org/abs/2411.10492)

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: darrin.bright2022@vitstudent.ac.in
