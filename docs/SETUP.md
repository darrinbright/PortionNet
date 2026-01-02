# Setup Guide for PortionNet

This guide will help you set up and run PortionNet for food nutrition estimation.

## Prerequisites

- Linux or macOS (Windows with WSL2 also works)
- Python 3.8 or higher
- CUDA-capable GPU with at least 8GB VRAM (recommended)
- 50GB+ free disk space for datasets

## Step 1: Environment Setup

### Option A: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n portionnet python=3.8
conda activate portionnet

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Option B: Using venv

```bash
# Create virtual environment
python -m venv portionnet_env
source portionnet_env/bin/activate  # On Windows: portionnet_env\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Step 2: Dataset Preparation

### MetaFood3D

1. Download the dataset:
   ```bash
   # Visit https://github.com/GCVCG/MetaFood3D
   # Download RGB images, point clouds, and nutrition labels
   ```

2. Organize the dataset:
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

3. Update paths in training scripts:
   ```bash
   # Edit scripts/train_metafood3d.sh
   DATA_DIR="/path/to/MetaFood3D"
   EXCEL_PATH="/path/to/MetaFood3D/_MetaFood3D_new_complete_dataset_nutrition_v2.xlsx"
   ```

### SimpleFood45 (Optional, for cross-dataset evaluation)

1. Download from https://github.com/GCVCG/SimpleFood45
2. Organize similarly to MetaFood3D

## Step 3: Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test Open3D installation
python -c "import open3d; print(f'Open3D: {open3d.__version__}')"

# Test imports
python -c "from src.models import PortionNet; print('PortionNet imported successfully!')"
```

## Step 4: Quick Start Training

### Single Seed Training

```bash
# Make script executable
chmod +x scripts/train_metafood3d.sh

# Run training
./scripts/train_metafood3d.sh
```

### Multi-Seed Training (Reproducible Results)

```bash
# Make script executable
chmod +x scripts/train_multiseed.sh

# Run training with seeds 7, 13, 2023
./scripts/train_multiseed.sh
```

### Manual Training

```bash
python src/train.py \
  --data_dir /path/to/MetaFood3D \
  --excel_path /path/to/MetaFood3D/_MetaFood3D_new_complete_dataset_nutrition_v2.xlsx \
  --output_dir ./outputs \
  --epochs 25 \
  --batch_size 16 \
  --seed 7
```

## Step 5: Evaluation

### Evaluate Trained Model

```bash
# Make script executable
chmod +x scripts/evaluate.sh

# Update checkpoint path in the script
# Then run evaluation
./scripts/evaluate.sh
```

### Manual Evaluation

```bash
# RGB-only mode (inference mode)
python src/evaluate.py \
  --data_dir /path/to/MetaFood3D \
  --excel_path /path/to/excel \
  --checkpoint ./outputs/best_model_seed7.pt \
  --mode rgb_only

# Multimodal mode (with point clouds)
python src/evaluate.py \
  --data_dir /path/to/MetaFood3D \
  --excel_path /path/to/excel \
  --checkpoint ./outputs/best_model_seed7.pt \
  --mode multimodal
```

## Step 6: Cross-Dataset Evaluation (SimpleFood45)

```bash
python src/evaluate.py \
  --data_dir /path/to/SimpleFood45 \
  --excel_path /path/to/SimpleFood45/labels.xlsx \
  --checkpoint ./outputs/best_model_seed7.pt \
  --mode rgb_only \
  --num_classes 12
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:
- Reduce batch size: `--batch_size 8`
- Use gradient accumulation (modify train.py)
- Use a smaller model: `--feature_dim 128`

### Point Cloud Loading Issues

If Open3D fails to load point clouds:
- Ensure Open3D is properly installed: `pip install open3d`
- Check .ply file format compatibility
- Verify file paths are correct

### Slow Training

To speed up training:
- Increase `--num_workers` (default: 4)
- Use mixed precision training (modify train.py to use `torch.cuda.amp`)
- Ensure data is on SSD, not HDD

### Import Errors

If you get import errors:
```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install as package
pip install -e .
```

## Hardware Requirements

### Minimum
- GPU: 8GB VRAM (e.g., RTX 3070)
- RAM: 16GB
- Storage: 50GB

### Recommended
- GPU: 16GB+ VRAM (e.g., RTX 4090, A100)
- RAM: 32GB+
- Storage: 100GB+ SSD

## Training Time Estimates

On RTX 4090:
- Single seed (25 epochs): ~2-3 hours
- Multi-seed (3 seeds): ~6-9 hours

On RTX 3090:
- Single seed (25 epochs): ~3-4 hours
- Multi-seed (3 seeds): ~9-12 hours

## Expected Results

After training, you should see:

**MetaFood3D (RGB-only mode):**
- Accuracy: ~98%
- Volume MAPE: ~17-28%
- Energy MAPE: ~15-42%
- R²: ~0.91-0.93

**SimpleFood45 (cross-dataset):**
- Accuracy: 100%
- Volume MAPE: ~17-22%
- Energy MAPE: ~12-13%

## Next Steps

1. **Experiment with hyperparameters**: Adjust learning rates, loss weights, etc.
2. **Visualize results**: Create plots of predictions vs ground truth
3. **Deploy model**: Export to ONNX for deployment on mobile devices
4. **Fine-tune on custom data**: Adapt to your own food dataset

## Getting Help

If you encounter issues:
1. Check the [GitHub Issues](https://github.com/yourusername/portionnet/issues)
2. Review the paper for methodology details
3. Contact: darrin.bright2022@vitstudent.ac.in

## Additional Resources

- Paper: https://arxiv.org/abs/2512.22304
- MetaFood3D: https://github.com/GCVCG/MetaFood3D
- SimpleFood45: https://github.com/GCVCG/SimpleFood45
- PyTorch Documentation: https://pytorch.org/docs/
