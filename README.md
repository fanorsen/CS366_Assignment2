# Assignment 2 — UNet++ Semantic Segmentation on Oxford-IIIT Pet

3-class semantic segmentation (pet / background / border) of the Oxford-IIIT Pet trimaps
using UNet++ (ResNet-34 encoder, ImageNet-pretrained) and PyTorch Lightning.

**Weights & Biases dashboard:** https://forge.coreweave.com/wandb/fanorsen-hamilton-college/cs366-ass2

## Results (official test split, 3,669 images)

| Run | Test mIoU | Test Dice | IoU pet | IoU bg | IoU border |
|---|---|---|---|---|---|
| baseline (CE) | 0.7884 | 0.8729 | 0.8645 | 0.9274 | 0.5734 |
| + CE+Dice loss | 0.7932 | 0.8763 | 0.8674 | 0.9296 | 0.5828 |
| + augmentation | 0.8035 | 0.8829 | 0.8800 | 0.9358 | 0.5946 |
| **+ both (best)** | **0.8062** | **0.8850** | 0.8798 | 0.9364 | 0.6024 |

Best checkpoint: `ce_dice_aug-epoch=28-val_miou=0.8040.ckpt` (best val mIoU).

## Project structure

```
README.md
requirements.txt
oxpet_download_and_viz_fixed.py   # provided data exploration script
src/
  datamodule_oxpet.py             # LightningDataModule (80/20 train/val split of trainval, test split)
  model_unetpp.py                 # LightningModule: UNet++, losses, mIoU/Dice metrics (macro + per-class)
  train.py                        # CLI: --mode train (fit) or --mode eval (test a checkpoint)
  viz.py                          # colorized masks + input/GT/prediction/overlay panels
outputs/
  checkpoints/  logs/  samples/
```

## 1. Setup

Tested with Python 3.13, PyTorch 2.11 (CUDA 12.8), on an NVIDIA RTX 5070 Ti (16 GB).

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
wandb login   # optional, only needed for --wandb logging
```

## 2. Download the dataset

```bash
mkdir -p data/oxford-iiit-pet && cd data/oxford-iiit-pet
wget https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz
wget https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz
tar -xzf images.tar.gz && tar -xzf annotations.tar.gz
rm images.tar.gz annotations.tar.gz
cd ../..
```

## 3. Explore the data

```bash
python oxpet_download_and_viz_fixed.py --root ./data --split trainval \
  --classes trimap --n 12 --resize 512 --save-dir outputs/samples
```

Produces `outputs/samples/oxpet_viz_grid.png` (image, colorized trimap, overlay).

## 4. Train

Trimap values {1, 2, 3} are mapped to {0 = pet, 1 = background, 2 = border}; masks are resized
with NEAREST interpolation. The best checkpoint by validation mIoU is saved to `outputs/checkpoints/`.

```bash
# Baseline: CrossEntropy, no augmentation
python src/train.py --root ./data --epochs 40 --batch-size 16 --image-size 256 \
  --num-workers 6 --wandb --run-name baseline

# Improvement 1: CrossEntropy + Dice loss
python src/train.py --root ./data --epochs 40 --batch-size 16 --image-size 256 \
  --num-workers 6 --wandb --run-name ce_dice --loss ce_dice

# Improvement 2: data augmentation
python src/train.py --root ./data --epochs 40 --batch-size 16 --image-size 256 \
  --num-workers 6 --wandb --run-name aug --augmentation

# Both improvements
python src/train.py --root ./data --epochs 40 --batch-size 16 --image-size 256 \
  --num-workers 6 --wandb --run-name ce_dice_aug --loss ce_dice --augmentation
```

Training settings: AdamW (lr 1e-3, weight decay 1e-4), ReduceLROnPlateau on val mIoU
(factor 0.5, patience 2), early stopping on val mIoU (patience 5), gradient clipping 1.0,
16-bit mixed precision (AMP), seed 42, 80/20 train/val split of `trainval`.
Each run took about 8–11 minutes; total GPU time across all runs was about 40 minutes.

## 5. Evaluate on the test split

```bash
python src/train.py --mode eval --root ./data --image-size 256 --batch-size 16 \
  --run-name ce_dice_aug --ckpt "outputs/checkpoints/ce_dice_aug-epoch=28-val_miou=0.8040.ckpt"
```

Prints overall and per-class mIoU / Dice and saves them to `outputs/logs/test_<run-name>.json`.
Repeat with each run's checkpoint to reproduce the results table.

## 6. Visualize predictions

```bash
python src/viz.py --root ./data \
  --checkpoint "outputs/checkpoints/ce_dice_aug-epoch=28-val_miou=0.8040.ckpt" \
  --split test --image-size 256 --n 12 --output-dir outputs/samples
```

Saves input / ground truth / prediction / overlay panels to `outputs/samples/`.

## Running on a Slurm cluster

We trained on Hamilton's cs-hpc Slurm cluster. Each command above was wrapped in a batch script like:

```bash
#!/bin/bash
#SBATCH --partition=students
#SBATCH --account=cs410
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:50:00
source /path/to/venv/bin/activate
cd /path/to/Assignment2
python src/train.py --root /path/to/data --epochs 40 --batch-size 16 --image-size 256 \
  --num-workers 6 --wandb "$@"
```

submitted with, e.g., `sbatch train.sbatch --run-name aug --augmentation`.

## References

- Z. Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation," DLMIA 2018.
- O. M. Parkhi et al., "Cats and Dogs," CVPR 2012 (Oxford-IIIT Pet dataset).
- segmentation-models-pytorch: https://github.com/qubvel-org/segmentation_models.pytorch
- PyTorch Lightning: https://lightning.ai
- TorchMetrics: https://lightning.ai/docs/torchmetrics