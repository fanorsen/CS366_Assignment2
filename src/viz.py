"""Evaluate a saved Lightning checkpoint and write at least 12 visual comparisons."""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score

from datamodule_oxpet import PetTrimapDataset, CLASS_NAMES
from model_unetpp import UNetPlusPlusLightning


COLORS = np.array([[220, 20, 60], [0, 0, 0], [34, 139, 34]], dtype=np.uint8)
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def colorize(mask):
    return COLORS[np.asarray(mask, dtype=np.int64)]


def denormalize(image_tensor):
    image = image_tensor.cpu() * STD + MEAN
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image, 0, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./data")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--output-dir", default="outputs/samples")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    if args.split == "test":
        dataset = PetTrimapDataset(args.root, "test", image_size=args.image_size, download=False)
    else:
        # Recreate the deterministic validation subset used by train.py.
        from torchvision.datasets import OxfordIIITPet
        base = OxfordIIITPet(root=args.root, split="trainval", target_types="segmentation", download=False)
        n_val = max(1, int(round(len(base) * args.val_fraction)))
        n_train = len(base) - n_val
        subset_train, subset_val = torch.utils.data.random_split(
            base, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )
        dataset = PetTrimapDataset(args.root, "trainval", subset_val.indices,
                                   args.image_size, augmentation=False, download=False)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    model = UNetPlusPlusLightning.load_from_checkpoint(args.checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    iou = MulticlassJaccardIndex(num_classes=3, average=None).to(device)
    dice = MulticlassF1Score(num_classes=3, average=None).to(device)
    stored = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            iou.update(predictions, masks)
            dice.update(predictions, masks)
            for image, mask, pred in zip(images.cpu(), masks.cpu(), predictions.cpu()):
                if len(stored) < args.n:
                    stored.append((denormalize(image), mask.numpy(), pred.numpy()))

    iou_values = iou.compute().cpu().numpy()
    dice_values = dice.compute().cpu().numpy()
    print("Per-class metrics (all classes included):")
    for name, iv, dv in zip(CLASS_NAMES, iou_values, dice_values):
        print(f"  {name:12s} IoU={iv:.4f} Dice={dv:.4f}")
    print(f"Macro mIoU: {iou_values.mean():.4f}")
    print(f"Macro Dice: {dice_values.mean():.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(stored), 4):
        chunk = stored[start:start + 4]
        fig, axes = plt.subplots(len(chunk), 4, figsize=(12, 3.1 * len(chunk)), squeeze=False)
        for row, (image, mask, pred) in enumerate(chunk):
            gt_rgb, pred_rgb = colorize(mask), colorize(pred)
            gt_overlay = np.clip(image * 0.55 + (gt_rgb / 255.0) * 0.45, 0, 1)
            pred_overlay = np.clip(image * 0.55 + (pred_rgb / 255.0) * 0.45, 0, 1)
            panels = [image, gt_rgb / 255.0, pred_rgb / 255.0, pred_overlay]
            titles = ["Input", "Ground truth", "Prediction", "Prediction overlay"]
            for col, (panel, title) in enumerate(zip(panels, titles)):
                axes[row, col].imshow(panel)
                axes[row, col].set_title(title)
                axes[row, col].axis("off")
            # Save one per-sample comparison, too.
            one, ax = plt.subplots(1, 4, figsize=(12, 3))
            for a, panel, title in zip(ax, panels, titles):
                a.imshow(panel)
                a.set_title(title)
                a.axis("off")
            one.tight_layout()
            one.savefig(out_dir / f"{args.split}_comparison_{start + row + 1:02d}.png", dpi=130)
            plt.close(one)
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.split}_comparisons_{start + 1:02d}_{start + len(chunk):02d}.png", dpi=130)
        plt.close(fig)
    print(f"Saved {len(stored)} comparisons to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
