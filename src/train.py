"""
ATLAS - Training script
BraTS 2020 - Attention U-Net

Trained on Colab T4 (15GB) - 50 epochs ~2.5 hrs
Author: Tejas - DL Sem 7
"""

import os
import csv
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import get_model
from dataset import get_dataloaders
from utils import DiceFocalLoss, dice_coef, iou_score, count_params


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for reproducibility
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="  train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        # mixed precision if cuda
        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(images)
                loss = criterion(preds, masks)
            scaler.scale(loss).backward()
            # gradient clipping - helps with stable training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        n += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / n if n > 0 else 0

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    n = 0
    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        preds = model(images)
        loss = criterion(preds, masks)
        val_loss += loss.item()
        val_dice += dice_coef(preds, masks).item()
        val_iou += iou_score(preds, masks).item()
        n += 1
    if n == 0:
        return 0, 0, 0
    return val_loss / n, val_dice / n, val_iou / n


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")

    # dataloaders
    print(f"[INFO] Loading BraTS from: {args.data_dir} | img_size={args.img_size} | bs={args.batch_size}")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # model
    print(f"[INFO] Model: {args.model}")
    model = get_model(args.model).to(device)
    print(f"[INFO] Total params: {count_params(model):,}")
    # print small summary
    try:
        from torchinfo import summary
        summary(model, input_size=(args.batch_size, 4, args.img_size, args.img_size), verbose=0)
    except:
        pass

    criterion = DiceFocalLoss(dice_weight=0.6)
    # tried AdamW as well but Adam gave slightly better dice in exp 2
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.patience, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    if scaler:
        print("[INFO] Using AMP (mixed precision)")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    log_path = os.path.join("results", "metrics.csv")
    # resume support - if metrics.csv exists append, else write header
    write_header = not os.path.exists(log_path) or args.epochs == 50
    if write_header:
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr", "time_min"])

    best_dice = 0.0
    patience_counter = 0
    start_time = time.time()

    print(f"[INFO] Starting training for {args.epochs} epochs | lr={args.lr} | patience={args.patience}")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        # scheduler on val dice (we want to maximize dice)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_dice)
        new_lr = optimizer.param_groups[0]['lr']

        epoch_time = (time.time() - epoch_start) / 60.0
        total_time = (time.time() - start_time) / 60.0

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
              f"dice: {val_dice:.4f} | iou: {val_iou:.4f} | "
              f"lr: {new_lr:.6f} | {epoch_time:.1f} min")

        if new_lr != old_lr:
            print(f"  -> LR reduced {old_lr:.6f} -> {new_lr:.6f}")

        # log
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                                    f"{val_dice:.4f}", f"{val_iou:.4f}", f"{new_lr:.6f}", f"{epoch_time:.2f}"])

        # checkpoint
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            patience_counter = 0
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_dice': best_dice,
                'args': vars(args)
            }
            torch.save(ckpt, "checkpoints/best_model.pth")
            # also save last for resume
            print(f"  -> Saved best model (dice {best_dice:.4f}) -> checkpoints/best_model.pth")
        else:
            patience_counter += 1
            # always save last
            torch.save(model.state_dict(), "checkpoints/last_model.pth")

        # early stopping
        if patience_counter >= args.early_stop:
            print(f"[INFO] Early stopping triggered at epoch {epoch} (no improv for {args.early_stop} epochs)")
            break

        # small debug: print GPU mem
        if device.type == "cuda" and epoch % 10 == 0:
            print(f"  [GPU mem] {torch.cuda.memory_allocated() / 1024**2:.0f} MB allocated")

    print(f"\n[INFO] Training done in {total_time:.1f} min | Best dice: {best_dice:.4f}")
    print(f"[INFO] Best model at checkpoints/best_model.pth")
    print(f"[INFO] Metrics logged to {log_path}")

    # quick final plot suggestion
    print(f"[INFO] Run: python src/evaluate.py --checkpoint checkpoints/best_model.pth  to generate plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATLAS - Brain Tumor Segmentation Training")
    parser.add_argument("--data_dir", type=str, default="data/brats20", help="path to sliced BraTS data")
    parser.add_argument("--epochs", type=int, default=50, help="max epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size (8 fits on T4, 4 on 8GB)")
    parser.add_argument("--lr", type=float, default=1e-4, help="initial lr")
    parser.add_argument("--img_size", type=int, default=128, help="resize to 128x128")
    parser.add_argument("--model", type=str, default="attention_unet", choices=["attention_unet", "vanilla"], help="model variant")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--patience", type=int, default=5, help="scheduler patience")
    parser.add_argument("--early_stop", type=int, default=12, help="early stopping patience (epochs)")
    args = parser.parse_args()

    main(args)
