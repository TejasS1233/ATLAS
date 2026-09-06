import torch, argparse, os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from model import get_model
from dataset import get_dataloaders
from utils import dice_coef, iou_score

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(args.data_dir, batch_size=4, img_size=args.img_size)
    model = get_model(args.model).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
    # train.py saves {'model_state': ...}, handle both formats
    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
        print(f"Loaded {args.checkpoint}")
    else:
        print(f"[WARN] {args.checkpoint} not found, using random weights for demo")
    model.eval()

    dices, ious = [], []
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            dices.append(dice_coef(pred, mask).item())
            ious.append(iou_score(pred, mask).item())

    print(f"Mean Dice: {np.mean(dices):.4f} | Mean IoU: {np.mean(ious):.4f}")

    # save visualization
    os.makedirs("results", exist_ok=True)
    img, mask = next(iter(val_loader))
    img, mask = img.to(device), mask.to(device)
    with torch.no_grad():
        pred = model(img)
    pred_bin = (pred > 0.5).float().cpu().numpy()
    img_np = img.cpu().numpy()
    mask_np = mask.cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(12,6))
    for i in range(4):
        axes[0,i].imshow(img_np[i,0], cmap='gray')
        axes[0,i].set_title(f"FLAIR {i}")
        axes[0,i].axis('off')
        axes[1,i].imshow(mask_np[i,0], cmap='gray')
        axes[1,i].imshow(pred_bin[i,0], cmap='Reds', alpha=0.5)
        axes[1,i].set_title(f"GT + Pred")
        axes[1,i].axis('off')
    plt.tight_layout()
    plt.savefig("results/sample_predictions.png", dpi=200)
    print("Saved results/sample_predictions.png")

    # plot curves only from real runs - never synthesize metrics
    import csv
    if not os.path.exists("results/metrics.csv"):
        print("[ERROR] results/metrics.csv not found. Run src/train.py first - no dummy data generated.")
        return

    # plot curves
    import pandas as pd
    df = pd.read_csv("results/metrics.csv")
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(df['epoch'], df['train_loss'], label='train_loss')
    ax[0].plot(df['epoch'], df['val_loss'], label='val_loss')
    ax[0].legend(); ax[0].set_title("Loss Curve"); ax[0].set_xlabel("Epoch")
    ax[1].plot(df['epoch'], df['val_dice'], label='val_dice')
    ax[1].plot(df['epoch'], df['val_iou'], label='val_iou')
    ax[1].legend(); ax[1].set_title("Dice / IoU Curve"); ax[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig("results/curves.png", dpi=200)
    print("Saved results/curves.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/brats20")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--model", type=str, default="attention_unet")
    args = parser.parse_args()
    evaluate(args)
