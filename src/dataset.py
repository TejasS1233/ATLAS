import os, glob, numpy as np, cv2
import torch
from torch.utils.data import Dataset
import albumentations as A

# BraTS 2020 has .npy slices after preprocessing, or raw .nii.gz
# We handle both. For quick demo we support 2D sliced .npy

class BraTSDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None, img_size=128):
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        # expects data_dir/train/images/*.npy and masks/*.npy
        # fallback: if not found, will work with dummy data for report generation
        self.images = sorted(glob.glob(os.path.join(data_dir, split, "images", "*.npy")))
        self.masks = sorted(glob.glob(os.path.join(data_dir, split, "masks", "*.npy")))
        self.has_data = len(self.images) > 0
        if not self.has_data:
            print(f"[WARN] No data found in {data_dir}/{split}. Using synthetic dummy for testing pipeline.")

    def __len__(self):
        return len(self.images) if self.has_data else 200  # dummy length

    def __getitem__(self, idx):
        if self.has_data:
            img = np.load(self.images[idx])  # shape [4, H, W] or [H,W,4]
            mask = np.load(self.masks[idx])  # shape [H,W]
            if img.shape[0] != 4 and img.shape[-1] == 4:
                img = np.transpose(img, (2,0,1))
        else:
            # synthetic data - so code runs without downloading
            img = np.random.randn(4, 240, 240).astype(np.float32)
            mask = (np.random.rand(240,240) > 0.9).astype(np.float32)

        # resize to img_size
        if img.shape[1] != self.img_size:
            img_resized = []
            for c in range(4):
                img_resized.append(cv2.resize(img[c], (self.img_size, self.img_size)))
            img = np.stack(img_resized)
            mask = cv2.resize(mask, (self.img_size, self.img_size))

        # albumentations expects HWC
        if self.transform:
            # convert to HWC for aug
            img_hwc = np.transpose(img, (1,2,0))
            augmented = self.transform(image=img_hwc, mask=mask)
            img_hwc = augmented['image']
            mask = augmented['mask']
            img = np.transpose(img_hwc, (2,0,1))

        # normalize per channel already 0-1
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # [1,H,W]
        return img, mask

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
            A.ElasticTransform(p=0.2),
        ])
    else:
        return None

def get_dataloaders(data_dir="data/brats20", batch_size=8, img_size=128):
    from torch.utils.data import DataLoader
    train_ds = BraTSDataset(data_dir, "train", transform=get_transforms(True), img_size=img_size)
    val_ds = BraTSDataset(data_dir, "val", transform=get_transforms(False), img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
