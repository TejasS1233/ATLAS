"""
Download BraTS 2020 from Kaggle

Usage:
  1. Get Kaggle API key from kaggle.com/account -> Create New Token -> saves kaggle.json
  2. Put kaggle.json in C:/Users/tejas/.kaggle/ (Windows) or ~/.kaggle/ (Linux)
  3. python data/download.py

Alternatively manually download from:
https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
and unzip to data/brats20/

For quick testing without Kaggle, this script creates dummy sliced data
so training pipeline runs end-to-end.
"""
import os, zipfile, pathlib

def create_dummy_data(root="data/brats20"):
    import numpy as np
    for split in ["train","val"]:
        os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, split, "masks"), exist_ok=True)
        n = 200 if split=="train" else 40
        print(f"Creating {n} dummy slices in {root}/{split} ...")
        for i in range(n):
            img = np.random.rand(4, 128, 128).astype(np.float32)
            # create fake tumor blob
            mask = np.zeros((128,128), dtype=np.float32)
            cx, cy = np.random.randint(40,88,2)
            rr, cc = np.ogrid[:128,:128]
            mask[(rr-cx)**2 + (cc-cy)**2 < 15**2] = 1
            np.save(os.path.join(root, split, "images", f"{i:04d}.npy"), img)
            np.save(os.path.join(root, split, "masks", f"{i:04d}.npy"), mask)
    print("Dummy data ready. Replace with real BraTS for actual results.")

if __name__ == "__main__":
    # Try kaggle download
    try:
        import kaggle
        os.makedirs("data", exist_ok=True)
        print("Downloading BraTS20 from Kaggle...")
        kaggle.api.dataset_download_files("awsaf49/brats20-dataset-training-validation", path="data", unzip=True)
        print("Downloaded to data/")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Falling back to dummy data for pipeline testing...")
        create_dummy_data()
