# file: processing.py
import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

class SegmentationCurvatureDataset(Dataset):
    def __init__(self, dataframe, base_path, transform=None):
        self.df = dataframe
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_path, "imageSEG", row["image_seg_name"])
        label = int(row["curvatura"])

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Imagen no encontrada: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        if self.transform:
            img = self.transform(image=img)["image"]
        else:
            img = torch.tensor(img).permute(2, 0, 1).float()

        return img, torch.tensor(label, dtype=torch.long)
