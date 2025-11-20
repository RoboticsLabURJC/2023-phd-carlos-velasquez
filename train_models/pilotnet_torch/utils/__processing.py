
import os
import cv2
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class SegmentedLaneDataset3CH(Dataset):
    def __init__(self, df_or_csv, transform=None, base_path="data/dataset"):
        if isinstance(df_or_csv, str):
            self.df = pd.read_csv(df_or_csv)
        else:
            self.df = df_or_csv.reset_index(drop=True)

        self.transform = transform
        self.base_path = base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self.df):
            return None

        try:
            row = self.df.iloc[idx]
            seg_path = os.path.join(self.base_path, row["seg_path"])
            
            # steer = np.interp(row["steer"], (-1, 1), (0, 1))
            # throttle = np.clip(row["throttle"], 0.0, 1.0)
            
            label = torch.tensor([row['steer'], row['throttle']], dtype=torch.float32)

            image_seg = cv2.imread(seg_path)
            if image_seg is None:
                raise FileNotFoundError(f"Imagen no encontrada: {seg_path}")

            calzada_color = [128, 64, 128]
            mask = cv2.inRange(
                image_seg, np.array(calzada_color), np.array(calzada_color)
            )
            masked_image = np.zeros_like(image_seg)
            masked_image[mask > 0] = [255, 255, 255]

            image_seg_rgb = cv2.resize(masked_image[200:-1, :], (200, 66))
            
            gray = cv2.cvtColor(image_seg_rgb, cv2.COLOR_BGR2GRAY)
            rgb_like = cv2.merge([gray, gray, gray])
            
            if random.random() < 0.5:
                rgb_like = cv2.flip(rgb_like, 1)
                label[0] = -label[0]  # Invertir dirección del giro
                
            if self.transform:
                augmented = self.transform(image=rgb_like, mask=image_seg_rgb)
                image_transformed = augmented['image']
                image_tensor = torch.tensor(image_transformed, dtype=torch.float32).permute(2, 0, 1)
            else:
                rgb_like = rgb_like.astype(np.float32) / 255.0
                rgb_like = (rgb_like - 0.5) / 0.5
                image_tensor = torch.tensor(rgb_like, dtype=torch.float32).permute(2, 0, 1)
            
            return image_tensor, label

        except Exception as e:
            if idx < 5 or idx % 1000 == 0:
                print(f"[ERROR] en índice {idx}: {e}")
            return None
  
