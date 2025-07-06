import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentedLaneDataset3CH(Dataset):
    def __init__(self, df_or_csv, transform=None, base_path=""):
        if isinstance(df_or_csv, str):
            self.df = pd.read_csv(df_or_csv)
        else:
            self.df = df_or_csv.reset_index(drop=True)

        self.transform = transform
        self.base_path = base_path  # <- NUEVO

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            # Agrega el base_path si se especificó
            seg_path = os.path.join(self.base_path, row["seg_path"])
            label = torch.tensor([row["steer"], row["throttle"]], dtype=torch.float32)

            image_seg = cv2.imread(seg_path)
            if image_seg is None:
                raise FileNotFoundError(f"Imagen no encontrada: {seg_path}")

            calzada_color = [128, 64, 128]
            mask = cv2.inRange(
                image_seg, np.array(calzada_color), np.array(calzada_color)
            )
            masked_image = np.zeros_like(image_seg)
            masked_image[mask > 0] = [255, 255, 255]

            cropped = masked_image[200:-1, :]
            resized = cv2.resize(cropped, (200, 66))  # PilotNet input size
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            rgb_like = cv2.merge([gray, gray, gray])
            img_tensor = (
                torch.tensor(rgb_like, dtype=torch.float32).permute(2, 0, 1) / 255.0
            )

            return img_tensor, label

        except Exception as e:
            print(f"[ERROR] en índice {idx}: {e}")
            return None
