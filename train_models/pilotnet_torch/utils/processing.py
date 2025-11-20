import os
import cv2
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class SegmentedLaneDataset3CH(Dataset):
    def __init__(self, df_or_csv, transform=None, base_path="data/dataset_dagger"):
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
            seg_rel_path = row["seg_path"]  # p.ej. runs_5/imageSEG/00007650_403.673.jpg

            # Construir ruta RGB a partir de la segmentada
            rgb_rel_path = seg_rel_path.replace("imageSEG", "imageRGB")
            rgb_path = os.path.join(self.base_path, rgb_rel_path)

            # Etiquetas
            label = torch.tensor([row["steer"], row["throttle"]], dtype=torch.float32)

            # Leer imagen RGB original
            image_rgb = cv2.imread(rgb_path)
            if image_rgb is None:
                raise FileNotFoundError(f"Imagen RGB no encontrada: {rgb_path}")

            # BGR -> RGB
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

            # Crop y resize igual que antes (manteniendo región inferior de la imagen)
            image_rgb = image_rgb[200:-1, :]               # recorte vertical
            image_rgb = cv2.resize(image_rgb, (200, 66))   # (W,H) = (200,66)

            # Aumentos / normalización
            if self.transform:
                # Albumentations espera HWC uint8 o float; usamos uint8 aquí
                augmented = self.transform(image=image_rgb)
                image_transformed = augmented["image"]     # ya float32 y normalizada si hay Normalize
                image_tensor = torch.tensor(
                    image_transformed, dtype=torch.float32
                ).permute(2, 0, 1)  # (C,H,W)
            else:
                # Sin transform: normalizamos a [-1,1] manualmente
                image_rgb = image_rgb.astype(np.float32) / 255.0
                image_rgb = (image_rgb - 0.5) / 0.5        # [-1,1]
                image_tensor = torch.tensor(
                    image_rgb, dtype=torch.float32
                ).permute(2, 0, 1)

            return image_tensor, label

        except Exception as e:
            if idx < 5 or idx % 1000 == 0:
                print(f"[ERROR] en índice {idx}: {e}")
            return None
