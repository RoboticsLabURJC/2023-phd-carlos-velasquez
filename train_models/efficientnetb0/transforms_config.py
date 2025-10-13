# file: transforms_config.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Tamaño de entrada
image_size = 224

# Coordenadas del recorte inferior central (para imágenes 800x600)
CROP_X1 = (800 - image_size) // 2
CROP_Y1 = 600 - image_size
CROP_X2 = CROP_X1 + image_size
CROP_Y2 = 600

train_transform = A.Compose([
    # Recorte inferior central (mantiene geometría)
    A.Crop(x_min=CROP_X1, y_min=CROP_Y1, x_max=CROP_X2, y_max=CROP_Y2),

    # Aumentaciones geométricas y de color
    A.HorizontalFlip(p=0.5),
    A.Affine(translate_percent=(0.05, 0.05), scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.ColorJitter(p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),

    # Normalización y conversión a tensor
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Crop(x_min=CROP_X1, y_min=CROP_Y1, x_max=CROP_X2, y_max=CROP_Y2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
