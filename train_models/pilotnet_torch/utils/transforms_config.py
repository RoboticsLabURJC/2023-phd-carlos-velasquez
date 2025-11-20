
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Affine(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
    A.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
    ),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.RandomShadow(p=0.3),
    A.RandomRain(p=0.2),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(0.05, 0.1),
        hole_width_range=(0.05, 0.1),
        fill=0,
        p=0.4
        ),
    # Normalizar a rango [-1, 1]
    A.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        max_pixel_value=255.0
    ),
    # ToTensorV2(),
])

# val_transform = A.Compose([ToTensorV2()])
val_transform = A.Compose([
    A.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        max_pixel_value=255.0
    ),
])

train_transform_desc = (
    "Affine, ColorJitter, Perspective, RandomShadow, RandomRain, "
    "GaussianBlur, GaussNoise, MotionBlur, RandomBrightnessContrast, CoarseDropout"
)
val_transform_desc = "No augmentation"

__all__ = [
    "train_transform",
    "val_transform",
    "train_transform_desc",
    "val_transform_desc"
]