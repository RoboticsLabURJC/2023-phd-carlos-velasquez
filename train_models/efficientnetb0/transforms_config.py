# file: transforms_config.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# train_transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.Normalize(mean=(0.5,), std=(0.5,)),
#     ToTensorV2()
# ])

# val_transform = A.Compose([
#     A.Normalize(mean=(0.5,), std=(0.5,)),
#     ToTensorV2()
# ])


image_size = 224

train_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
    A.RandomBrightnessContrast(p=0.5),
    A.ColorJitter(p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
