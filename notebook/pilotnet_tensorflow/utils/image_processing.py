import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from albumentations import ReplayCompose, HorizontalFlip, ColorJitter, RandomBrightnessContrast, RandomGamma, FancyPCA, GaussianBlur
from tensorflow.keras.utils import Sequence


train_transform = A.ReplayCompose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
        ),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomResizedCrop(height=66, width=200, scale=(0.8, 1.0), p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
    ],
    p=1.0,
)

val_transform = ReplayCompose([])


def balance_dataset(labels_df, target_column='steer', desired_count=3000, max_samples=2000, bins=50):
    resampled = pd.DataFrame()
    bin_edges = np.linspace(labels_df[target_column].min(), labels_df[target_column].max(), bins + 1)
    for i in range(len(bin_edges) - 1):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        part_df = labels_df[(labels_df[target_column] >= lower) & (labels_df[target_column] < upper)]
        if part_df.shape[0] > max_samples:
            part_df = part_df.sample(max_samples, random_state=42)
        repeat_factor = max(1, desired_count // part_df.shape[0])
        part_df_repeated = pd.concat([part_df] * repeat_factor, ignore_index=True)
        remaining_samples = desired_count - part_df_repeated.shape[0]
        if remaining_samples > 0:
            part_df_repeated = pd.concat([part_df_repeated, part_df.sample(remaining_samples, replace=True, random_state=1)], ignore_index=True)
        resampled = pd.concat([resampled, part_df_repeated], ignore_index=True)
    return resampled


def get_images_array(df, max_workers=8):
    
    image_info_list = [(df['rgb_path'].iloc[i], df['seg_path'].iloc[i], df['steer'].iloc[i]) for i in range(len(df))]
    images, labels = [], []
    
    for rgb_path, seg_path, steer_value in tqdm(image_info_list, desc="Cargando imágenes"):
        try:
            img = preprocess_image(rgb_path, seg_path)  
            images.append(img)
            labels.append(steer_value)
        except Exception as e:
            print(f"Error al procesar imagen: {e}")
    
    return np.array(images), np.array(labels)


def preprocess_image(rgb_path, seg_path):
    
    image_rgb = cv2.imread(rgb_path)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb[200:-1, :], (200, 66))
    
    
    image_seg = cv2.imread(seg_path)

    
    calzada_color = [128, 64, 128]
    mask = cv2.inRange(image_seg, np.array(calzada_color), np.array(calzada_color))
    image_seg_masked = np.zeros_like(image_seg)
    image_seg_masked[mask > 0] = [255, 255, 255]
    
    
    image_seg_gray = cv2.cvtColor(image_seg_masked, cv2.COLOR_BGR2GRAY)
    image_seg_gray = cv2.resize(image_seg_gray[200:-1, :], (200, 66))

    
    concatenated_image = np.dstack((image_rgb, image_seg_gray))

    return concatenated_image

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size=32, transform=None):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.transform = transform

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x_processed = []
        adjusted_batch_y = []
        
        for img, label in zip(batch_x, batch_y):
            
            rgb_img = img[:, :, :3]
            mask_img = img[:, :, 3:]

            
            if self.transform:
                transformed = self.transform(image=rgb_img)
                rgb_transformed = transformed["image"]

                
                if 'replay' in transformed and transformed['replay'].get('HorizontalFlip', {}).get('applied', False):
                    label = -label  
            else:
                rgb_transformed = rgb_img

            
            img_transformed = np.concatenate((rgb_transformed, mask_img), axis=-1)
            batch_x_processed.append(img_transformed)
            adjusted_batch_y.append(label)

        return np.array(batch_x_processed), np.array(adjusted_batch_y)

