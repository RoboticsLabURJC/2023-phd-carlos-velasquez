import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import albumentations as A
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

# Transformaciones de entrenamiento y validación
train_transform = A.ReplayCompose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.RandomResizedCrop(height=66, width=200, scale=(0.8, 1.0), p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
], p=1.0)

val_transform = A.ReplayCompose([])

# Clase CustomDataset para manejar los datos
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        steer_label, throttle_label = self.labels[idx]

        rgb_img = img[:, :, :3]
        mask_img = img[:, :, 3:]

        if self.transform:
            transformed = self.transform(image=rgb_img)
            rgb_transformed = transformed["image"]
        else:
            rgb_transformed = rgb_img

        img_transformed = np.concatenate((rgb_transformed, mask_img), axis=-1)
        img_transformed = torch.tensor(img_transformed, dtype=torch.float32).permute(2, 0, 1)
        
        # Retornar las etiquetas `steer` y `throttle`
        labels = torch.tensor([steer_label, throttle_label], dtype=torch.float32)

        return img_transformed, labels

# Función para procesar cada imagen
def process_image(info):
    rgb_path, seg_path, steer_value, throttle_value = info
    concatenated_image = preprocess_image(rgb_path, seg_path)
    return concatenated_image, (steer_value, throttle_value)

# Modificación en get_images_array para incluir throttle
def get_images_array(df, max_workers=8):
    image_info_list = [(df['rgb_path'].iloc[i], df['seg_path'].iloc[i], df['steer'].iloc[i], df['throttle'].iloc[i]) for i in range(len(df))]
    images, labels = [], []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(process_image, info): info for info in image_info_list}
        
        for future in tqdm(as_completed(future_to_image), total=len(future_to_image), desc="Cargando imágenes"):
            try:
                concatenated_image, label = future.result()
                images.append(concatenated_image)
                labels.append(label)
            except Exception as exc:
                print(f'Error al procesar la imagen: {exc}')

    return np.array(images), np.array(labels)

# Función para preprocesar la imagen
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

def preprocess_image_rt(rgb_image, seg_image):
    image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb[200:-1, :], (200, 66))

    calzada_color = [128, 64, 128]
    mask = cv2.inRange(seg_image, np.array(calzada_color), np.array(calzada_color))
    image_seg_masked = np.zeros_like(seg_image)
    image_seg_masked[mask > 0] = [255, 255, 255]

    image_seg_gray = cv2.cvtColor(image_seg_masked, cv2.COLOR_BGR2GRAY)
    image_seg_gray = cv2.resize(image_seg_gray[200:-1, :], (200, 66))

    concatenated_image = np.dstack((image_rgb, image_seg_gray))
    return concatenated_image

# Función para balancear el dataset
def balance_dataset(
    labels_df,
    target_column="steer",
    desired_count=6000,
    max_samples=5000,
    bins=30,
):
    """
    Balancea un dataset para que las distribuciones de 'target_column' estén equilibradas.
    """
    resampled = pd.DataFrame()
    bin_edges = np.linspace(
        labels_df[target_column].min(), labels_df[target_column].max(), bins + 1
    )

    for i in range(len(bin_edges) - 1):
        lower, upper = bin_edges[i], bin_edges[i + 1]

        part_df = labels_df[
            (labels_df[target_column] >= lower) & (labels_df[target_column] < upper)
        ]

        if part_df.shape[0] > 0:
            if part_df.shape[0] > max_samples:
                part_df = part_df.sample(max_samples, random_state=42)

            repeat_factor = max(1, desired_count // part_df.shape[0])
            part_df_repeated = pd.concat([part_df] * repeat_factor, ignore_index=True)

            remaining_samples = desired_count - part_df_repeated.shape[0]
            if remaining_samples > 0:
                part_df_remaining = part_df.sample(
                    remaining_samples, replace=True, random_state=1
                )
                part_df_repeated = pd.concat(
                    [part_df_repeated, part_df_remaining], ignore_index=True
                )

            resampled = pd.concat([resampled, part_df_repeated], ignore_index=True)

    return resampled

# Función para recortar el pico en la distribución del target
def recortar_pico(df, target_column="steer", interval=(-0.1, 0.1), max_samples=5000):
    """
    Recorta el pico en el intervalo especificado para el target_column, limitando el número de muestras a max_samples.
    """
    pico_df = df[
        (df[target_column] >= interval[0]) & (df[target_column] <= interval[1])
    ]

    if pico_df.shape[0] > max_samples:
        pico_df = pico_df.sample(max_samples, random_state=42)

    resto_df = df[(df[target_column] < interval[0]) | (df[target_column] > interval[1])]

    df_recortado = pd.concat([resto_df, pico_df], ignore_index=True)
    return df_recortado

import matplotlib.pyplot as plt

def plot_balanced_data_distributions(original_data, balanced_steer_data, recortado_data, column_steer="steer", column_throttle="throttle"):
    """
    Grafica las distribuciones de `steer` y `throttle` en los datos originales, 
    datos balanceados en `steer`, datos con `steer` recortado, y datos balanceados en `throttle`.
    """
    plt.figure(figsize=(18, 10))

    # Distribución de `steer` en los datos originales
    plt.subplot(2, 2, 1)
    plt.hist(original_data[column_steer], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Distribución Original de {column_steer}")
    plt.xlabel(column_steer)
    plt.ylabel("Frecuencia")

    # Distribución de `steer` en los datos balanceados
    plt.subplot(2, 2, 2)
    plt.hist(balanced_steer_data[column_steer], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Distribución Balanceada de {column_steer}")
    plt.xlabel(column_steer)
    plt.ylabel("Frecuencia")

    # Distribución de `steer` en los datos recortados
    plt.subplot(2, 2, 3)
    plt.hist(recortado_data[column_steer], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Distribución Recortada de {column_steer}")
    plt.xlabel(column_steer)
    plt.ylabel("Frecuencia")

    # Distribución de `throttle` en los datos balanceados
    plt.subplot(2, 2, 4)
    plt.hist(recortado_data[column_throttle], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Distribución Balanceada de {column_throttle}")
    plt.xlabel(column_throttle)
    plt.ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()
