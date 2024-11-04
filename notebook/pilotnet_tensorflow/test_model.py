import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from utils.image_processing import preprocess_image  
from tqdm import tqdm


def load_test_data(csv_path, base_path, img_shape):
    data_df = pd.read_csv(csv_path)
    images, labels = [], []

    rgb_dir = os.path.join(base_path, 'imageRGB')
    seg_dir = os.path.join(base_path, 'imageSEG')

    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Cargando imágenes de prueba"):
        rgb_path = os.path.join(rgb_dir, row['image_rgb_name'])
        seg_path = os.path.join(seg_dir, row['image_seg_name'])
        steer_value = row['steer'] 

        img = preprocess_image(rgb_path, seg_path)  
        images.append(img)
        labels.append(steer_value)
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def evaluate_model(model_path, csv_path, base_path, img_shape):
    model = load_model(model_path)
    print("Modelo cargado desde:", model_path)
    
    images, labels = load_test_data(csv_path, base_path, img_shape)

    predictions = model.predict(images)

    
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')

    
    plt.figure(figsize=(10, 5))
    plt.plot(labels, label="Valores Reales", color="blue")
    plt.plot(predictions, label="Predicciones", color="red")
    plt.title("Comparación de Predicciones y Valores Reales")
    plt.xlabel("Instancia")
    plt.ylabel("Steer")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Ruta del modelo entrenado (.h5 o .keras)")
    parser.add_argument("--csv_path", type=str, required=True, help="Ruta del archivo CSV con datos de prueba")
    parser.add_argument("--base_path", type=str, required=True, help="Ruta al directorio base del dataset")
    parser.add_argument("--img_shape", type=str, default="66,200,4", help="Forma de la imagen H,W,C")
    args = parser.parse_args()

    
    img_shape = tuple(map(int, args.img_shape.split(',')))
    
    
    evaluate_model(args.model_path, args.csv_path, args.base_path, img_shape)
