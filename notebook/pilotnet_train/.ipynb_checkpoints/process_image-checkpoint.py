import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image(image_info):
    image_path, steer_value, base_path = image_info
    full_image_path = os.path.join(base_path, image_path)
    
    if not os.path.isfile(full_image_path):
        raise FileNotFoundError(f"La imagen {full_image_path} no se encuentra.")
    
    imagen = cv2.imread(full_image_path, cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError(f"Error al leer la imagen {full_image_path}.")
    
    img_cropped_resized = cv2.resize(imagen[200:-1, :], (200, 66))
    array_imagen = np.array(img_cropped_resized)
    
    return array_imagen, steer_value

def get_images_array(df, base_path, max_workers=8):
    required_columns = ['image_name', 'steer']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"La columna '{column}' no se encuentra en el DataFrame.")

    steering_values = df['steer'].to_numpy()
    image_paths = df['image_name'].tolist()

    if len(image_paths) != len(steering_values):
        raise ValueError("El número de nombres de imágenes no coincide con el número de valores de dirección.")

    imagenes = []
    labels = []

    image_info_list = [(image_paths[i], steering_values[i], base_path) for i in range(len(image_paths))]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(process_image, info): info for info in image_info_list}
        for future in as_completed(future_to_image):
            try:
                imagen, label = future.result()
                imagenes.append(imagen)
                labels.append(label)
            except Exception as exc:
                print(f'Error al procesar la imagen: {exc}')

    return np.array(imagenes), np.array(labels)
