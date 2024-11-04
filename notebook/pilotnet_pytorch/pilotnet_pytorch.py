import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor, as_completed


import os
import pandas as pd
import glob

def verificar_datos(dataset_base_path):
    dataset_dirs = sorted(glob.glob(os.path.join(dataset_base_path, 'dataset_*')))
    problemas = []

    for dataset_idx, dataset_dir in enumerate(dataset_dirs):
        labels_file = os.path.join(dataset_dir, 'labels.csv')
        if not os.path.exists(labels_file):
            print(f"Advertencia: No se encontró el archivo {labels_file}")
            continue

        labels_df = pd.read_csv(labels_file)
        for _, row in labels_df.iterrows():
            rgb_path = os.path.join(dataset_dir, 'imageRGB', row['image_rgb_name'])
            seg_path = os.path.join(dataset_dir, 'imageSEG', row['image_seg_name'])

            if not os.path.exists(rgb_path):
                problemas.append(f"Falta imagen RGB: {rgb_path}")

            if not os.path.exists(seg_path):
                problemas.append(f"Falta imagen SEG: {seg_path}")

    if problemas:
        print("Problemas encontrados:")
        for problema in problemas:
            print(problema)
    else:
        print("Todos los archivos están correctamente alineados.")


verificar_datos('/home/canveo/carla_ws/dataset_borracho/')


def balance_dataset(labels_df, target_column='steer', image_column='image', desired_count=9000, max_samples=8000, bins=30):
    """
    Balancea un dataset para que las distribuciones de 'steer' estén equilibradas.
    Se conserva la correspondencia entre las imágenes y los valores de 'steer'.
    """
    resampled = pd.DataFrame()
    bin_edges = np.linspace(labels_df[target_column].min(), labels_df[target_column].max(), bins + 1)  

    for i in range(len(bin_edges) - 1):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        
        part_df = labels_df[(labels_df[target_column] >= lower) & (labels_df[target_column] < upper)]

        if part_df.shape[0] > 0:
            
            if part_df.shape[0] > max_samples:
                part_df = part_df.sample(max_samples, random_state=42)

            
            repeat_factor = max(1, desired_count // part_df.shape[0])
            part_df_repeated = pd.concat([part_df] * repeat_factor, ignore_index=True)

            
            remaining_samples = desired_count - part_df_repeated.shape[0]
            if remaining_samples > 0:
                part_df_remaining = part_df.sample(remaining_samples, replace=True, random_state=1)
                part_df_repeated = pd.concat([part_df_repeated, part_df_remaining], ignore_index=True)

            
            resampled = pd.concat([resampled, part_df_repeated], ignore_index=True)
    print(f'Resampled data: {resampled.shape}')

    return resampled


def create_combined_dataset(dataset_base_path):
    combined_data = []
    dataset_dirs = sorted(glob.glob(os.path.join(dataset_base_path, 'dataset_*')))

    for dataset_idx, dataset_dir in enumerate(dataset_dirs):
        labels_file = os.path.join(dataset_dir, 'labels.csv')
        
        if os.path.exists(labels_file):
            labels_df = pd.read_csv(labels_file)
            if not labels_df.empty:
                
                labels_df['rgb_path'] = labels_df['image_rgb_name'].apply(
                    lambda name: os.path.join(dataset_dir, 'imageRGB', name)
                )
                labels_df['seg_path'] = labels_df['image_seg_name'].apply(
                    lambda name: os.path.join(dataset_dir, 'imageSEG', name)
                )

                
                labels_df = labels_df[['rgb_path', 'seg_path', 'curvarade', 'steer', 'throttle', 'brake']]

                combined_data.append(labels_df)
            else:
                print(f"Advertencia: El archivo {labels_file} está vacío.")
        else:
            print(f"Advertencia: No se encontró el archivo {labels_file}")

    if combined_data:
        
        combined_df = pd.concat(combined_data, ignore_index=True)

        
        combined_df.to_csv('combined_data.csv', index=False)
        print(f"Total de filas combinadas: {len(combined_df)}")
        print(combined_df.head())  
    else:
        print("Error: No se encontraron datos para combinar.")
        raise ValueError("No objects to concatenate")



dataset_base_path = '/home/canveo/carla_ws/dataset_borracho/'
create_combined_dataset(dataset_base_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 24, kernel_size=5, stride=2)  
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        
        self.fc1 = nn.Linear(1152, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        
        x = x.view(x.size(0), -1)
        
        
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  

        return x


def process_image(image_info):
    image_path, steer_value, seg_path = image_info  
    full_image_path = image_path
    full_seg_path = seg_path
    
    
    if not os.path.isfile(full_image_path):
        raise FileNotFoundError(f"La imagen {full_image_path} no se encuentra.")
    if not os.path.isfile(full_seg_path):
        raise FileNotFoundError(f"La imagen segmentada {full_seg_path} no se encuentra.")
    
    
    image_rgb = cv2.imread(full_image_path, cv2.IMREAD_COLOR)
    if image_rgb is None:
        raise ValueError(f"Error al leer la imagen {full_image_path}.")
    
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb[200:-1, :], (200, 66))
    
    
    image_seg = cv2.imread(full_seg_path)
    if image_seg is None:
        raise ValueError(f"Error al leer la imagen segmentada {full_seg_path}.")
    
    calzada_color = [128, 64, 128]
    mask = cv2.inRange(image_seg, np.array(calzada_color), np.array(calzada_color))
    image_seg_masked = np.zeros_like(image_seg)
    image_seg_masked[mask > 0] = [255, 255, 255]
    image_seg_gray = cv2.cvtColor(image_seg_masked, cv2.COLOR_BGR2GRAY)
    image_seg_gray = cv2.resize(image_seg_gray[200:-1, :], (200, 66))
    
    
    concatenated_image = np.concatenate((image_rgb, np.expand_dims(image_seg_gray, axis=-1)), axis=-1)

    return concatenated_image, steer_value


from concurrent.futures import ThreadPoolExecutor, as_completed

def get_images_array(df, max_workers=8):
    required_columns = ['rgb_path', 'seg_path', 'steer', 'curvarade']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"La columna '{column}' no se encuentra en el DataFrame.")
        
    
    
    
    steering_values = df['steer'].to_numpy()
    image_paths = df['rgb_path'].tolist()
    seg_paths = df['seg_path'].tolist()  

    
    if len(image_paths) != len(steering_values) or len(image_paths) != len(seg_paths):
        raise ValueError("El número de nombres de imágenes no coincide con el número de valores de dirección o rutas de segmentación.")
    
    
    image_info_list = [(image_paths[i], steering_values[i], seg_paths[i]) for i in range(len(image_paths))]

    images = []
    labels = []

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


import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A


train_transform = A.ReplayCompose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1, p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.RandomGamma(gamma_limit=(80, 150), p=0.4),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.4
    ),
    A.FancyPCA(alpha=0.1, p=0.4),
    A.GaussianBlur(blur_limit=(1, 3), p=0.4),
])

val_transform = A.ReplayCompose([])  

class DataGenerator(Dataset):
    def __init__(self, x_set, y_set, transform=None):
        self.x = x_set
        self.y = y_set
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx].copy()  

        
        if self.transform:
            
            transformed = self.transform(image=img[:, :, :3])
            img_rgb_transformed = transformed["image"]
            
            img_transformed = np.concatenate((img_rgb_transformed, img[:, :, 3:4]), axis=-1)

            
            if 'HorizontalFlip' in transformed['replay'] and transformed['replay']['HorizontalFlip'].get('applied', False):
                label = -label  

        else:
            img_transformed = img

        
        img_transformed = torch.tensor(img_transformed, dtype=torch.float32).permute(2, 0, 1)  
        label = torch.tensor(label, dtype=torch.float32)

        return img_transformed, label



dataset_base_path = '/home/canveo/carla_ws/dataset_borracho/'


combined_data = create_combined_dataset(dataset_base_path)



labels_df = pd.read_csv('combined_data.csv')


labels_df = labels_df[labels_df['curvarade'] == 'Recta']
print(f"Total de imágenes después del filtro 'Recta': {len(labels_df)}")

balanced_data = balance_dataset(labels_df, target_column='steer', desired_count=3000, max_samples=2000, bins=50)


plt.figure(figsize=(10, 5))
plt.hist(balanced_data['steer'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Steer Angle')
plt.ylabel('Frequency')
plt.title('Distribución del Dataset Balanceado')
plt.show()

balanced_data


all_images, all_labels = get_images_array(balanced_data)


X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)


train_dataset = DataGenerator(X_train, y_train, transform=train_transform)
val_dataset = DataGenerator(X_val, y_val, transform=val_transform)


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=10, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=10, pin_memory=True)

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Utilizando dispositivo: {device}')

model = PilotNet().to(device)


optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()  
scaler = torch.cuda.amp.GradScaler()  


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, patience=3, save_best=True):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improve_count = 0

    for epoch in range(epochs):
        
        model.train()
        running_loss = 0.0
        print(f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in tqdm(train_loader, desc="Entrenando", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f'Train Loss: {train_loss:.4f}')

        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validando", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        print(f'Validation Loss: {val_loss:.4f}')

        if save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Modelo guardado con Validation Loss: {val_loss:.4f}')
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f'Early stopping. No mejora en {patience} épocas consecutivas.')
            break

    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()



train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=100,       
    patience=10,      
    save_best=True   
)


import random
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset_path = 'combined_data.csv'
combined_data = pd.read_csv(dataset_path)


sampled_data = combined_data.sample(n=6, random_state=42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Utilizando dispositivo: {device}')



model = PilotNet()  
model.load_state_dict(torch.load('best_model.pth'))  
model.to(device)  
model.eval()  


def preprocess_image(rgb_path, seg_path):
    
    image_rgb = cv2.imread(rgb_path)
    if image_rgb is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen RGB: {rgb_path}")

    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (200, 66))

    
    image_seg = cv2.imread(seg_path)
    if image_seg is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen segmentada: {seg_path}")

    calzada_color = [128, 64, 128]
    mask = cv2.inRange(image_seg, np.array(calzada_color), np.array(calzada_color))
    image_seg_masked = np.zeros_like(image_seg)
    image_seg_masked[mask > 0] = [255, 255, 255]

    
    image_seg_gray = cv2.cvtColor(image_seg_masked, cv2.COLOR_BGR2GRAY)
    image_seg_gray = cv2.resize(image_seg_gray, (200, 66))

    
    image_rgb_pil = Image.fromarray(image_rgb)
    image_seg_pil = Image.fromarray(image_seg_gray)

    
    transform_to_tensor = transforms.ToTensor()
    tensor_rgb = transform_to_tensor(image_rgb_pil).unsqueeze(0)  
    tensor_seg = transform_to_tensor(image_seg_pil).unsqueeze(0)  

    
    combined_tensor = torch.cat((tensor_rgb, tensor_seg), dim=1)  

    
    combined_tensor = combined_tensor.squeeze(0)  

    return combined_tensor


def predict(model, rgb_path, seg_path):
    
    input_tensor = preprocess_image(rgb_path, seg_path)

    
    input_tensor = input_tensor.unsqueeze(0).to(device)  

    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    
    steer_value = prediction.item()
    return steer_value


def visualize_predictions(sampled_data, model):
    plt.figure(figsize=(15, 10))
    for i, (idx, row) in enumerate(sampled_data.iterrows()):
        if i >= 6:  
            break
        rgb_image_path = row['rgb_path']
        seg_image_path = row['seg_path']
        steer_value = row['steer']
        steer_prediction = predict(model, rgb_image_path, seg_image_path)
        
        
        image_rgb = cv2.imread(rgb_image_path)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        
        
        plt.subplot(2, 3, i + 1)
        plt.imshow(image_rgb)
        plt.title(f'Real: {steer_value:.4f}\nPredicción: {steer_prediction:.4f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


visualize_predictions(sampled_data, model)




