import os
import datetime
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.image_processing import get_images_array, balance_dataset, CustomDataset, train_transform, val_transform
from utils.model import PilotNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Utilizando dispositivo: {device}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Ruta al directorio del dataset")
    parser.add_argument("--num_epochs", type=int, default=100, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=128, help="Tamaño del lote")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Tasa de aprendizaje inicial")
    parser.add_argument("--patience", type=int, default=10, help="Número de épocas de paciencia para EarlyStopping")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    data_dir = args.data_dir
    patience = args.patience

    images_npy_path = os.path.join(data_dir, "all_images.npy")
    labels_npy_path = os.path.join(data_dir, "all_labels.npy")
    
    if os.path.exists(images_npy_path) and os.path.exists(labels_npy_path):
        print("Cargando dataset desde archivos .npy...")
        all_images = np.load(images_npy_path)
        all_labels = np.load(labels_npy_path)
    else:
        print("Procesando dataset y guardando en archivos .npy...")
        labels_df = pd.read_csv(os.path.join(data_dir, 'combined_data.csv'))
        labels_df = labels_df[labels_df['curvarade'] == 'Recta']
        
        balanced_data = balance_dataset(labels_df, target_column='steer', desired_count=4000, max_samples=2500, bins=50)   
        print(f"Total de imágenes en balanced_data: {len(balanced_data)}")
            
        all_images, all_labels = get_images_array(balanced_data)
        print(f"Total de imágenes cargadas: {len(all_images)}")
        print(f"Total de etiquetas cargadas: {len(all_labels)}")
        
        os.makedirs(data_dir, exist_ok=True)
        np.save(images_npy_path, all_images)
        np.save(labels_npy_path, all_labels)
        
    X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
    
    train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
    val_dataset = CustomDataset(X_val, y_val, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = PilotNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    early_stop_counter = 0
    best_val_loss = float("inf")

    csv_path = "training_log.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,loss,mae,mse,val_loss,val_mae,val_mse\n")

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_mae, running_mse = 0.0, 0.0, 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mse += ((outputs - labels) ** 2).mean().item()
            running_mae += torch.abs(outputs - labels).mean().item()

        train_losses.append(running_loss / len(train_loader))
        train_mse = running_mse / len(train_loader)
        train_mae = running_mae / len(train_loader)

        model.eval()
        val_loss, val_mse, val_mae = 0.0, 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_mse += ((outputs - labels) ** 2).mean().item()
                val_mae += torch.abs(outputs - labels).mean().item()

        val_losses.append(val_loss / len(val_loader))
        val_mse /= len(val_loader)
        val_mae /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            early_stop_counter = 0  
        else:
            early_stop_counter += 1

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{train_losses[-1]},{train_mae},{train_mse},{val_losses[-1]},{val_mae},{val_mse}\n")

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        print(f"Tasa de aprendizaje actual: {optimizer.param_groups[0]['lr']}")

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
