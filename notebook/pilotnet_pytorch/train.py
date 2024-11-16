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
from torch.utils.tensorboard import SummaryWriter
from utils.image_processing import get_images_array, balance_dataset, CustomDataset, train_transform, val_transform, recortar_pico, plot_balanced_data_distributions
from utils.model import PilotNetTwoOutput as PilotNet

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

    # Configuración de TensorBoard
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    images_npy_path = os.path.join(data_dir, "all_images.npy")
    labels_npy_path = os.path.join(data_dir, "all_labels.npy")
    
    if os.path.exists(images_npy_path) and os.path.exists(labels_npy_path):
        print("Cargando dataset desde archivos .npy...")
        all_images = np.load(images_npy_path)
        all_labels = np.load(labels_npy_path)
    else:
        print("Procesando dataset y guardando en archivos .npy...")
        labels_df = pd.read_csv(os.path.join(data_dir, 'combined_data.csv'))
        labels_df = labels_df[labels_df['curvarade'] == 'Curva']
        
        # Balanceo de `steer`
        balanced_steer_data = balance_dataset(labels_df, target_column='steer', desired_count=3000, max_samples=2000, bins=50)
        print(f"Total de imágenes después del balanceo de `steer`: {len(balanced_steer_data)}")
        
        # Balanceo de `throttle` sobre los datos ya balanceados en `steer`
        balanced_data = balance_dataset(balanced_steer_data, target_column='throttle', desired_count=6000, max_samples=5000)
        print(f"Total de imágenes después del balanceo de `throttle`: {len(balanced_data)}")
        
        # Recorte del pico de `steer` en el intervalo (-0.1, 0.1)
        balanced_data_recortado = recortar_pico(balanced_data, target_column="steer", interval=(-0.1, 0.1), max_samples=10000)
        print(f"Total de imágenes después del recorte de pico: {len(balanced_data_recortado)}")
        
        # Visualización de las distribuciones de datos: steer original, steer balanceado, steer recortado, y throttle balanceado
        plot_balanced_data_distributions(labels_df, balanced_steer_data, balanced_data_recortado, column_steer="steer", column_throttle="throttle")

            
        all_images, all_labels = get_images_array(balanced_data_recortado)
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
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
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
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Obtener las predicciones
            steer_pred, throttle_pred = model(inputs)
            steer_target, throttle_target = labels[:, 0].unsqueeze(1), labels[:, 1].unsqueeze(1)

            # Calcular la pérdida MSE para cada salida
            loss_steer = criterion(steer_pred, steer_target)
            loss_throttle = criterion(throttle_pred, throttle_target)
            
            # Combinar las pérdidas
            loss = loss_steer + loss_throttle
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mse += ((steer_pred - steer_target) ** 2).mean().item()
            running_mae += torch.abs(steer_pred - steer_target).mean().item()

        train_loss = running_loss / len(train_loader)
        train_mse = running_mse / len(train_loader)
        train_mae = running_mae / len(train_loader)
        
        train_losses.append(train_loss)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("MSE/Train", train_mse, epoch)
        writer.add_scalar("MAE/Train", train_mae, epoch)

        model.eval()
        val_loss, val_mse, val_mae = 0.0, 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                steer_pred, throttle_pred = model(inputs)
                steer_target, throttle_target = labels[:, 0].unsqueeze(1), labels[:, 1].unsqueeze(1)

                loss_steer = criterion(steer_pred, steer_target)
                loss_throttle = criterion(throttle_pred, throttle_target)

                loss = loss_steer + loss_throttle

                val_loss += loss.item()
                val_mse += ((steer_pred - steer_target) ** 2).mean().item()
                val_mae += torch.abs(steer_pred - steer_target).mean().item()

        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_mae /= len(val_loader)

        val_losses.append(val_loss)

        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("MSE/Validation", val_mse, epoch)
        writer.add_scalar("MAE/Validation", val_mae, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            early_stop_counter = 0  
        else:
            early_stop_counter += 1

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{train_loss},{train_mae},{train_mse},{val_loss},{val_mae},{val_mse}\n")

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
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

    writer.close()
