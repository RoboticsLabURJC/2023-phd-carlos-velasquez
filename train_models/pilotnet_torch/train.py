
import os
import argparse
import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json

from utils.processing import SegmentedLaneDataset3CH
from utils.pilotnet import PilotNet
# from utils.pilotnet_two_output import PilotNetTwoOutput
from utils.transforms_config import train_transform

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def train_model(csv_path, batch_size, epochs, lr, dropout, val_split, patience, num_workers):
    timestamp = datetime.now().strftime("pilotnet_%Y%m%d_%H%M")
    base_dir = os.path.join("experiments", timestamp)
    os.makedirs(base_dir, exist_ok=True)
    model_dir = os.path.join(base_dir, "trained_models")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    args_dict = {
        "csv_path": csv_path,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "dropout": dropout,
        "val_split": val_split,
        "num_workers": num_workers,
        "patience": patience,
        "timestamp": timestamp
    }
    with open(os.path.join(base_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    df = pd.read_csv(csv_path)[["seg_path", "steer", "throttle"]]
    # val_size = int(len(df) * val_split)
    # train_size = len(df) - val_size

    # generator = torch.Generator().manual_seed(42)
    # train_df, val_df = random_split(df, [train_size, val_size], generator=generator)
    # train_df = df.iloc[train_df.indices].reset_index(drop=True)
    # val_df = df.iloc[val_df.indices].reset_index(drop=True)
    
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=42, shuffle=True)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    base_path = "data/control_manual/"  # Cambiar según el dataset utilizado

    train_dataset = SegmentedLaneDataset3CH(train_df, transform=train_transform, base_path=base_path)    # dataset_dagger
    val_dataset = SegmentedLaneDataset3CH(val_df, transform=None, base_path=base_path)  # dataset_dagger

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    sample_image, _ = train_dataset[0]
    # sample_image.shape es (C, H, W)
    W, C, H = sample_image.shape
    print(f"Forma de la imagen de entrada: (C={C}, H={H}, W={W})")  
    img_shape = (C, H, W)  
    print(f"Forma de la imagen de entrada: {img_shape}")

    model = PilotNet(image_shape=img_shape, num_labels=2, dropout_rate=dropout).to(device)

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=log_dir)
    best_val_loss = float("inf")
    no_improvement = 0

    train_loss_mse_history = []
    train_loss_mae_history = []
    val_loss_mse_history = []
    val_loss_mae_history = [] 

    for epoch in range(epochs):
        model.train()
        running_loss_mse = 0.0
        running_loss_mae = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", colour="#007f00"):
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
           
            loss_mse = criterion_mse(outputs, targets)
            loss_mae = criterion_mae(outputs, targets)
            loss_mse.backward()
            optimizer.step()
            running_loss_mse += loss_mse.item()
            running_loss_mae += loss_mae.item()

        avg_train_loss_mse = running_loss_mse / len(train_loader)
        avg_train_loss_mae = running_loss_mae / len(train_loader)
        writer.add_scalar("Loss/Train_MSE", avg_train_loss_mse, epoch)
        writer.add_scalar("Loss/Train_MAE", avg_train_loss_mae, epoch)
       
        train_loss_mse_history.append(avg_train_loss_mse)
        train_loss_mae_history.append(avg_train_loss_mae)
        
        
        model.eval()
        running_val_loss_mse = 0.0
        running_val_loss_mae = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", colour="#00ff00"):
                inputs, targets = inputs.to(device).float(), targets.to(device).float()
                outputs = model(inputs)
                loss_mse = criterion_mse(outputs, targets)
                loss_mae = criterion_mae(outputs, targets)
                running_val_loss_mse += loss_mse.item()
                running_val_loss_mae += loss_mae.item()
                

        avg_val_loss_mse = running_val_loss_mse / len(val_loader)
        avg_val_loss_mae = running_val_loss_mae / len(val_loader)
        writer.add_scalar("Loss/Val_MSE", avg_val_loss_mse, epoch)
        writer.add_scalar("Loss/Val_MAE", avg_val_loss_mae, epoch)

        val_loss_mse_history.append(avg_val_loss_mse)
        val_loss_mae_history.append(avg_val_loss_mae)

        print(f"📉 Epoch {epoch+1}: Train MSE = {avg_train_loss_mse:.4f} | Val MSE = {avg_val_loss_mse:.4f} | Val MAE = {avg_val_loss_mae:.4f}")

        if avg_val_loss_mse < best_val_loss:
            best_val_loss = avg_val_loss_mse
            no_improvement = 0
            file_name= os.path.join(model_dir, f"pilotnet2out-epoch_{epoch+1:02d}-val_loss-{avg_val_loss_mse:.4f}.pth")
            torch.save(model.state_dict(), file_name)
            print("✅ Nuevo mejor modelo guardado.")
        else:
            no_improvement += 1
            print(f"⚠️ No mejora: {no_improvement}/{patience} (paciencia)")

        if no_improvement >= patience:
            print("🛑 Early stopping activado.")
            break

    torch.save(model.state_dict(), os.path.join(model_dir, "last_model.pth"))

    df = pd.DataFrame({
        "epoch": list(range(1, len(train_loss_mse_history) + 1)), 
        "train_loss_mse": train_loss_mse_history,
        "train_loss_mae": train_loss_mae_history,
        "val_loss_mse": val_loss_mse_history,
        "val_loss_mae": val_loss_mae_history,   
    })
    df.to_csv(os.path.join(base_dir, "loss_history.csv"), index=False)

    plt.figure(figsize=(8, 5))
    
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss_mse"], label="Train MSE")
    plt.plot(df["epoch"], df["val_loss_mse"], label="Val MSE")
    plt.plot(df["epoch"], df["val_loss_mae"], label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train & Validation Loss (MSE and MAE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, "loss_plot.png"))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train_model(
        args.csv_path,
        args.batch_size,
        args.epochs,
        args.lr,
        args.dropout,
        args.val_split,
        args.patience,
        args.num_workers,
    )
