#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.processing import SegmentedLaneDataset3CH
from utils.transforms_config import train_transform, val_transform
from utils.pilotnet import PilotNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train PilotNet on lane-following dataset")

    parser.add_argument("--csv_path", type=str, required=True,
                        help="Ruta al CSV balanceado (p.ej. data/dataset/balanced_data.csv)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15,
                        help="Paciencia para early stopping (en epochs sin mejora de val_loss)")
    parser.add_argument("--experiment_name", type=str, default="pilotnet",
                        help="Nombre base del experimento (para la carpeta de salida)")

    return parser.parse_args()


def create_experiment_dir(experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "trained_models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    # ------------------------------
    # 1. Cargar CSV y hacer split
    # ------------------------------
    df = pd.read_csv(args.csv_path)
    # Nos quedamos solo con lo que necesitamos
    df = df[["seg_path", "steer", "throttle"]]

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_split,
        random_state=42,
        shuffle=True,
    )

    # base_path = carpeta donde está el CSV (normalmente data/dataset)
    base_path = os.path.dirname(args.csv_path)
    print(f"[INFO] base_path (imágenes): {base_path}")

    train_dataset = SegmentedLaneDataset3CH(
        train_df,
        transform=train_transform,
        base_path=base_path,
    )
    val_dataset = SegmentedLaneDataset3CH(
        val_df,
        transform=val_transform,
        base_path=base_path,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ------------------------------
    # 2. Crear modelo PilotNet
    # ------------------------------
    sample_image, _ = train_dataset[0]  # (C,H,W) -> ya normalizada [-1,1]
    C, H, W = sample_image.shape
    print(f"[INFO] Forma de la imagen de entrada: (C={C}, H={H}, W={W})")
    
    import cv2
    import numpy as np

    img = train_dataset[0][0].permute(1,2,0).cpu().numpy()
    img_vis = ((img * 0.5) + 0.5) * 255
    cv2.imwrite("debug_input.png", img_vis.astype(np.uint8))
    print("Guardada debug_input.png")


    model = PilotNet(
        image_shape=(C, H, W),
        num_labels=2,
        dropout_rate=args.dropout,
    ).to(device)

    print(model)

    # ------------------------------
    # 3. Criterios de pérdida y optimizador
    # ------------------------------
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------
    # 4. Directorio del experimento
    # ------------------------------
    exp_dir = create_experiment_dir(args.experiment_name)
    models_dir = os.path.join(exp_dir, "trained_models")
    plots_dir = os.path.join(exp_dir, "plots")

    # Guardamos configuración
    config = vars(args).copy()
    config["device"] = str(device)
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[INFO] Configuración guardada en: {config_path}")

    # ------------------------------
    # 5. Loop de entrenamiento
    # ------------------------------
    best_val_loss = np.inf
    epochs_without_improvement = 0

    history = {
        "train_mse": [],
        "train_mae": [],
        "val_mse": [],
        "val_mae": [],
        "val_loss": [],
    }

    for epoch in range(args.epochs):
        model.train()
        running_train_mse = 0.0
        running_train_mae = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            mse_loss = criterion_mse(outputs, labels)
            mae_loss = criterion_mae(outputs, labels)

            # pérdida híbrida
            loss = 0.7 * mae_loss + 0.3 * mse_loss

            loss.backward()
            optimizer.step()

            running_train_mse += mse_loss.item() * images.size(0)
            running_train_mae += mae_loss.item() * images.size(0)

        epoch_train_mse = running_train_mse / len(train_loader.dataset)
        epoch_train_mae = running_train_mae / len(train_loader.dataset)

        # ---- Validación ----
        model.eval()
        running_val_mse = 0.0
        running_val_mae = 0.0
        running_val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                mse_loss = criterion_mse(outputs, labels)
                mae_loss = criterion_mae(outputs, labels)
                loss = 0.7 * mae_loss + 0.3 * mse_loss

                running_val_mse += mse_loss.item() * images.size(0)
                running_val_mae += mae_loss.item() * images.size(0)
                running_val_loss += loss.item() * images.size(0)

        epoch_val_mse = running_val_mse / len(val_loader.dataset)
        epoch_val_mae = running_val_mae / len(val_loader.dataset)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        history["train_mse"].append(epoch_train_mse)
        history["train_mae"].append(epoch_train_mae)
        history["val_mse"].append(epoch_val_mse)
        history["val_mae"].append(epoch_val_mae)
        history["val_loss"].append(epoch_val_loss)

        print(
            f"[Epoch {epoch+1:03d}/{args.epochs}] "
            f"Train MSE: {epoch_train_mse:.4f} | "
            f"Train MAE: {epoch_train_mae:.4f} | "
            f"Val MSE: {epoch_val_mse:.4f} | "
            f"Val MAE: {epoch_val_mae:.4f} | "
            f"Val Loss(hybrid): {epoch_val_loss:.4f}"
        )

        # ---- Chequeo de predicciones cada 5 epochs ----
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                print("  [Debug] Ejemplos de predicción en train_dataset:")
                for i in range(5):
                    x, y_true = train_dataset[i]
                    x = x.unsqueeze(0).to(device)
                    y_pred = model(x)[0].cpu().numpy()
                    print(f"    Ejemplo {i}: y_true={y_true.numpy()}  y_pred={y_pred}")
            model.train()

        # ---- Guardado de mejor modelo (early stopping) ----
        if epoch_val_loss < best_val_loss - 1e-5:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0

            best_model_path = os.path.join(models_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  [INFO] Nuevo mejor modelo guardado en {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(f"  [INFO] Epochs sin mejora: {epochs_without_improvement}/{args.patience}")

        if epochs_without_improvement >= args.patience:
            print("[INFO] Early stopping activado.")
            break

    # Guardar último modelo
    last_model_path = os.path.join(models_dir, "last_model.pth")
    torch.save(model.state_dict(), last_model_path)
    print(f"[INFO] Último modelo guardado en {last_model_path}")

    # ------------------------------
    # 6. Graficar pérdidas
    # ------------------------------
    epochs_range = range(1, len(history["train_mse"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_mse"], label="Train MSE")
    plt.plot(epochs_range, history["val_mse"], label="Val MSE")
    plt.plot(epochs_range, history["val_mae"], label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train & Validation Loss (MSE and MAE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, "loss_plot.png")
    plt.savefig(plot_path)
    print(f"[INFO] Gráfico de pérdidas guardado en {plot_path}")


if __name__ == "__main__":
    main()
