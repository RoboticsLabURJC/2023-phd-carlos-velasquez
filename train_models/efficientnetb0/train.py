import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from processing import SegmentationCurvatureDataset
from transforms_config import train_transform, val_transform

def train(config_path):
    with open(config_path) as f:
        params = json.load(f)

    # Crear carpeta del experimento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_dir = os.path.join("experiments", f"efficientnetb0_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    model_dir = os.path.join(experiment_dir, "trained_models")
    os.makedirs(model_dir, exist_ok=True)

    # Guardar copia de configuración
    with open(os.path.join(experiment_dir, "params.json"), "w") as f_out:
        json.dump(params, f_out, indent=4)

    # Dataset
    df = pd.read_csv(params["csv_path"])
    val_size = int(len(df) * params["val_split"])
    train_size = len(df) - val_size
    train_df, val_df = random_split(df, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataset = SegmentationCurvatureDataset(train_df.dataset, params["base_path"], transform=train_transform)
    val_dataset = SegmentationCurvatureDataset(val_df.dataset, params["base_path"], transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modelo
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Binaria
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    best_val_loss = float("inf")
    patience_counter = 0

    train_losses, val_losses = [], []

    for epoch in range(params["epochs"]):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['epochs']} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{params['epochs']} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = correct / total

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print("✅ Modelo mejorado guardado")

            # Guardar también en ONNX
            dummy_input = torch.randn(1, 3, params["image_size"][0], params["image_size"][1]).to(device)
            onnx_path = os.path.join(model_dir, "best_model.onnx")
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=11
            )
            print("✅ Modelo exportado a ONNX")
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                print("🛑 Early stopping")
                break

    # Guardar CSV con pérdidas
    loss_df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses)+1)),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    loss_df.to_csv(os.path.join(experiment_dir, "loss_history.csv"), index=False)

    # Graficar curva de pérdida
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "loss_plot.png"))
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo JSON de configuración")
    args = parser.parse_args()
    train(args.config)
