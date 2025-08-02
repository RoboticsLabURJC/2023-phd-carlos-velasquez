import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, efficientnet_v2_s, ResNet18_Weights, EfficientNet_V2_S_Weights
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
import onnxruntime as ort
from torchvision import transforms
import cv2
from sklearn.metrics import root_mean_squared_error, mean_absolute_error 
from datetime import datetime


# MODEL_NAME ="efficientnet" 
MODEL_NAME ="pilotnet"
# MODEL_PATH = "experiments/resnet18_20250621_1109/trained_models/last_model.pth"
# MODEL_PATH = "experiments/efficientnet_v2_s_20250621_2002/trained_models/last_model.pth"
# MODEL_PATH = "experiments/pilotnet_control_manual_20250703_1723/trained_models/last_model.pth"
MODEL_PATH = "/home/canveo/2023-phd-carlos-velasquez/train_models/pilotnet_torch/model_onnx/pilotnet_control_manual.onnx"  # Ruta al modelo .pth o .onnx
CSV_PATH = "data/data_test/combined_data.csv"
BASE_IMG_DIR = "/home/canveo/2023-phd-carlos-velasquez/train_models/pilotnet_torch/data/data_test/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear subcarpeta en results con timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("results", f"test_{MODEL_NAME}_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXTENSION = os.path.splitext(MODEL_PATH)[1].lower()

# Imprime información del modelo
console = Console()
tabla = Table(title="Configuración", show_lines=True)
tabla.add_column("Campo", style="bold")
tabla.add_column("Valor")
tabla.add_row("Modelo", MODEL_NAME)
tabla.add_row("Ruta", MODEL_PATH)
# tabla.add_row("Dispositivo", DEVICE)
console.print(tabla)

def load_model(name, path, device):
    if EXTENSION == ".pth":
        if name == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, 2)
        elif name == "efficientnet":
            model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        elif name == "pilotnet":
            from utils.pilotnet import PilotNet
            model = PilotNet(image_shape=(66,200,3), num_labels=2, dropout_rate=0.3)
        else:
            raise ValueError(f"Modelo no reconocido: {name}")

        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model
    elif EXTENSION == ".onnx":
        # Cargar modelo ONNX
        providers = [('CUDAExecutionProvider', {})] if ort.get_available_providers().__contains__('CUDAExecutionProvider') else ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(path, providers=providers)
        
        # nombre de las entradas y salidas del modelo
        in_name = ort_session.get_inputs()[0].name
        out_name = ort_session.get_outputs()[0].name

        return ort_session

def preprocess_image(seg_path):
    full_path = os.path.join(BASE_IMG_DIR, seg_path)
    image = cv2.imread(full_path)

    if image is None:
        raise FileNotFoundError(f"❌ Imagen no encontrada: {full_path}")

    calzada_color = [128, 64, 128]
    mask = cv2.inRange(image, np.array(calzada_color), np.array(calzada_color))
    masked = np.zeros_like(image)
    masked[mask > 0] = [255, 255, 255]

    resized = cv2.resize(masked[200:-1, :], (200, 66))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    rgb_like = cv2.merge([gray, gray, gray])  # (66, 200, 3)
    
    tensor = torch.tensor(rgb_like, dtype=torch.float32).permute(2, 0, 1)
    return tensor

df = pd.read_csv(CSV_PATH)
model = load_model(MODEL_NAME, MODEL_PATH, DEVICE)

y_true, y_pred = [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    if EXTENSION == ".pth":
        image_tensor = preprocess_image(row["seg_path"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prediction = model(image_tensor)[0].cpu().numpy()
    else:  # ONNX
        image_tensor = preprocess_image(row["seg_path"]).unsqueeze(0).numpy()
        ort_inputs = {model.get_inputs()[0].name: image_tensor}
        prediction = model.run(None, ort_inputs)[0][0]

    label = np.array([row["steer"], row["throttle"]])
    y_true.append(label)
    y_pred.append(prediction)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


df_result = pd.DataFrame({
    "steer_true": y_true[:, 0],
    "steer_pred": y_pred[:, 0],
    "throttle_true": y_true[:, 1],
    "throttle_pred": y_pred[:, 1],
})
df_result.to_csv(os.path.join(OUTPUT_DIR, "offline_predictions.csv"), index=False)

mse_steer = root_mean_squared_error(y_true[:, 0], y_pred[:, 0])
mae_steer = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
mse_throttle = root_mean_squared_error(y_true[:, 1], y_pred[:, 1])
mae_throttle = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Steer MSE: {mse_steer:.4f}\n")
    f.write(f"Steer MAE: {mae_steer:.4f}\n")
    f.write(f"Throttle MSE: {mse_throttle:.4f}\n")
    f.write(f"Throttle MAE: {mae_throttle:.4f}\n")


labels = ["Steer", "Throttle"]
for i, label in enumerate(labels):
    plt.figure(figsize=(18, 5))
    plt.plot(y_true[:, i], label=f"Real {label}", color="blue", alpha=0.6)
    plt.plot(y_pred[:, i], label=f"Pred {label}", color="red", alpha=0.6)
    plt.title(f"Predicción vs Real - {label}")
    plt.xlabel("Índice")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{label.lower()}_plot.png"))
    plt.show()