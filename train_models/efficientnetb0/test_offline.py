import os
import json
import torch
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from processing import SegmentationCurvatureDataset
from transforms_config import val_transform
from datetime import datetime
from tqdm import tqdm


def load_model_pth(model_path, device):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_model_onnx(model_path):
    return ort.InferenceSession(model_path)


def predict_pth(model, dataloader, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return y_true, y_pred


def predict_onnx(onnx_model, dataloader):
    y_true, y_pred = [], []
    for images, labels in tqdm(dataloader, desc="Inferencia", unit="batch", colour="green"):
        ort_inputs = {onnx_model.get_inputs()[0].name: images.numpy()}
        ort_outs = onnx_model.run(None, ort_inputs)
        outputs = torch.tensor(ort_outs[0])
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())
    return y_true, y_pred


def save_results(y_true, y_pred, class_names, results_dir):
    # Guardar clasificación
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))

    # Guardar matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    # Gráficas de precisión, recall, F1 por clase
    for metric in ["precision", "recall", "f1-score"]:
        plt.figure()
        values = [report[class_name][metric] for class_name in class_names]
        plt.bar(class_names, values)
        plt.title(metric.capitalize())
        plt.ylabel(metric)
        plt.savefig(os.path.join(results_dir, f"{metric}_bar.png"))
        plt.close()


def main(config_path):
    with open(config_path) as f:
        config = json.load(f)

    # Crear carpeta de resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Dataset
    df = pd.read_csv(config["csv_path"])
    dataset = SegmentationCurvatureDataset(df, config["base_path"], transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    class_names = ["recta", "curva"]

    if config["model_path"].endswith(".pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_pth(config["model_path"], device)
        y_true, y_pred = predict_pth(model, dataloader, device)
    elif config["model_path"].endswith(".onnx"):
        model = load_model_onnx(config["model_path"])
        y_true, y_pred = predict_onnx(model, dataloader)
    else:
        raise ValueError("Formato de modelo no soportado (debe ser .pth o .onnx)")

    save_results(y_true, y_pred, class_names, results_dir)
    print(f"✅ Resultados guardados en {results_dir}")


if __name__ == "__main__":
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo JSON de configuración")
    args = parser.parse_args()
    main(args.config)
