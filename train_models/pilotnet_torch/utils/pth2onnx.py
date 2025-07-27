import torch
import torch.nn as nn
import pandas as pd
from torchvision.models import efficientnet_v2_s, resnet18, ResNet18_Weights, EfficientNet_V2_S_Weights
from pilotnet import PilotNet

# ────────────────────────────────────────────────────────────
# 1) Construye la red EXACTAMENTE igual a la que usas
# net = efficientnet_v2_s(weights=None)
# net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)


net = PilotNet(image_shape=(66,200,3), num_labels=2, dropout_rate=0.3)

# (Si tu primera conv se cambió a 1 canal copia los pesos
#  aquí, tal como vimos antes.)

# ────────────────────────────────────────────────────────────
# 2) Carga el checkpoint
# ckpt_path = "experiments/efficientnet_v2_s_monolitico_Control_manual_20250704_1453/trained_models/efficientnet_control_manual.pth"
ckpt_path = "experiments/pilotnet_control_manual_20250703_1723/trained_models/pilotnet_control_manual.pth"
net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
net.eval()                       # modo inferencia

# ────────────────────────────────────────────────────────────
# 3) Prepara un dummy input con la MISMA shape que en producción
dummy = torch.zeros(1, 3, 66, 200, dtype=torch.float32) 

# ────────────────────────────────────────────────────────────
# 4) Exporta a ONNX
onnx_path = "pilotnet_control_manual.onnx"
torch.onnx.export(
    net,                    # modelo
    dummy,                  # ejemplo de entrada
    onnx_path,              # fichero destino
    export_params=True,     # guarda los pesos
    opset_version=17,       # 11 o superior (17 recomendado si usas ORT ≥1.17)
    input_names=["image"],  # nombres legibles
    output_names=["controls"],
)

print(f"Modelo exportado a {onnx_path}")
