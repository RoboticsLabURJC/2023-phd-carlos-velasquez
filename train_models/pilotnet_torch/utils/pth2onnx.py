import torch
import torch.nn as nn
from pilotnet import PilotNet

net = PilotNet(image_shape=(3, 66, 200), num_labels=2, dropout_rate=0.3)

ckpt_path = "experiments/pilotnet_dataset_nuevo/trained_models/last_model.pth"
net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
net.eval()

dummy = torch.zeros(1, 3, 66, 200, dtype=torch.float32)

onnx_path = "experiments/pilotnet_dataset_nuevo/trained_models/pilotnet_dataset_nuevo.onnx"
torch.onnx.export(
    net,
    dummy,
    onnx_path,
    export_params=True,
    opset_version=17,
    input_names=["image"],
    output_names=["controls"],
    dynamic_axes={"image": {0: "batch_size"}, "controls": {0: "batch_size"}},
)

print(f"Modelo exportado a {onnx_path}")
