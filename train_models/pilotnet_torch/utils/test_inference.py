import torch
from processing import SegmentedLaneDataset3CH
from pilotnet import PilotNet
from transforms_config import train_transform  # o None si entrenaste sin aug
import pandas as pd

csv_path = "/home/canveo/2023-phd-carlos-velasquez/train_models/pilotnet_torch/data/dataset/balanced_data.csv"
df = pd.read_csv(csv_path)
base_path = "data/dataset"  # el mismo que en train.py

dataset = SegmentedLaneDataset3CH(df, transform=train_transform, base_path=base_path)

model = PilotNet(image_shape=(3, 66, 200), num_labels=2, dropout_rate=0.3)
state = torch.load("experiments/pilotnet_dataset_nuevo_2/trained_models/last_model.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

for i in range(5):
    x, y = dataset[i]
    x = x.unsqueeze(0)
    with torch.no_grad():
        pred = model(x)[0].numpy()
    print(f"Ejemplo {i}:")
    print("  y_true =", y.numpy())
    print("  y_pred =", pred)
