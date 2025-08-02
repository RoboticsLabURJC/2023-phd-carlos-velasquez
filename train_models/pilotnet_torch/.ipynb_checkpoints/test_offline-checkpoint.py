
import argparse
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.pilotnet import PilotNet
from utils.processing import SegmentedLaneDataset3CH

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True, help="Ruta al CSV de prueba")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo entrenado (.pth)")
    parser.add_argument("--outdir", type=str, default=None, help="Directorio de salida para resultados")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = SegmentedLaneDataset3CH(df_or_csv=args.test_csv, base_path=os.path.dirname(args.test_csv))

    # Obtener input_shape
    for item in test_dataset:
        if item is not None:
            input_shape = item[0].shape
            break

    model = PilotNet(image_shape=input_shape, num_labels=2).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    all_t = []
    all_steer_gt = []
    all_steer_pred = []
    all_throttle_gt = []
    all_throttle_pred = []
    total_loss_steer = 0
    total_loss_throttle = 0

    for idx in tqdm(range(len(test_dataset)), desc="Evaluando PilotNet"):
        item = test_dataset[idx]
        if item is None:
            continue
        input_tensor, label = item
        steer_gt, throttle_gt = label.tolist()

        input_tensor = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            steer_pred, throttle_pred = model(input_tensor).squeeze().cpu().numpy()

        all_t.append(idx)
        all_steer_gt.append(steer_gt)
        all_steer_pred.append(steer_pred)
        all_throttle_gt.append(throttle_gt)
        all_throttle_pred.append(throttle_pred)

        total_loss_steer += abs(steer_gt - steer_pred)
        total_loss_throttle += abs(throttle_gt - throttle_pred)

    result_str = (
        f"Promedio error absoluto STEER: {total_loss_steer / len(all_t):.5f}\n"
        f"Promedio error absoluto THROTTLE: {total_loss_throttle / len(all_t):.5f}"
    )
    print(result_str)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "results.txt"), "w") as f:
            f.write(result_str)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(all_t, all_steer_gt, label="GT Steer", color="b")
        plt.plot(all_t, all_steer_pred, label="Pred Steer", color="orange")
        plt.title("Comparación de STEER")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(all_t, all_throttle_gt, label="GT Throttle", color="b")
        plt.plot(all_t, all_throttle_pred, label="Pred Throttle", color="orange")
        plt.title("Comparación de THROTTLE")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "results.png"))
    else:
        plt.show()

if __name__ == "__main__":
    main()