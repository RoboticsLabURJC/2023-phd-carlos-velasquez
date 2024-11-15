import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
data = pd.read_csv("training_log.csv")

# Crear una figura y subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 18))

# Gráfico de Loss
axes[0].plot(data["epoch"], data["loss"], label="Training Loss")
axes[0].plot(data["epoch"], data["val_loss"], label="Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training and Validation Loss over Epochs")
axes[0].legend()
axes[0].grid(True)

# Gráfico de MAE
axes[1].plot(data["epoch"], data["mae"], label="Training MAE")
axes[1].plot(data["epoch"], data["val_mae"], label="Validation MAE")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE")
axes[1].set_title("Training and Validation MAE over Epochs")
axes[1].legend()
axes[1].grid(True)

# Gráfico de MSE
axes[2].plot(data["epoch"], data["mse"], label="Training MSE")
axes[2].plot(data["epoch"], data["val_mse"], label="Validation MSE")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("MSE")
axes[2].set_title("Training and Validation MSE over Epochs")
axes[2].legend()
axes[2].grid(True)

# Ajustar el espacio entre los gráficos
# Ajustar el espacio entre los gráficos
plt.tight_layout()
plt.savefig("training_plots.png")  # Guarda el gráfico en un archivo


