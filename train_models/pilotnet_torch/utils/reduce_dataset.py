import pandas as pd
import os
import matplotlib.pyplot as plt


# Ruta del archivo original
csv_path = "data/dataset_dagger/pilotnet_data.csv"

# Número de muestras a tomar
num_samples = 35000

# Cargar el CSV
df = pd.read_csv(csv_path)
print(f"Total de filas en el dataset original: {len(df)}")

# Tomar una muestra aleatoria
sample_df = df.sample(n=num_samples, random_state=42)

# Construir ruta de salida en la misma carpeta
output_path = os.path.join("data/dataset_dagger", f"pilotnet_data_sample_{num_samples}.csv")

# Guardar el nuevo CSV reducido
sample_df.to_csv(output_path, index=False)

print(f"✅ Muestra guardada en: {output_path}")

bins = 100

# Visualización del balanceo
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(sample_df["steer"], bins=bins, color="skyblue", edgecolor="black")
plt.title("Distribución de 'steer'")
plt.xlabel("steer")
plt.ylabel("Frecuencia")

plt.subplot(1, 2, 2)
plt.hist(sample_df["throttle"], bins=bins, color="lightgreen", edgecolor="black")
plt.title("Distribución de 'throttle'")
plt.xlabel("throttle")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.savefig("data/dataset_dagger/histogram_sample_15000.png")
plt.show()