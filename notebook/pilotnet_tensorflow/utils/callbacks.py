# utils/callbacks.py
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import numpy as np
import sys

def progressbar(it, prefix="", size=60, out=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size * j / count)
        print(f"{prefix}[{'#' * x}{'.' * (size - x)}] {j}/{count}", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print(flush=True, file=out)

class PlotCallback(Callback):
    def __init__(self, annotations_val, images_val, plot_frequency=10):
        super().__init__()
        self.annotations_val = annotations_val
        self.images_val = images_val
        self.plot_frequency = plot_frequency

    def on_epoch_end(self, epoch, logs=None):
        if isinstance(epoch, int) and epoch % self.plot_frequency == 0: 
            print('Guardando gráficos...')

            # Valores reales y predicciones de `steer`
            x_true, y_predicted = [], []

            for i in progressbar(range(0, len(self.annotations_val), 50), "Calculando: ", 40):
                x_true.append(self.annotations_val[i])

                final_image = self.images_val[i] / 255.0
                final_image = np.expand_dims(final_image, axis=0)
                prediction = self.model.predict(final_image)

                y_predicted.append(prediction[0][0])  # Asumimos que steer es el primer valor en la salida del modelo

            # Gráfico para `steer`
            plt.figure(figsize=(20, 10))
            plt.plot(x_true, label='Steer Real', color="green")
            plt.plot(y_predicted, label='Steer Predicho', color="red")
            plt.xlabel("Instancia")
            plt.ylabel("Steering")
            plt.legend()
            plt.title(f'Steering Predictions - Epoch {epoch}')
            plt.savefig(f'plot_steering_epoch_{epoch}.png')
            plt.close()
