import os
import datetime
import argparse
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from utils.model import pilotnet_model
from utils.image_processing import get_images_array, balance_dataset, DataGenerator, train_transform, val_transform
from utils.callbacks import PlotCallback  

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Ruta al directorio del dataset")
    parser.add_argument("--num_epochs", type=int, default=100, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=128, help="Tamaño del lote")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Tasa de aprendizaje para PilotNet")
    parser.add_argument("--img_shape", type=str, default="66,200,4", help="Forma de la imagen en H,W,C")  
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    img_shape = tuple(map(int, args.img_shape.split(',')))
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    data_dir = args.data_dir

    labels_df = pd.read_csv(os.path.join(data_dir, 'combined_data.csv'))
    labels_df = labels_df[labels_df['curvarade'] == 'Curva']  
    balanced_data = balance_dataset(labels_df, target_column='steer', desired_count=3000, max_samples=2000, bins=50)

   
    all_images, all_labels = get_images_array(balanced_data)
    
    train_gen = DataGenerator(all_images, all_labels, batch_size=batch_size, transform=train_transform)
    val_gen = DataGenerator(all_images, all_labels, batch_size=batch_size, transform=val_transform)

    model = pilotnet_model(img_shape=img_shape, learning_rate=learning_rate)
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        EarlyStopping(monitor="val_loss", patience=30, verbose=1),
        ModelCheckpoint("best_model.keras", monitor='val_loss', save_best_only=True, verbose=1),
        CSVLogger("training_log.csv"),
        PlotCallback(all_labels, all_images, plot_frequency=10) 
    ]

    model.fit(
        train_gen,
        epochs=num_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    score = model.evaluate(val_gen, verbose=0)
    print(f'Test loss: {score[0]}, MSE: {score[1]}, MAE: {score[2]}')

    model.save("pilotnet_model.keras")
