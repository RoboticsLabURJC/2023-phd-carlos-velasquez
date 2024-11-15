from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

def pilotnet(img_shape=(66, 200, 3), learning_rate=0.0001):
    model = Sequential()
    model.add(tf.keras.layers.InputLayer(shape=img_shape))  # Usar InputLayer como primera capa
    model.add(BatchNormalization(epsilon=0.001, axis=-1))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", padding='valid'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu", padding='valid'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu", padding='valid'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding='valid'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding='valid'))
    
    model.add(Flatten())
    model.add(Dense(1164, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    
    # Configurar el optimizador
    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss="mse", metrics=['mse', 'mae'])
    return model


# Función para cargar el modelo EfficientNet con pesos preentrenados
def build_efficientnet_inference(dropout_rate=0.3):
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Reemplaza el clasificador completo
    x = Dropout(dropout_rate)(x)
    outputs = Dense(2, activation='softmax')(x)  # 2 clases para recta o curva
    model = Model(inputs=base_model.input, outputs=outputs)
    return model