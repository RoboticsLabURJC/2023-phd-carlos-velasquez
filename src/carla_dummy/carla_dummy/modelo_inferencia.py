import tensorflow as tf
from tensorflow.keras.models import load_model
import carla
import cv2
import numpy as np
import time

# Definir y registrar la función 'mse'
# @tf.keras.utils.register_keras_serializable()
# def mse(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))

# # Registrar la función personalizada
# tf.keras.utils.get_custom_objects().update({'mse': mse})

# # Cargar el modelo entrenado
# model = load_model("/home/canveo/Projects/notebook/model/modelos nuevos/nuevo_dp05_monolitico_epoch_202.h5")

# Conectar al servidor CARLA y cargar el mundo
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
try:
    world = client.load_world('Town02')
except RuntimeError as e:
    print("Unable to connect to CARLA server. Please make sure the simulator is running.")
    raise e

# Configurar condiciones climáticas
weather = carla.WeatherParameters(
    cloudiness=80.0, 
    precipitation=0.0,
    sun_altitude_angle=80.0, 
    sun_azimuth_angle=0.0, 
    precipitation_deposits=80.0, 
    wind_intensity=0.0,
    fog_density=0.0,
    wetness=0.0,
)
world.set_weather(weather)

# Eliminar otros vehículos
for actor in world.get_actors():
    if 'vehicle' in actor.type_id:
        actor.destroy()

# Obtener la biblioteca de planos y puntos de generación
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# Generar el vehículo
vehicle_bp = bp_lib.find('vehicle.tesla.model3')
ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[79])

# Configurar la cámara RGB
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Ajustar la posición
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

# Inicializar la ventana de OpenCV
cv2.namedWindow("CARLA Manual Control", cv2.WINDOW_AUTOSIZE)

# Definir velocidad máxima en m/s (30 km/h ≈ 8.33 m/s)
MAX_SPEED = 8.33

# Variables para calcular FPS
fps = 0
prev_time = time.time()

def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Extraer canales RGB
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    return array

def image_callback(image):
    global prev_time, fps

    # Procesar la imagen
    frame = process_image(image)

    # Redimensionar y normalizar la imagen para el modelo
    input_image = cv2.resize(frame, (200, 66))  # Ajustar según las dimensiones de entrada del modelo
    input_image = input_image / 255.0  # Normalizar la imagen
    input_image = np.expand_dims(input_image, axis=0)

    # Predicción del modelo (solo dirección)
    # prediction = model.predict(input_image)
    # steer = prediction[0][0]  # Extraer el primer valor de la predicción

    # Acceder al estado de control actual
    control = carla.VehicleControl()
    control.steer = 0.1 #float(steer)
    control.throttle = 0.5  # Valor fijo de aceleración
    control.brake = 0.0     # Sin freno

    # Verificar y limitar la velocidad
    velocity = ego_vehicle.get_velocity()
    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # Convertir a km/h
    if speed > MAX_SPEED * 3.6:  # Convertir MAX_SPEED a km/h para comparación
        ego_vehicle.disable_constant_velocity()
        control.throttle = 0.0
        control.brake = 1.0
    else:
        ego_vehicle.enable_constant_velocity(carla.Vector3D(MAX_SPEED, 0, 0))

    ego_vehicle.apply_control(control)

    # Calcular FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # Mostrar información en el frame
    info_text = [
        f"Throttle: {control.throttle:.2f}",
        f"Brake: {control.brake:.2f}",
        f"Steer: {control.steer:.2f}",
        f"Speed: {speed:.2f} km/h",
        f"FPS: {fps:.2f}"
    ]

    # Asegurarse de que el frame esté en el formato correcto para la visualización
    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Posición para cada línea de texto
    y0, dy = 30, 30
    for i, line in enumerate(info_text):
        y = y0 + i*dy
        cv2.putText(display_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostrar el frame
    cv2.imshow("CARLA Manual Control", display_frame)
    cv2.waitKey(1)  # Añadir un pequeño retraso para permitir la actualización de la ventana

# Adjuntar el callback a la cámara
camera.listen(image_callback)

done = False
while not done:
    # Condición de salida
    if cv2.waitKey(1) & 0xFF == ord('q'):
        done = True

    world.tick()

# Limpiar
camera.stop()
camera.destroy()
ego_vehicle.destroy()
cv2.destroyAllWindows()
