import carla
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image as PILImage
from threading import Lock
import time
from timm import create_model
import torch
from torchvision import transforms

# Cargar los modelos de TensorFlow para control en rectas y curvas
recta_model_path = "model_1/models_recta/best_model.keras"
curva_model_path = "model_1/models_curva/best_model.keras"
curvature_model_path = "/home/canveo/carla_ws/src/carla_dummy/model/efficientnet_model_2.pth"

# Configuración del dispositivo (usar GPU si está disponible)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_model_tf(model_path):
    return load_model(model_path)

# Cargar los modelos entrenados
recta_model = load_model_tf(recta_model_path)
curva_model = load_model_tf(curva_model_path)

# Cargar el modelo de predicción de curvatura (usando PyTorch)
class EfficientNetWithDropout(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(EfficientNetWithDropout, self).__init__()
        self.base_model = create_model('efficientnet_b0', pretrained=True, num_classes=2)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.base_model.classifier(x)
        return x

# Cargar el modelo de predicción de curvatura
curvature_model = EfficientNetWithDropout(dropout_rate=0.3)
curvature_model.load_state_dict(torch.load(curvature_model_path, map_location='cpu'))
curvature_model.eval()

# Preprocesar la imagen de entrada en tiempo real (igual que durante el entrenamiento)
def preprocess_image_from_camera(image_rgb, image_seg):
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb[200:-1, :], (200, 66))

    calzada_color = [128, 64, 128]
    mask = cv2.inRange(image_seg, np.array(calzada_color), np.array(calzada_color))
    image_seg_masked = np.zeros_like(image_seg)
    image_seg_masked[mask > 0] = [255, 255, 255]

    image_seg_gray = cv2.cvtColor(image_seg_masked, cv2.COLOR_BGR2GRAY)
    image_seg_gray = cv2.resize(image_seg_gray[200:-1, :], (200, 66))

    concatenated_image = np.dstack((image_rgb, image_seg_gray))

    return concatenated_image

# Funciones para predecir los controles del vehículo
def predict_controls(model, image_seg):
    input_tensor = tf.convert_to_tensor(image_seg, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Agregar batch dimension

    predictions = model(input_tensor, training=False)
    steer_value = predictions[0][0]
    return steer_value

# Función para predecir si es curva o recta (usando PyTorch)
def predict_curvature(model, img_array):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        img_pil = PILImage.fromarray(img_array)
        image_tensor = preprocess(img_pil).unsqueeze(0)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return 'curva' if predicted.item() == 1 else 'recta'

class DummyControl:
    def __init__(self, world):
        self.world = world
        self.camera_image = None
        self.segmented_image = None
        self.lock = Lock()

        # Crear una ventana de OpenCV para visualizar la simulación
        cv2.namedWindow("Lane Control", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Lane Control", 1024, 512)

        self.speed = 0.0
        self.average_status = 'desconocido'

    def reset_control_msg(self):
        self.control_msg = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": False,
            "reverse": False,
            "gear": 1,
            "manual_gear_shift": False,
        }

    def _setup_weather(self):
        # Configurar las condiciones climáticas
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=10.0,
            sun_azimuth_angle=70.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            fog_density=0.0,
            wetness=0.0,
        )
        self.world.set_weather(weather)

    def _clear_vehicles(self):
        # Eliminar otros vehículos en el entorno
        for actor in self.world.get_actors():
            if 'vehicle' in actor.type_id:
                actor.destroy()

    def _spawn_vehicle(self):
        # Obtener la biblioteca de planos y puntos de aparición
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        # Crear y retornar el vehículo ego
        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[79])
        return vehicle

    def image_callback(self, image, image_type):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
        if image_type == 'rgb':
            self.camera_image = array
            # Predecir la curvatura cada vez que se reciba una nueva imagen RGB
            self.average_status = predict_curvature(curvature_model, self.camera_image)
        elif image_type == 'segmentation':
            self.segmented_image = array

    def _setup_rgb_camera(self):
        # Configurar la cámara RGB
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90.0')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        camera.listen(lambda image: self.image_callback(image, 'rgb'))
        return camera

    def _setup_segmentation_camera(self):
        # Configurar la cámara de segmentación
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90.0')
        camera_bp.set_attribute('sensor_tick', '0.1')  # Frecuencia de captura
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        camera.listen(lambda image: self.image_callback(image, 'segmentation'))
        return camera

    def control_vehicle(self):
        if self.camera_image is not None and self.segmented_image is not None:
            processed_image = preprocess_image_from_camera(self.camera_image, self.segmented_image)
            predicted_steer = self.predict_controls(processed_image)
            steer, throttle, brake = self.descomponer_prediccion(predicted_steer)

            print(f"Predicciones - Steer: {steer}, Throttle: {throttle}, Brake: {brake}")

            control = carla.VehicleControl()
            control.throttle = float(throttle)
            control.steer = float(steer)
            control.brake = float(brake)
            self.ego_vehicle.apply_control(control)

    def descomponer_prediccion(self, steer):
        throttle = 0.2  # Valor fijo de throttle
        brake = 0
        return steer, throttle, brake

    def predict_controls(self, processed_image):
        if self.average_status == 'recta':
            return predict_controls(recta_model, processed_image)
        elif self.average_status == 'curva':
            return predict_controls(curva_model, processed_image)
        else:
            return 0.0

    def update_display(self):
        if self.camera_image is not None:
            # Redimensionar la imagen RGB para la visualización
            camera_image_resized = cv2.resize(self.camera_image, (1024, 512))

            # Añadir texto sobre la predicción actual
            cv2.putText(camera_image_resized, f"Curvatura: {self.average_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Mostrar solo la imagen RGB
            cv2.imshow("Lane Control", camera_image_resized)
            cv2.waitKey(1)  # Necesario para refrescar la ventana de OpenCV

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    try:
        world = client.load_world('Town04')
    except RuntimeError as e:
        print("No se pudo conectar al servidor CARLA. Asegúrese de que el simulador esté en ejecución.")
        raise e

    control = DummyControl(world)
    control._setup_weather()
    control._clear_vehicles()
    control.ego_vehicle = control._spawn_vehicle()
    control.rgb_camera = control._setup_rgb_camera()
    control.segmentation_camera = control._setup_segmentation_camera()

    # Mantener la simulación en marcha
    while True:
        world.tick()

        # Realizar control del vehículo
        control.control_vehicle()

        # Actualizar la pantalla independientemente de la llegada de nuevas imágenes
        control.update_display()

        # Esperar para mantener una tasa estable de actualización
        time.sleep(0.05)

if __name__ == '__main__':
    main()
