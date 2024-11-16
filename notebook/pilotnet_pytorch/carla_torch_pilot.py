import carla
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image as PILImage
from threading import Lock
import time
from timm import create_model
from utils.image_processing import preprocess_image
from utils.model import PilotNetTwoOutput
import os

# Deshabilitar la comprobación automática de actualizaciones
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Cargar los modelos de PyTorch para control en rectas y curvas
recta_model_path = "model_2/model_recta/best_model.pth"
curva_model_path = "model_1/model_curva/best_model.pth"
curvature_model_path = "/home/canveo/carla_ws/src/carla_dummy/model/efficientnet_model_2.pth"

# Dispositivo de inferencia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicializa los modelos y carga los pesos entrenados
model_recta = PilotNetTwoOutput().to(device)
model_curva = PilotNetTwoOutput().to(device)

model_recta.load_state_dict(torch.load(recta_model_path, map_location=device))
model_curva.load_state_dict(torch.load(curva_model_path, map_location=device))

model_recta.eval()
model_curva.eval()

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

curvature_model = EfficientNetWithDropout(dropout_rate=0.3).to(device)
curvature_model.load_state_dict(torch.load(curvature_model_path, map_location=device))
curvature_model.eval()

# Funciones para predecir los controles del vehículo
def predict_controls(model, input_tensor):
    # Pasar los datos al modelo y hacer la predicción
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # `prediction` es un tensor con el valor de steer
    # steer_value = prediction.item()
    return prediction[0][0].item()



# Función para predecir si es curva o recta (usando PyTorch)
def predict_curvature(model, img_array):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        img_pil = PILImage.fromarray(img_array)
        image_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return 'curva' if predicted.item() == 1 else 'recta'

# Actualización de la función preprocess_image_rt para evitar el error de variable no asignada
def preprocess_image_rt(rgb_image, seg_image):
    image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb[200:-1, :], (200, 66))

    calzada_color = [128, 64, 128]
    mask = cv2.inRange(seg_image, np.array(calzada_color), np.array(calzada_color))
    image_seg_masked = np.zeros_like(seg_image)
    image_seg_masked[mask > 0] = [255, 255, 255]

    image_seg_gray = cv2.cvtColor(image_seg_masked, cv2.COLOR_BGR2GRAY)
    image_seg_gray = cv2.resize(image_seg_gray[200:-1, :], (200, 66))

    concatenated_image = np.dstack((image_rgb, image_seg_gray))
    return concatenated_image

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
            processed_image = preprocess_image_rt(self.camera_image, self.segmented_image)
            processed_image_tensor = torch.tensor(processed_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) 
            print(f"Dimensiones del tensor antes de predecir los controles: {processed_image_tensor.shape}")
            
            predicted_steer = self.predict_controls(processed_image_tensor)
            
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
            return predict_controls(model_recta, processed_image)
        elif self.average_status == 'curva':
            return predict_controls(model_curva, processed_image)
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
