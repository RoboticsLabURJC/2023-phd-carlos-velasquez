import carla
from carla import Vector3D

import argparse
import cv2
import numpy as np
import time
import os
import queue
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage
from timm import create_model
import random

# Definir el modelo EfficientNet con Dropout
class EfficientNetWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(EfficientNetWithDropout, self).__init__()
        self.base_model = create_model('efficientnet_b0', pretrained=True, num_classes=2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pooling = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.base_model.classifier(x)
        return x

class CarlaDataCollector:
    def __init__(self, output_folder_rgb, output_folder_seg, output_labels_file, model_path):
        self.output_folder_rgb = output_folder_rgb        
        self.output_folder_seg = output_folder_seg
        self.output_labels_file = output_labels_file

        # Crear carpetas de salida si no existen
        self._create_folders()

        # Crear cola unificada para sincronización de imágenes
        self.image_queue = queue.Queue()

        # Inicializar archivo CSV para etiquetas
        self._initialize_csv()
        
        # Configuración de parseador
        parser = argparse.ArgumentParser(description="Script para generar Dataset con imagenes RGB, seg y etiquetas con comandos de control")
        parser.add_argument('--town', type=str, default='Town02', help='Nombre del mapa a cargar')
        parser.add_argument('--spawn_point', type=str, help='Punto de spawn en formato "x,y,z,roll,pitch,yaw"')   
        
        args = parser.parse_args()   
        
        # Asignar los argumentos como atrirbuto de clase
        self.town_name = args.town
        self.spawn_point_str = args.spawn_point  

        # Conectar al servidor de CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.town_name)
        self.traffic_manager = self.client.get_trafficmanager()
        
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True  # Activar el modo síncrono
        self.settings.fixed_delta_seconds = 0.05  # Tiempo por cada tick (20 FPS)
        self.world.apply_settings(self.settings)

        # Configurar clima
        self._setup_weather()

        # Eliminar otros vehículos
        self._clear_vehicles()

        # Configurar vehículo y sensores
        self.ego_vehicle = self._spawn_vehicle(self.spawn_point_str)
        self.camera_rgb = self._setup_rgb_camera()
        self.camera_seg = self._setup_seg_camera()
        
        # Configurar el tráfico para que el vehículo haga caso omiso de la señalización
        self.traffic_manager.ignore_lights_percentage(self.ego_vehicle, 100.0)
        self.traffic_manager.ignore_signs_percentage(self.ego_vehicle, 100.0)
        self.traffic_manager.random_left_lanechange_percentage(self.ego_vehicle, 0.0)  # Evitar cambiar al carril izquierdo
        self.traffic_manager.random_right_lanechange_percentage(self.ego_vehicle, 100.0)  # Forzar cambio de carril a la derecha

        # Inicializar modelo de curvatura
        self.model = self._load_model(model_path)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Variables de control
        self.target_speed = 8.33  # 30 km/h en m/s
        self.max_throttle = 0.5 
        self.speed = 0.0
        self.rgb_image = None
        self.rgb_timestamp = None
        self.segmentation_image = None
        self.seg_timestamp = None
        self.curvature_label = None
        self.manual_mode = False
        self.manual_timer_end = 0  # temporizador piloto aleatorio
        self.next_manual_mode_time = 0
        
        # Parámetros control PID
        self.integral = 0.0
        self.previous_error = 0.0
        self.kp = 0.10  
        self.ki = 0.01 
        self.kd = 0.05
        
        # Umbral sincronización de timestamp RGB y segmentation
        self.THRESHOLD = 0.05
        
        # Probalbilidad de  entrar en modo aleatorio (0.3 -> 30%)
        self.manual_mode_probability = 0.
        
        self.max_throttle = 0.5
        self.max_brake = 0.5
        self.max_steer = 0.8

    def _create_folders(self):
        os.makedirs(self.output_folder_rgb, exist_ok=True)
        os.makedirs(self.output_folder_seg, exist_ok=True)

    def _initialize_csv(self):
        with open(self.output_labels_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image_rgb_name', 'image_seg_name', 'curvarade', 'steer', 'throttle', 'brake'])  # Cabecera del archivo CSV

    def _setup_weather(self):
        # Configurar las condiciones climáticas
        weather = carla.WeatherParameters(
            cloudiness=30.0,  # 0.0
            precipitation=30.0,  #0.0
            sun_altitude_angle=10.0, 
            sun_azimuth_angle=90.0,  # 70.0
            precipitation_deposits=20.0, # 1.0
            wind_intensity=0.0,
            fog_density=1.0,
            wetness=0.0,
        )
        self.world.set_weather(weather)

    def _clear_vehicles(self):
        # Eliminar otros vehículos en el entorno
        for actor in self.world.get_actors():
            if 'vehicle' in actor.type_id:
                actor.destroy()

    def _spawn_vehicle(self, spawn_string):
        # Obtener la biblioteca de planos y puntos de aparición
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        
        # Convertir la cadena en una lista de flotantes
        x, y, z, roll, pitch, yaw = [float(value) for value in spawn_string.split(",")]
        
         # Crear un objeto Transform a partir de la información proporcionada
        spawn_point = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
        )
        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        
        # spawn_points = self.world.get_map().get_spawn_points()
        # # Crear y retornar el vehículo ego
        # vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[])
        return vehicle

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
    
    def _setup_seg_camera(self):
        # Configurar la cámara de segemntación
        bp_lib = self.world.get_blueprint_library()
        camera_seg_bp = bp_lib.find('sensor.camera.semantic_segmentation')
        camera_seg_bp.set_attribute('image_size_x', '800')
        camera_seg_bp.set_attribute('image_size_y', '600')
        camera_seg_bp.set_attribute('fov', '90.0')
        camera_seg_bp.set_attribute('sensor_tick', '0.05')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera_seg = self.world.spawn_actor(camera_seg_bp, camera_transform, attach_to=self.ego_vehicle)
        camera_seg.listen(lambda image: self.image_callback(image, 'segmentation'))
        return camera_seg

    def _load_model(self, model_path):
        # Cargar el modelo de curvatura con la arquitectura EfficientNet con Dropout
        model = EfficientNetWithDropout(dropout_rate=0.3)
        model.load_state_dict(torch.load(model_path, map_location="cuda"))
        model = model.to("cuda")
        torch.cuda.empty_cache()  # Liberar memoria de la GPU
        model.eval()
        return model

    def process_image(self, image):
        # Convertir la imagen capturada en formato RGB
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        return array[:, :, :3]  # Devolver solo los canales RGB
    
    def process_segmentation_image(self, image):
        # Convertir la imagen de segmentatción con la paleta Cityscapes
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        return array[:, :, :3]  # delvuelve la imagen convertida en RGB
        
    def get_prediction(self, img_array):
        # Obtener la predicción de si la carretera es recta o curva
        with torch.no_grad():
            img_pil = PILImage.fromarray(img_array)  # Convertir de numpy array a PIL image
            image_tensor = self.preprocess(img_pil).unsqueeze(0).to("cuda")  # Preprocesar y mover a CUDA
            output = self.model(image_tensor)  # Obtener la salida del modelo
            _, predicted = torch.max(output, 1)
            return predicted.item()
    
    def run(self):
        # Habilitar piloto automático
        self.ego_vehicle.set_autopilot(True)
   
        # Sincronizar y guardar imágenes RGB
        done = False
        prev_time = time.time()
        while not done:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                done = True                   

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time
           
            # Obtener imágenes de la cola
            while not self.image_queue.empty():
                image, image_type, timestamp = self.image_queue.get()
                if image_type == 'rgb':
                    self.rgb_image = self.process_image(image)
                    self.rgb_timestamp = timestamp
                elif image_type == 'segmentation':
                    self.segmentation_image = self.process_segmentation_image(image)
                    self.seg_timestamp = timestamp

                # Realizar inferencia de curvatura
                if self.rgb_image is not None and self.segmentation_image is not None:
                    time_diff = abs(self.rgb_timestamp - self.seg_timestamp)
                    if time_diff < self.THRESHOLD:
                        road_status = self.get_prediction(self.rgb_image)
                        curvarade = 'Curva' if road_status == 1 else 'Recta'
                        timestamp_ms = int(timestamp * 1000)
                        rgb_image_name = f"frame_{timestamp_ms}_rgb.png"
                        seg_image_name = f"frame_{timestamp_ms}_seg.png"
                        
                        rgb_image_path = os.path.join(self.output_folder_rgb, rgb_image_name)
                        seg_image_path = os.path.join(self.output_folder_seg, seg_image_name)

                        # Obtener los valores de dirección, aceleración y frenado
                        control = self.ego_vehicle.get_control()
                        steer = control.steer
                        throttle = control.throttle
                        brake = control.brake
                        
                        velocity = self.ego_vehicle.get_velocity()
                        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6   # km/h
                                                
                        # Cambia a modo manual aletoriamente
                        if time.monotonic() >= self.next_manual_mode_time:
                            # cambiar modo manual durante 1 seg
                            self.manual_mode = True
                            self.ego_vehicle.set_autopilot(False)
                            self.manual_timer_end = time.monotonic() + 0.4              # 0.5 modo manual por un segundo
                            self.next_manual_mode_time = self.manual_timer_end + 8      # cada cuanto se cambia de modo
                            
                        if self.manual_mode and time.monotonic() > self.manual_timer_end:
                            self.manual_mode = False
                            self.ego_vehicle.set_autopilot(True)
                        
                        if not self.manual_mode:
                            # Guardar la imagen en disco
                            cv2.imwrite(rgb_image_path, self.rgb_image)
                            cv2.imwrite(seg_image_path, self.segmentation_image)

                            # Guardar la etiqueta de curvatura y controles en el archivo CSV
                            with open(self.output_labels_file, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([rgb_image_name, seg_image_name, curvarade, steer, throttle, brake])
                        else:
                            # piloto borracho
                            # control.steer = random.choice([-0.6, 0.6])
                            control.steer = random.uniform(-0.20, 0.20)  #0.2 steer y 0.1 throttle
                            control.throttle = 0.1
                            self.ego_vehicle.apply_control(control)

                        # Dibujar información en la imagen
                        img_rgb = self.rgb_image.copy()
                                              
                        # recuadro negro traslucido
                        x1, y1 = 35, 30
                        x2, y2 = 200, 160
                        alpha = 0.5
                        
                        overlay = img_rgb.copy()
                        
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)  # dibuja recuadro
                        cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0, img_rgb)
                        
                        cv2.putText(img_rgb, f"Calzada: {curvarade}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(img_rgb, f"Speed: {speed:.2f} km/h", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(img_rgb, f"Steer: {steer:.2f}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(img_rgb, f"Throttle: {throttle:.2f}", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(img_rgb, f"Brake: {brake:.2f}", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) 
                        cv2.putText(img_rgb, f"Modo: {'Manual' if self.manual_mode else 'Automatic'}", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.namedWindow('Curvature Classification', cv2.WINDOW_GUI_NORMAL) 
                        cv2.resizeWindow('Curvature Classification', 800, 600)                    
                        cv2.imshow("Curvature Classification", img_rgb)
                        cv2.waitKey(1)

                        # Resetear para esperar nuevas imágenes
                        self.rgb_image = None
                        self.segmentation_image = None

            # Mantener la velocidad constante                
            # self.control_speed(self.ego_vehicle, self.target_speed)
            
            # Avanzar un tick en el simulador
            self.world.tick()

        # Detener y limpiar
        self.camera_rgb.stop()
        self.camera_rgb.destroy()
        self.camera_seg.stop()
        self.camera_seg.destroy()
        self.ego_vehicle.destroy()
               
        cv2.destroyAllWindows()

    def control_speed(self, vehicle, target_speed):
        # Control de velocidad simple
        velocity = vehicle.get_velocity()
        self.speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        error = target_speed - self.speed
        
        self.integral += error * 0.05
        derivative = (error - self.previous_error) / 0.05
        
        control_signal = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.previous_error = error
        
        # control = carla.VehicleControl()
        # if self.speed < target_speed:
        #     control.throttle = 0.2
        # else:
        #     control.throttle = 0.0
        #     control.brake = 0.1
        control = carla.VehicleControl()
        control.throttle = np.clip(control_signal, 0.0, 1.0)
        control.brake = 0.0 if control_signal > 0 else np.clip(-control_signal, 0.0, 1.0)
        vehicle.apply_control(control)
        
    def image_callback(self, image, image_type):
        self.image_queue.put((image, image_type, image.timestamp))


def main():
    # Configurar las carpetas de salida y el archivo de etiquetas
    output_folder_rgb = "/home/canveo/carla_ws/dataset/imageRGB"
    output_folder_seg = "/home/canveo/carla_ws/dataset/imageSEG"
    output_labels_file = "/home/canveo/carla_ws/dataset/labels.csv"
    model_path = "/home/canveo/carla_ws/src/carla_dummy/model/efficientnet_model_2.pth"

    # Crear una instancia de la clase CarlaDataCollector
    data_collector = CarlaDataCollector(output_folder_rgb, output_folder_seg, output_labels_file, model_path)

    # Ejecutar el método run() para comenzar la recolección de datos
    data_collector.run()


if __name__ == "__main__":
    main()