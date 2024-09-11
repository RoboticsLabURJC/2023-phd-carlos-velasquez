import os
import time

import pygame.draw_py

import rclpy
from rclpy.node import Node
from carla_msgs.msg import CarlaEgoVehicleStatus
from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
import pygame
import cv2
import numpy as np

import torch
import re
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
from torch import LongTensor
import albumentations as albu

import carla


class CarlaLanesDataset(Dataset):
    CLASSES = ['background', 'left_marker', 'right_marker']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        get_label_name = lambda fn: re.sub(".png", "_label.png", fn)
        self.masks_fps = [os.path.join(masks_dir, get_label_name(image_id)) for image_id in self.ids]

        # Convertir nombres de clase a valores de clase en las máscaras
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Leer datos
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # Aplicar aumentos
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Aplicar preprocesamiento
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, LongTensor(mask)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def get_validation_augmentation():
        return None

    @staticmethod
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    @staticmethod
    def get_preprocessing(preprocessing_fn):
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=CarlaLanesDataset.to_tensor),
        ]
        return albu.Compose(_transform)


class LanePredict(Node):
    ENCODER = 'efficientnet-b0'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d'
    DEVICE = 'cuda'

    def __init__(self):
        super().__init__('detect_lane')

        self.bridge = CvBridge()

        self.image_surface = None
        size = 1024, 512
        self.screen = pygame.display.set_mode(size=size)
        pygame.display.set_caption('Lane detection')

        self.speed_kmh = 0

        self.positions: list = []
        self.a, self.b, self.c = 0.0, 0.0, 0.0

        self.vehicle_status_subscriber = self.create_subscription(
            CarlaEgoVehicleStatus,
            '/carla/ego_vehicle/vehicle_status',
            self.status_callback,
            10
        )

        self.speed_value = 0.0

        self.subscription_img = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.image_callback,
            10
        )

        self.curvature_publisher = self.create_publisher(
            String,
            '/curvature_status',
            10
        )

        # Define la ruta al modelo entrenado
        model_path = '/home/canveo/Documents/carla_laneddetection/lane_detection.pth'

        # Carga el modelo entrenado en el dispositivo
        self.model = smp.FPN(
            encoder_name=self.ENCODER,
            encoder_weights=None,
            classes=len(CarlaLanesDataset.CLASSES),
            activation=self.ACTIVATION,
        ).to(self.DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.DEVICE)))
        self.model.eval()

        # Define la función de preprocesamiento para las nuevas imágenes
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.ENCODER, self.ENCODER_WEIGHTS)

         # Conexión al servidor de CARLA
        # self.client = carla.Client('localhost', 2000)
        # self.client.set_timeout(10.0)
        # self.world = self.client.get_world()
        # self.map = self.world.get_map()
        # # self.create_route()

        # self.curvatura_text = 'eee'

    def status_callback(self, msg):
        self.speed_kmh = msg.velocity * 3.6

    def lane_center(self, centery: int, right_mask: np.ndarray, left_mask: np.ndarray, threshold: float) -> int:
        i: int = 1023

        while i > 0:
            if right_mask[centery, i] >= threshold:
                break
            i -= 1

        z: int = 1023
        while z > 0:
            if left_mask[centery, z] >= threshold:
                break
            z -= 1

        centerx: int = (i + z) // 2

        return centerx
    
    def first_derivative(self, y):
        return 2*self.a*y + self.b
    
    def second_derivative(self):
        return 2*self.a
    
    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 512))

        # Aplicar preprocesamiento a la imagen
        preprocessed_image = CarlaLanesDataset.get_preprocessing(self.preprocessing_fn)(image=img)["image"]
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Agrega una dimensión de lote
        preprocessed_image = torch.tensor(preprocessed_image).to(self.DEVICE)

        start_time = time.time()

        # Realizar predicción con el modelo cargado
        with torch.no_grad():  # Desactivar el cálculo de gradientes
            output = self.model(preprocessed_image)

        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000

        # self.get_logger().info(f'Tiempo de inferencia: {inference_time_ms:.2f} ms')

        # Obtener máscara predicha
        predicted_mask = np.argmax(output.squeeze().cpu().numpy(), axis=0)

        left_mask = (predicted_mask == CarlaLanesDataset.CLASSES.index('left_marker')).astype(np.uint8) * 255
        right_mask = (predicted_mask == CarlaLanesDataset.CLASSES.index('right_marker')).astype(np.uint8) * 255
        background = (predicted_mask == CarlaLanesDataset.CLASSES.index('background')).astype(np.uint8) * 255

        left_mask_rgb = cv2.cvtColor(left_mask, cv2.COLOR_GRAY2RGB)
        right_mask_rgb = cv2.cvtColor(right_mask, cv2.COLOR_GRAY2RGB)

        # Asignar colores diferentes a cada clase
        left_mask_rgb[left_mask > 0.5] = [255, 0, 0]  
        right_mask_rgb[right_mask > 0.5] = [0, 0, 255] # == 255

        alpha = 0.2  # Transparencia de la máscara
        overlaid_img = img.copy()
        overlaid_img = cv2.addWeighted(overlaid_img, 1 - alpha, left_mask_rgb, alpha, 0)
        overlaid_img = cv2.addWeighted(overlaid_img, 1 - alpha, right_mask_rgb, alpha, 0)
        # overlaid_img = cv2.addWeighted(overlaid_img, 1 - alpha, background, alpha, 0)

        # Convertir la imagen fusionada a una superficie Pygame
        overlaid_img_surface = pygame.surfarray.make_surface(overlaid_img.swapaxes(0, 1))
        img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        background_surface = pygame.surfarray.make_surface(background.swapaxes(0, 1))

        self.screen.blit(background_surface, (0, 0))
        self.screen.blit(overlaid_img_surface, (0, 0))

       
        ### lineas horizontales
        y1, x1 = np.where(right_mask > 0.5)
        y2, x2 = np.where(left_mask > 0.5)

        self.positions: list = []
        for i in [290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390]:
            centerx: int = self.lane_center(i, right_mask, left_mask, 0.5)
            
            pygame.draw.line(overlaid_img_surface, (255, 255, 255), (0, i), (1024, i), 2)
            pygame.draw.circle(overlaid_img_surface, (0, 255, 0), (centerx, i), 2)
            
            self.positions.append((centerx, i))
        
        # reference_x: float = np.mean(np.array(self.positions))
        x_values = [pos[0] for pos in self.positions]
        reference_x = float(np.mean(x_values))
        pygame.draw.circle(overlaid_img_surface, (0, 255, 255), (int(reference_x), 330), 2)   

        # calculo de curvatura
        self.positions = np.array(self.positions)
        x = self.positions[:, 0]
        y = self.positions[:, 1] 

        coefficients = np.polyfit(y, x, 2)  # fit x = ay^2 + by +c
        self.a, self.b, self.c = coefficients

        y_eval = 256

        first_deriv = self.first_derivative(y_eval)
        second_deriv = self.second_derivative()

        radius_of_curvature = ((1 + first_deriv**2)**(3/2)) / np.abs(second_deriv)
        straight_threshold = 1000  

        
        if radius_of_curvature > straight_threshold:
            # print("El segmento es recto.")
            self.curvatura_text = 'recto'
            r, g, b = 0, 0, 255
        else:
            # print("El segmento es curvo.")
            self.curvatura_text = 'curvo'
            r, g, b = 255, 0, 0

        # publisher data curvature
        curvature_message = String()
        curvature_message.data = self.curvatura_text
        self.curvature_publisher.publish(curvature_message)

       

        # Mostrar la imagen original y la imagen fusionada verticalmente en la pantalla
        # self.screen.blit(overlaid_img_surface, (0, 0))
     
        self.screen.blit(overlaid_img_surface, (0, 0))

        # Visualizar la velocidad
        speed_text = f'Speed: {self.speed_kmh:.1f} km/h'
        inference_time_text = f'Tiempo de inferencia: {inference_time_ms:.2f} ms'
        curvature_text = f'Radio de curvatura: {radius_of_curvature:.2f}' 

        self.render_text(self.curvatura_text, speed_text, inference_time_text, curvature_text)

        # vehicle = self.world.get_actors().filter('vehicle.*')[0]
        # self.world.debug.draw_string(
        #     vehicle.get_location(), 
        #     'X',
        #     draw_shadow=False, 
        #     color=carla.Color(r, g, b), 
        #     life_time=1e6, 
        #     persistent_lines=True)


    def render_text(self, curvatura_text, speed_text, inference_time_text, curvature_text):
        font = pygame.font.SysFont(None, 30)

        text_surface_sp = font.render(speed_text, True, (255, 0, 0))
        text_surface_tm = font.render(inference_time_text, True, (255, 0, 0))
        text_surface_rc = font.render(curvature_text, True, (255, 0, 0))
        text_surface_cv = font.render(curvatura_text, True, (255, 0, 0))

        self.screen.blit(text_surface_sp, (10, 10))
        self.screen.blit(text_surface_tm, (10, 30))
        self.screen.blit(text_surface_rc, (10, 50))
        self.screen.blit(text_surface_cv, (512, 220))
        pygame.display.flip()


def main(args=None):
    pygame.init()
    rclpy.init(args=args)
    node = LanePredict()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
