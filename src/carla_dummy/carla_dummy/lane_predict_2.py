import os
import time

import rclpy
from rclpy.node import Node
from carla_msgs.msg import CarlaEgoVehicleStatus
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

import torch
import re
# import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
from torch import LongTensor
import albumentations as albu

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

        cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Lane Detection", 1024, 512)

        # self.image_surface = None
        # size = 1024, 512
        # self.screen = pygame.display.set_mode(size=size)
        # pygame.display.set_caption('Lane detection')

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

    def status_callback(self, msg):
        self.speed_kmh = msg.velocity * 3.6

    def lane_center(self, y, left_line_points, right_line_points):
        left_x = int(np.polyval(left_line_points, y))
        right_x = int(np.polyval(right_line_points, y))
        center_x = (left_x + right_x) // 2
        return center_x

    def first_derivative(self, y):
        return 2 * self.a * y + self.b

    def second_derivative(self):
        return 2 * self.a

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
        right_mask_rgb[right_mask > 0.5] = [0, 0, 255]

        alpha = 0.3  # Transparencia de la máscara
        overlaid_img = img.copy()
        overlaid_img = cv2.addWeighted(overlaid_img, 1 - alpha, left_mask_rgb, alpha, 0)
        overlaid_img = cv2.addWeighted(overlaid_img, 1 - alpha, right_mask_rgb, alpha, 0)

        # # polyfit
        left_line_x, right_line_x = self.draw_lane_lines(overlaid_img, left_mask, right_mask)
      
        ### lineas horizontales
        self.positions = []
        for i in range(290, 400, 10):
            # centerx = self.lane_center(i, right_mask, left_mask, 0.5)
            centerx = self.lane_center(i, right_line_x, left_line_x)
            cv2.line(overlaid_img, (0, i), (1024, i), (255, 255, 255), 1)
            cv2.circle(overlaid_img, (centerx, i), 2, (0, 255, 0), -1)
            self.positions.append((centerx, i))


        x_values = [pos[0] for pos in self.positions]
        reference_x = float(np.mean(x_values))
        cv2.circle(overlaid_img, (int(reference_x), 330), 2, (0, 255, 255), -1)

        # self.draw_stable_cross_lines(overlaid_img, left_mask, right_mask)
        self.calculate_curvature()

        speed_text = f'Speed: {self.speed_kmh:.1f} km/h'
        inference_time_text = f'Tiempo de inferencia: {inference_time_ms:.2f} ms'
        curvature_text = f'Radio de curvatura: {self.radius_of_curvature:.2f}'

        self.render_text(overlaid_img, self.curvatura_text, speed_text, inference_time_text, curvature_text)

        cv2.imshow("Lane Detection", cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def calculate_curvature(self):
        self.positions = np.array(self.positions)
        x = self.positions[:, 0]
        y = self.positions[:, 1]

        coefficients = np.polyfit(y, x, 2)  # Ajustar una curva polinómica de segundo grado
        self.a, self.b, self.c = coefficients

        y_eval = 256

        first_deriv = self.first_derivative(y_eval)
        second_deriv = self.second_derivative()

        if second_deriv == 0:
            self.radius_of_curvature = float('inf')  # O elige un valor grande para indicar una curva infinita (recta)
        else:
            self.radius_of_curvature = ((1 + first_deriv ** 2) ** (3 / 2)) / np.abs(second_deriv)
    

        # self.radius_of_curvature = ((1 + first_deriv ** 2) ** (3 / 2)) / np.abs(second_deriv)
        # Define threshold for curvature categories
        straight_threshold = 1000

        if self.radius_of_curvature > straight_threshold:
            # print("El segmento es recto.")
            self.curvatura_text = f'recto'
        else:
            # print("El segmento es curvo.")
            self.curvatura_text = f'curvo'

    def render_text(self, image, curvatura_text, speed_text, inference_time_text, curvature_text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # Red color in BGR
        thickness = 1

        # Position coordinates for each text string
        position_sp = (10, 30)
        position_tm = (10, 45)
        position_rc = (10, 60)
        position_cv = (512, 250)

        # Render the text onto the image
        cv2.putText(image, speed_text, position_sp, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(image, inference_time_text, position_tm, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(image, curvature_text, position_rc, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(image, curvatura_text, position_cv, font, font_scale, color, thickness, cv2.LINE_AA)

        # Display the final image
        # cv2.imshow("Lane Detection with Text", image)
        cv2.waitKey(1)

    def draw_lane_lines(self, image, left_mask, right_mask):
        # Obtener las coordenadas de los píxeles que forman las marcas de carril
        left_y, left_x = np.where(left_mask > 0.5)
        right_y, right_x = np.where(right_mask > 0.5)

        # Ajustar una línea polinómica de primer grado (recta) a los puntos
        left_fit = np.polyfit(left_y, left_x, 1) if len(left_x) > 0 else None
        right_fit = np.polyfit(right_y, right_x, 1) if len(right_x) > 0 else None

        # Dibujar las líneas ajustadas sobre la imagen original
        y_max = image.shape[0]
        y_min = 0

        left_line_points = []
        right_line_points = []

        left_line_poly = []
        right_line_poly = []


        if left_fit is not None:
            left_line_y = np.linspace(y_min, y_max, y_max - y_min)
            left_line_x = np.polyval(left_fit, left_line_y).astype(int)
            left_line_poly = np.polyfit(left_line_y.astype(int), left_line_x, 1)
            left_line_points = list(zip(left_line_x, left_line_y.astype(int)))
            for x, y in left_line_points:
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # Línea roja

        if right_fit is not None:
            right_line_y = np.linspace(y_min, y_max, y_max - y_min)
            right_line_x = np.polyval(right_fit, right_line_y).astype(int)
            right_line_poly = np.polyfit(right_line_y.astype(int), right_line_x, 1)
            right_line_points = list(zip(right_line_x, right_line_y.astype(int)))
            for x, y in right_line_points:
                cv2.circle(image, (x, y), 2, (255, 0, 0), -1)  # Línea azul

        return left_line_poly, right_line_poly

    

def main(args=None):
    rclpy.init(args=args)
    node = LanePredict()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
