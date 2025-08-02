import os
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from carla_msgs.msg import CarlaEgoVehicleControl
from std_msgs.msg import Bool, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
)
from utils.pilotnet import PilotNet  

import time

def load_model_by_name(name, device):
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "efficientnet":
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    elif name == "pilotnet":
        model = PilotNet(image_shape=(66, 200, 3), num_labels=2, dropout_rate=0.3)
    else:
        raise ValueError(f"Modelo desconocido: {name}")

    model.load_state_dict(torch.load(MODEL_PATHS[name], map_location=device))
    model.to(device)
    model.eval()
    return model


# SELECTED_MODEL = "resnet18"  # Cambiar a "efficientnet" si se desea usar EfficientNetB0
SELECTED_MODEL = "pilotnet"  # Cambiar a "resnet18" si se desea usar ResNet18
MODEL_PATHS = {
    "resnet18": "experiments/resnet18_20250621_1109/trained_models/last_model.pth",
    "efficientnet": "experiments/efficientnet_v2_s_20250621_2002/trained_models/efficientnet_v2_s-epoch_67-val_loss-0.0176.pth",
    #"pilotnet": "experiments/pilotnet_dagger_20250628_1625/trained_models/pilotnet_dagger.pth",
    "pilotnet": "experiments/pilotnet_control_manual_20250703_1723/trained_models/pilotnet_control_manual.pth"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_by_name(SELECTED_MODEL, device)


# Procesar imagen segmentada
def predict_controls(model, image_seg):
    # Preprocesamiento como en el entrenamiento
    calzada_color = [128, 64, 128]
    mask = cv2.inRange(image_seg, np.array(calzada_color), np.array(calzada_color))

    image_seg_masked = np.zeros_like(image_seg)
    image_seg_masked[mask > 0] = [255, 255, 255]

    mask_resized = cv2.resize(image_seg_masked[200:-1, :], (200, 66))
    mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
    rgb_like = cv2.merge([mask_gray, mask_gray, mask_gray])  # (66, 200, 3)

    # Convertir a tensor, normalizar y mover a dispositivo
    input_tensor = torch.tensor(rgb_like, dtype=torch.float32).permute(2, 0, 1)  # (3, 66, 200)
    input_tensor = input_tensor.unsqueeze(0)  # (1, 3, 66, 200)
    # input_tensor = input_tensor / 255.0  # Normalización 0–1
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)

    steer = prediction[0][0].item()
    throttle = prediction[0][1].item()
    return steer, throttle


# Nodo principal
class DummyControl(Node):
    def __init__(self):
        super().__init__("carla_dummy")
        self.bridge = CvBridge()
        self.camera_image = None
        self.segmented_image = None
        # self.lock = Lock()
        self.average_status = "desconocido"

        # Configuración de la ventana para visualizar la simulación
        cv2.destroyAllWindows()
        cv2.namedWindow("Lane Control", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Lane Control", 1024, 512)

        qos_profile = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Publicadores y suscriptores
        self.publisher_control = self.create_publisher(
            CarlaEgoVehicleControl,
            "/carla/ego_vehicle/vehicle_control_cmd_manual",
            qos_profile,
        )
        self.publisher_control_manual_override = self.create_publisher(
            Bool, "/carla/ego_vehicle/vehicle_control_manual_override", 10
        )
        self.publisher_autopilot = self.create_publisher(
            Bool, "/carla/ego_vehicle/enable_autopilot", 10
        )

        self.image_subscriber = self.create_subscription(
            Image, "/carla/ego_vehicle/rgb_front/image", self.vehicle_image_callback, 10
        )
        self.segmentation_image_subscriber = self.create_subscription(
            Image,
            "/carla/ego_vehicle/semantic_segmentation_front/image",
            self.segmentation_image_callback,
            10,
        )

        self.control_msg = CarlaEgoVehicleControl()
        self.reset_control_msg()
        self.timer = self.create_timer(1.0 / 40, self.control_vehicle)
        
        self.create_timer(0.5, self.init_vehicle_control, callback_group=None)         

        self.get_logger().info("Nodo DummyControl iniciado correctamente.")
        
    def init_vehicle_control(self):
        self.set_autopilot()
        self.set_control_manual_override()
        # self.get_logger().info("Se activó control manual (manual_override=True, autopilot=False)")


    def set_control_manual_override(self):
        self.publisher_control_manual_override.publish(Bool(data=True))

    def set_autopilot(self):
        self.publisher_autopilot.publish(Bool(data=False))

    def reset_control_msg(self):
        self.control_msg = CarlaEgoVehicleControl(
            header=Header(stamp=self.get_clock().now().to_msg()),
            throttle=0.0,
            steer=0.0,
            brake=0.0,
            hand_brake=False,
            reverse=False,
            gear=1,
            manual_gear_shift=False,
        )

    def vehicle_image_callback(self, image):
        # self.get_logger().info("Recibiendo imagen de la cámara")
        self.camera_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        self.update_display()

    def segmentation_image_callback(self, image):
        self.get_logger().info("Recibiendo imagen segmentada")
        self.segmented_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

    def control_vehicle(self):
        self.set_autopilot()
        self.set_control_manual_override()
        
        tic = time.perf_counter()

        # self.get_logger().info("Controlando el vehículo...")
        # self.get_logger().info(f"Imagen de cámara: {self.segmented_image_image.shape if self.segmented_image is not None else 'None'}")

        if self.camera_image is not None and self.segmented_image is not None:
            steer, throttle = predict_controls(
                model=model, image_seg=self.segmented_image
            )
            brake = 0.0  # Valor fijo de brake
            
            dt = (time.perf_counter() - tic) * 1000

            # self.get_logger().info(
            #     f"Predicciones - Steer: {steer}, Throttle: {throttle}, Brake: {brake}"
            # )

            self.control_msg.throttle = float(throttle)
            self.control_msg.steer = float(steer)
            self.control_msg.brake = float(brake)

            self.control_msg.header.stamp = self.get_clock().now().to_msg()

            self.publisher_control.publish(self.control_msg)
            print(f"Inference time: {dt:.2f} ms")


    def update_display(self):
        if self.camera_image is not None:
            camera_image_resized = cv2.resize(self.camera_image, (1024, 512))
            cv2.putText(
                camera_image_resized,
                f"Curvatura: {self.average_status}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Lane Control", camera_image_resized)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DummyControl()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
