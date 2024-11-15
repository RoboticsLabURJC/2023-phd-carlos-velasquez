import os
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus
from std_msgs.msg import Bool, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array


curvature_model_path = (
    "/home/canveo/Projects/efficientnetb0/best_efficientnet_model_with_dropout.h5"
)
recta_model_path = "/home/canveo/Projects/pilotnet_seg_mask_tensorflow/modelo_1/modelo_recta/modelo_pilotnet_recta.keras"
curva_model_path = "/home/canveo/Projects/pilotnet_seg_mask_tensorflow/modelo_1/modelo_curva/modelo_pilotnet_curva.keras"


def create_model(dropout_rate=0.3):
    base_model = EfficientNetB0(
        include_top=False, input_shape=(224, 224, 3), weights="imagenet"
    )
    base_model.trainable = True

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


curvature_model = create_model(dropout_rate=0.3)
curvature_model.load_weights(curvature_model_path)


recta_model = tf.keras.models.load_model(recta_model_path)
curva_model = tf.keras.models.load_model(curva_model_path)


def predict_curvature(model, img_rgb):
    img_resize = cv2.resize(img_rgb, (224, 224))
    img_array = img_to_array(img_resize) / 255.0
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return "recta" if predicted_class == 0 else "curva"


def process_image_rgb(image_seg):
    calzada_color = [128, 64, 128]
    mask = cv2.inRange(image_seg, np.array(calzada_color), np.array(calzada_color))

    image_seg_masked = np.zeros_like(image_seg)
    image_seg_masked[mask > 0] = [255, 255, 255]

    image_seg_rgb = cv2.resize(image_seg_masked[200:-1, :], (200, 66))

    image_seg_rgb = cv2.cvtColor(image_seg_rgb, cv2.COLOR_BGR2GRAY)
    image_seg_rgb = cv2.merge([image_seg_rgb, image_seg_rgb, image_seg_rgb])

    return image_seg_rgb


def predict_controls(model, image_seg):
    input_tensor = np.expand_dims(process_image_rgb(image_seg) / 255.0, axis=0)
    prediction = model.predict(input_tensor)
    return prediction[0][0]


class DummyControl(Node):
    def __init__(self):
        super().__init__("carla_dummy")
        self.bridge = CvBridge()
        self.camera_image = None
        self.segmented_image = None
        self.lock = Lock()
        self.average_status = "desconocido"

        cv2.namedWindow("Lane Control")
        cv2.resizeWindow("Lane Control", 800, 600)

        self.publisher_control = self.create_publisher(
            CarlaEgoVehicleControl,
            "/carla/ego_vehicle/vehicle_control_cmd_manual",
            10,
        )
        self.publisher_control_manual_override = self.create_publisher(
            Bool,
            "/carla/ego_vehicle/vehicle_control_manual_override",
            qos_profile=rclpy.qos.qos_profile_system_default,
        )
        self.publisher_autopilot = self.create_publisher(
            Bool, "/carla/ego_vehicle/enable_autopilot", 10
        )

        self.status_subscriber = self.create_subscription(
            CarlaEgoVehicleStatus,
            "/carla/ego_vehicle/vehicle_status",
            self.status_callback_speed,
            10,
        )

        self.image_subscriber = self.create_subscription(
            Image, "/carla_custom/rgb_front/image", self.vehicle_image_callback, 10
        )
        self.segmentation_image_subscriber = self.create_subscription(
            Image,
            "/carla_custom/semantic_segmentation_front/image",
            self.segmentation_image_callback,
            10,
        )

        self.speed = 0.0
        self.control_msg = CarlaEgoVehicleControl()
        self.reset_control_msg()
        self.timer = self.create_timer(1.0 / 40, self.control_vehicle)

    def set_control_manual_override(self):
        self.publisher_control_manual_override.publish(Bool(data=True))

    def set_autopilot(self):
        self.publisher_autopilot.publish(Bool(data=False))

    def status_callback_speed(self, msg):
        self.speed = msg.velocity * 3.6

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
        self.camera_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        self.update_display()

    def segmentation_image_callback(self, image):
        self.segmented_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

    def predict_controls(self):
        self.get_logger().info(f"Average: {self.average_status}")

        if self.average_status == "recta":
            return predict_controls(recta_model, self.segmented_image)
        elif self.average_status == "curva":
            return predict_controls(curva_model, self.segmented_image)
        else:
            return 0.0

    def descomponer_prediccion(self, steer):
        throttle = 0.2
        brake = 0
        return steer, throttle, brake

    def control_vehicle(self):
        self.set_autopilot()
        self.set_control_manual_override()

        if self.camera_image is not None:
            self.average_status = predict_curvature(curvature_model, self.camera_image)

        if self.camera_image is not None and self.segmented_image is not None:
            predicted_steer = self.predict_controls()
            self.get_logger().info(f"Predicted - Steer: {predicted_steer}")

            steer, throttle, brake = self.descomponer_prediccion(predicted_steer)

            self.control_msg.throttle = float(throttle)
            self.control_msg.steer = float(steer)
            self.control_msg.brake = float(brake)

            self.control_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_control.publish(self.control_msg)

    def update_display(self):
        if self.camera_image is not None:

            cv2.putText(
                self.camera_image,
                f"Curvatura: {self.average_status}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Lane Control", self.camera_image)
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
