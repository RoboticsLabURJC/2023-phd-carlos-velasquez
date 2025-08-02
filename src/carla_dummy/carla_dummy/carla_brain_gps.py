import rclpy
from rclpy.node import Node
from carla_msgs.msg import CarlaEgoVehicleControl
from std_msgs.msg import Bool, Header
from sensor_msgs.msg import Image, NavSatFix
from cv_bridge import CvBridge
import cv2
import numpy as np
from keras.models import load_model
from threading import Thread, Lock
import tensorflow as tf
from geopy.distance import geodesic  


@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

model_path = "/home/canveo/Projects/notebook/wandb_train/logs/experiment_20240906_190454/trained_model.h5"
model = tf.keras.models.load_model(model_path, custom_objects={'mse': mse})

class DummyControl(Node):
    def __init__(self):
        super().__init__('carla_dummy')

        self.bridge = CvBridge()
        self.camera_image = None
        self.lock = Lock()

        cv2.namedWindow("Lane Control", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Lane Control", 1024, 512)

        self.publisher_control = self.create_publisher(CarlaEgoVehicleControl, "/carla/ego_vehicle/vehicle_control_cmd_manual", 10)
        self.publisher_control_manual_override = self.create_publisher(Bool, "/carla/ego_vehicle/vehicle_control_manual_override", qos_profile=rclpy.qos.qos_profile_system_default)
        self.publisher_autopilot = self.create_publisher(Bool, "/carla/ego_vehicle/enable_autopilot", 10)

        self.image_subscriber = self.create_subscription(Image, "/carla/ego_vehicle/rgb_front/image", self.vehicle_image_callback, 10)

        self.gnss_subscriber = self.create_subscription(NavSatFix, "/carla/ego_vehicle/gnss", self.gnss_callback, 10)

        self.control_msg = CarlaEgoVehicleControl()
        self.reset_control_msg()

        self.prediction_thread = Thread(target=self.predict_steer)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()

        self.timer = self.create_timer(1.0 / 40, self.control_vehicle)

        self.auto_mode = False  # manual/automatico
        self.start_gnss = None  
        self.distance_threshold = 5.0  # distancia que recorre en automatico

    def reset_control_msg(self):
        self.control_msg.header = Header()
        self.control_msg.throttle = 0.0
        self.control_msg.steer = 0.0
        self.control_msg.brake = 0.0
        self.control_msg.hand_brake = False
        self.control_msg.reverse = False
        self.control_msg.gear = 1
        self.control_msg.manual_gear_shift = False

    def vehicle_image_callback(self, image):
        with self.lock:
            img = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
            self.camera_image = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            self.camera_image = cv2.resize(self.camera_image[200:-1, :], (200, 66))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (1024, 512))  

            cv2.imshow("Lane Control", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

    def gnss_callback(self, msg):
        lat = msg.latitude
        lon = msg.longitude
        alt = msg.altitude

        if abs(lat + 0.001793) < 0.00001 and abs(lon + 0.000041) < 0.00001 and not self.auto_mode:
            self.get_logger().info('Coordinates detected: Switching to automatic mode')
            self.auto_mode = True
            self.start_gnss = (lat, lon)  

        # verifica la distancia recorrida en modo automatico
        if self.auto_mode and self.start_gnss is not None:
            current_position = (lat, lon)
            distance_traveled = geodesic(self.start_gnss, current_position).meters
            self.get_logger().info(f'Distance traveled: {distance_traveled} meters')

            if distance_traveled >= self.distance_threshold:
                self.get_logger().info(f'Distance threshold reached: Switching back to prediction mode')
                self.auto_mode = False  # vuelve al modelo predictivo

        self.get_logger().info(f'GNSS Data - Lat: {lat}, Lon: {lon}, Alt: {alt}')

    def control_vehicle(self):
        self.set_autopilot()
        self.set_control_manual_override()

        if self.auto_mode:
            self.get_logger().info('Automatic mode enabled. Autopilot is taking control.')
            return  

        if self.camera_image is not None:
            predicted_steer = self.predict_steer()

            self.control_msg.throttle = 0.18
            self.control_msg.steer = float(predicted_steer)
            self.control_msg.brake = 0.0

            self.control_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_control.publish(self.control_msg)

    def predict_steer(self):
        prediction = 0.0 
        while rclpy.ok():
            with self.lock:
                if self.camera_image is not None:
                    prediction = model.predict(np.expand_dims(self.camera_image, axis=0))[0]
                    self.get_logger().info(f"Prediction: {prediction}")

            if prediction is not None:
                predicted_steer = prediction
                self.get_logger().info(f"Steer: {predicted_steer}")
                return predicted_steer

    def set_control_manual_override(self):
        self.publisher_control_manual_override.publish(Bool(data=True))

    def set_autopilot(self):
        self.publisher_autopilot.publish(Bool(data=self.auto_mode))

def main(args=None):
    rclpy.init(args=args)
    node = DummyControl()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
