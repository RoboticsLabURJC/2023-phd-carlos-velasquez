import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl
from cv_bridge import CvBridge
from collections import deque
import numpy as np
import cv2
import os
import csv
import math
import carla
import time
from threading import Lock

class CurvatureCollector(Node):
    def __init__(self):
        super().__init__('curvature_collector')

        # Configuración general
        self.output_dir = "data/ros2_curvature_dataset"
        self.rgb_dir = os.path.join(self.output_dir, "imageRGB")
        self.seg_dir = os.path.join(self.output_dir, "imageSEG")
        self.csv_path = os.path.join(self.output_dir, "labels.csv")

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.seg_dir, exist_ok=True)

        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(["image_rgb_name", "image_seg_name", "curvatura", "steer", "throttle", "brake"])

        # Inicialización de variables
        self.bridge = CvBridge()
        self.rgb_image = None
        self.seg_image = None
        self.rgb_stamp = None
        self.seg_stamp = None
        self.csv_lock = Lock()

        self.CURVATURE_THRESHOLD = 0.015
        self.positions = deque(maxlen=3)  # solo necesitamos 3 puntos para calcular la curvatura

        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0

        self.enable_visual_debug = True

        # Autopilot
        self.publisher_autopilot = self.create_publisher(Bool, "/carla/ego_vehicle/enable_autopilot", 10)
        self.set_autopilot()
        self.configure_traffic_manager()

        # Subscripciones
        self.create_subscription(Image, "/carla/ego_vehicle/rgb_front/image", self.rgb_callback, 10)
        self.create_subscription(Image, "/carla/ego_vehicle/semantic_segmentation_front/image", self.seg_callback, 10)
        self.create_subscription(CarlaEgoVehicleControl, "/carla/ego_vehicle/vehicle_control_cmd", self.control_callback, 10)
        self.create_subscription(Odometry, "/carla/ego_vehicle/odometry", self.odom_callback, 10)

    def set_autopilot(self):
        self.publisher_autopilot.publish(Bool(data=True))
        self.get_logger().info("✅ Autopilot habilitado vía ROS 2.")

    def configure_traffic_manager(self):
        try:
            client = carla.Client("localhost", 2000)
            client.set_timeout(5.0)
            world = client.get_world()
            tm = client.get_trafficmanager()  # puerto: 8000 para TrafficManager
            self.get_logger().info(f"TrafficManager puerto: {tm.get_port()}")

            ego_vehicle = None
            for _ in range(20):
                actors = world.get_actors()
                ego_vehicle = next((a for a in actors if a.type_id.startswith("vehicle.") and a.attributes.get("role_name") == "ego_vehicle"), None)
                if ego_vehicle:
                    break
                time.sleep(0.1)

            if ego_vehicle:
                self.get_logger().info("🚦 Configurando TrafficManager para ego_vehicle...")
                tm.ignore_lights_percentage(ego_vehicle, 100.0)
                tm.ignore_signs_percentage(ego_vehicle, 100.0)
                tm.auto_lane_change(ego_vehicle, False)
                tm.distance_to_leading_vehicle(ego_vehicle, 0.0)
            else:
                self.get_logger().warn("⚠️ No se encontró ego_vehicle para configurar TrafficManager.")
        except Exception as e:
            self.get_logger().error(f"❌ Error al configurar TrafficManager: {e}")

    def rgb_callback(self, msg):
        self.rgb_stamp = msg.header.stamp
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.try_save_sample()

    def seg_callback(self, msg):
        self.seg_stamp = msg.header.stamp
        self.seg_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.try_save_sample()

    def control_callback(self, msg):
        self.steer = msg.steer
        self.throttle = msg.throttle
        self.brake = msg.brake

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        self.positions.append((pos.x, pos.y))

    def estimate_curvature(self):
        if len(self.positions) < 3:
            return 0.0
        (x1, y1), (x2, y2), (x3, y3) = self.positions

        num = 2 * abs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))
        denom = math.sqrt(
            ((x2 - x1)**2 + (y2 - y1)**2) *
            ((x3 - x1)**2 + (y3 - y1)**2) *
            ((x3 - x2)**2 + (y3 - y2)**2)
        )
        return num / denom if denom != 0 else 0

    def try_save_sample(self):
        if (
            self.rgb_image is not None and
            self.seg_image is not None and
            self.rgb_stamp == self.seg_stamp
        ):
            timestamp_ms = int(self.rgb_stamp.sec * 1e3 + self.rgb_stamp.nanosec / 1e6)
            rgb_name = f"frame_{timestamp_ms}_rgb.png"
            seg_name = f"frame_{timestamp_ms}_seg.png"

            cv2.imwrite(os.path.join(self.rgb_dir, rgb_name), self.rgb_image)
            cv2.imwrite(os.path.join(self.seg_dir, seg_name), self.seg_image)

            curvature = self.estimate_curvature()
            label = 1 if curvature > self.CURVATURE_THRESHOLD else 0

            with self.csv_lock:
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([rgb_name, seg_name, label, self.steer, self.throttle, self.brake])

            if self.enable_visual_debug:
                vis = self.rgb_image.copy()
                cv2.putText(vis, f"Curvatura: {curvature:.4f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                cv2.putText(vis, f"Curva" if label else "Recta", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                cv2.putText(vis, f"Steer: {self.steer:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                cv2.imshow("Curvature ROS2", vis)
                cv2.waitKey(1)

            self.rgb_image = None
            self.seg_image = None
            self.rgb_stamp = None
            self.seg_stamp = None


def main(args=None):
    rclpy.init(args=args)
    node = CurvatureCollector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
