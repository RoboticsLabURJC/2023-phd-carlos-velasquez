import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus
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
import subprocess

class CurvatureCollector(Node):
    def __init__(self):
        super().__init__('curvature_collector')

        # Configuración general
        self.output_dir = "dataTest"
        # self.rgb_dir = os.path.join(self.output_dir, "imageRGB")  # no save rgb for lightweight dataset
        self.seg_dir = os.path.join(self.output_dir, "imageSEG")
        self.csv_path = os.path.join(self.output_dir, "labels.csv")

        # os.makedirs(self.rgb_dir, exist_ok=True) # no save rgb for lightweight dataset
        os.makedirs(self.seg_dir, exist_ok=True)

        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(["image_seg_name", "curvatura", "steer", "throttle", "brake"])  # remove "image_rgb_name" to match the new format

        # Inicialización de variables
        self.bridge = CvBridge()
        self.rgb_image = None
        self.seg_image = None
        self.rgb_stamp = None
        self.seg_stamp = None
        self.csv_lock = Lock()
        
        # town01 0.015
        # town03 0.010
        # town04 0.22

        self.CURVATURE_THRESHOLD = 0.0045
        self.positions = deque(maxlen=3)  # solo necesitamos 3 puntos para calcular la curvatura

        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0

        self.enable_visual_debug = True
        self.latest_vis_image = None

        # Autopilot
        self.publisher_autopilot = self.create_publisher(Bool, "/carla/ego_vehicle/enable_autopilot", 10)
        self.set_autopilot()
        self.configure_traffic_manager()

        # Subscripciones
        self.create_subscription(Image, "/carla/ego_vehicle/rgb_front/image", self.rgb_callback, 10)
        self.create_subscription(Image, "/carla/ego_vehicle/semantic_segmentation_front/image", self.seg_callback, 10)
        self.create_subscription(CarlaEgoVehicleStatus, "/carla/ego_vehicle/vehicle_status", self.status_callback, 10)
        self.create_subscription(Odometry, "/carla/ego_vehicle/odometry", self.odom_callback, 10)

    def set_autopilot(self):
        self.publisher_autopilot.publish(Bool(data=True))
        self.get_logger().info("Autopilot habilitado vía ROS 2.")

    def configure_traffic_manager(self, max_retries=5, between_retries=0.5):
        try:
            client = carla.Client("localhost", 2000)
            client.set_timeout(20)
            world = client.get_world()
            
            tm = None
            for attemp in range(max_retries):   
                try:             
                    tm = client.get_trafficmanager(8000)  # puerto: 8000 para TrafficManager
                    if tm.get_port() == 8000:
                        self.get_logger().info(f"TrafficManager puerto: {tm.get_port()}")
                        break
                except Exception as e:
                    self.get_logger().warn(f"Intento {attemp+1}/{max_retries} fallido para conectar con TrafficManager: {e}")
                    time.sleep(between_retries)
                    
            if tm is None:
                self.get_logger().error("No se pudo conectar con TrafficManager después de varios intentos.")
                return

            ego_vehicle = None
            max_ego_retries = 30
            for attempt in range(max_ego_retries):
                actors = world.get_actors()
                ego_vehicle = next(
                    (a for a in actors if a.type_id.startswith("vehicle.") and a.attributes.get("role_name") == "ego_vehicle"), 
                    None
                )
                if ego_vehicle:
                    self.get_logger().info(f"ego_vehicle encontrado en intento {attempt+1}")
                    break
                else:
                    self.get_logger().warn(f"Intento {attempt+1}/{max_ego_retries}: esperando ego_vehicle...")
                    time.sleep(0.5)

            if ego_vehicle:
                self.get_logger().info("Configurando TrafficManager para ego_vehicle...")
                tm.ignore_lights_percentage(ego_vehicle, 100.0)
                tm.ignore_signs_percentage(ego_vehicle, 100.0)
                tm.auto_lane_change(ego_vehicle, False)
                tm.distance_to_leading_vehicle(ego_vehicle, 0.0)
            else:
                self.get_logger().warn("No se encontró ego_vehicle para configurar TrafficManager.")
        except Exception as e:
            self.get_logger().error(f"Error al configurar TrafficManager: {e}")

    def rgb_callback(self, msg):
        self.rgb_stamp = msg.header.stamp
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.try_save_sample()

    def seg_callback(self, msg):
        self.seg_stamp = msg.header.stamp
        self.seg_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.try_save_sample()

    def status_callback(self, msg):
        self.steer = msg.control.steer
        self.throttle = msg.control.throttle
        self.brake = msg.control.brake

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

            # cv2.imwrite(os.path.join(self.rgb_dir, rgb_name), self.rgb_image) # no save rgb for lightweight dataset
            cv2.imwrite(os.path.join(self.seg_dir, seg_name), self.seg_image)

            curvature = self.estimate_curvature()
            label = 1 if curvature > self.CURVATURE_THRESHOLD else 0

            with self.csv_lock:
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([seg_name, label, self.steer, self.throttle, self.brake])  #

            if self.enable_visual_debug:
                vis = self.rgb_image.copy()
                cv2.putText(vis, f"Curvatura: {curvature:.4f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                cv2.putText(vis, f"Curva" if label else "Recta", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                cv2.putText(vis, f"Steer: {self.steer:.4f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                # cv2.imshow("Curvature ROS2", vis)
                # cv2.waitKey(1)
                self.latest_vis_image = vis

            self.rgb_image = None
            self.seg_image = None
            self.rgb_stamp = None
            self.seg_stamp = None

# def kill_previous_processes():
#     targets = ["curvature_collector", "ros2"]
#     for target in targets:
#         try:
#             subprocess.run(["pkill", "-9", "-f", target], check=False)
#         except Exception as e:
#             print(f"No se pudo matar {target}: {e}")


def main(args=None):
    # kill_previous_processes()
    rclpy.init(args=args)
    node = CurvatureCollector()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.enable_visual_debug and node.latest_vis_image is not None:
                cv2.startWindowThread()
                cv2.imshow("Curvature ROS2", node.latest_vis_image)
                key = cv2.waitKey(1)
                if key == 27:  # ESC key to exit
                    break
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
