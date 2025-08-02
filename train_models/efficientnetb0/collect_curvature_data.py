import carla
from carla import Vector3D

import argparse
import cv2
import numpy as np
import time
import os
import queue
import csv
import math
import json


class CarlaDataCollector:
    def __init__(self):    
        self.image_queue = queue.Queue()

        parser = argparse.ArgumentParser(description="Script para generar Dataset con imágenes RGB, segmentadas y etiquetas de control")
        parser.add_argument('--config', type=str, required=True, help='Ruta al archivo JSON de configuración')
        args = parser.parse_args()
        config = self.load_config_from_json(args.config)

        self.output_folder_rgb = config["output"]["folder_rgb"]
        self.output_folder_seg = config["output"]["folder_seg"]
        self.output_labels_file = config["output"]["labels_file"]

        self._create_folders()
        self._initialize_csv()

        self.town_name = config['simulation']['town']
        self.spawn_point_str = config['simulation']['spawn_point']

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world(self.town_name)
        self.traffic_manager = self.client.get_trafficmanager()

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(self.settings)

        self._setup_weather()
        self._clear_vehicles()

        self.ego_vehicle = self._spawn_vehicle(self.spawn_point_str)
        self.camera_rgb = self._setup_rgb_camera()
        self.camera_seg = self._setup_seg_camera()

        # self.traffic_manager.ignore_lights_percentage(self.ego_vehicle, 100.0)
        # # self.traffic_manager.vehicle_percentage_speed_difference(self.ego_vehicle, 90)
        # # self.traffic_manager.auto_lane_change(self.ego_vehicle, False)

        # self.traffic_manager.ignore_signs_percentage(self.ego_vehicle, 100.0)
        # self.traffic_manager.random_left_lanechange_percentage(self.ego_vehicle, 0.0)
        # self.traffic_manager.random_right_lanechange_percentage(self.ego_vehicle, 100.0)

        self.rgb_image = None
        self.rgb_timestamp = None
        self.segmentation_image = None
        self.seg_timestamp = None

        self.THRESHOLD = 0.05
        self.CURVATURE_THRESHOLD = 0.015

    @staticmethod
    def load_config_from_json(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def _create_folders(self):
        os.makedirs(self.output_folder_rgb, exist_ok=True)
        os.makedirs(self.output_folder_seg, exist_ok=True)

    def _initialize_csv(self):
        with open(self.output_labels_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image_rgb_name', 'image_seg_name', 'curvarade', 'steer', 'throttle', 'brake'])

    def _setup_weather(self):
        weather = carla.WeatherParameters(
            cloudiness=30.0,
            precipitation=30.0,
            sun_altitude_angle=10.0,
            sun_azimuth_angle=90.0,
            precipitation_deposits=20.0,
            wind_intensity=0.0,
            fog_density=1.0,
            wetness=0.0,
        )
        self.world.set_weather(weather)

    def _clear_vehicles(self):
        for actor in self.world.get_actors():
            if 'vehicle' in actor.type_id:
                actor.destroy()

    def _spawn_vehicle(self, spawn_string):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        x, y, z, roll, pitch, yaw = map(float, spawn_string.split(","))
        spawn_point = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
        )
        return self.world.try_spawn_actor(vehicle_bp, spawn_point)

    def _setup_rgb_camera(self):
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
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90.0')
        camera_bp.set_attribute('sensor_tick', '0.05')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        camera.listen(lambda image: self.image_callback(image, 'segmentation'))
        return camera

    def process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        return array[:, :, :3]

    def process_segmentation_image(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        return array[:, :, :3]

    @staticmethod
    def compute_curvature(wp1, wp2, wp3):
        x1, y1 = wp1.transform.location.x, wp1.transform.location.y
        x2, y2 = wp2.transform.location.x, wp2.transform.location.y
        x3, y3 = wp3.transform.location.x, wp3.transform.location.y
        num = 2 * abs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))
        denom = math.sqrt(((x2 - x1)**2 + (y2 - y1)**2) * ((x3 - x1)**2 + (y3 - y1)**2) * ((x3 - x2)**2 + (y3 - y2)**2))
        return num / denom if denom != 0 else 0

    def image_callback(self, image, image_type):
        self.image_queue.put((image, image_type, image.timestamp))

    def run(self):
        self.ego_vehicle.set_autopilot(True)
        done = False

        while not done:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            while not self.image_queue.empty():
                image, image_type, timestamp = self.image_queue.get()
                if image_type == 'rgb':
                    self.rgb_image = self.process_image(image)
                    self.rgb_timestamp = timestamp
                elif image_type == 'segmentation':
                    self.segmentation_image = self.process_segmentation_image(image)
                    self.seg_timestamp = timestamp

                if (self.rgb_image is not None and self.segmentation_image is not None and
                        abs(self.rgb_timestamp - self.seg_timestamp) < self.THRESHOLD):
                    location = self.ego_vehicle.get_location()
                    carla_map = self.world.get_map()
                    wp1 = carla_map.get_waypoint(location)
                    wp2 = wp1.next(2.0)[0]
                    wp3 = wp2.next(2.0)[0]
                    curvature = self.compute_curvature(wp1, wp2, wp3)
                    curvature_label = 1 if curvature > self.CURVATURE_THRESHOLD else 0

                    timestamp_ms = int(timestamp * 1000)
                    rgb_name = f"frame_{timestamp_ms}_rgb.png"
                    seg_name = f"frame_{timestamp_ms}_seg.png"
                    rgb_path = os.path.join(self.output_folder_rgb, rgb_name)
                    seg_path = os.path.join(self.output_folder_seg, seg_name)

                    control = self.ego_vehicle.get_control()
                    steer = control.steer
                    throttle = control.throttle
                    brake = control.brake

                    velocity = self.ego_vehicle.get_velocity()
                    speed = np.linalg.norm([velocity.x, velocity.y, velocity.z]) * 3.6

                    cv2.imwrite(rgb_path, self.rgb_image)
                    cv2.imwrite(seg_path, self.segmentation_image)

                    with open(self.output_labels_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([rgb_name, seg_name, curvature_label, steer, throttle, brake])

                    vis = self.rgb_image.copy()
                    overlay = vis.copy()
                    cv2.rectangle(overlay, (35, 30), (200, 160), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
                    cv2.putText(vis, f"Calzada: {curvature}", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    cv2.putText(vis, f"Calzada: {'curva' if curvature_label == 1 else 'recta'}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    cv2.putText(vis, f"Speed: {speed:.2f} km/h", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    cv2.putText(vis, f"Steer: {steer:.2f}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    cv2.putText(vis, f"Throttle: {throttle:.2f}", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    cv2.putText(vis, f"Brake: {brake:.2f}", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    cv2.imshow("Curvature Classification", vis)
                    self.rgb_image = None
                    self.segmentation_image = None

            self.world.tick()

        self.camera_rgb.stop()
        self.camera_rgb.destroy()
        self.camera_seg.stop()
        self.camera_seg.destroy()
        self.ego_vehicle.destroy()
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)
        cv2.destroyAllWindows()


def main():  
    collector = CarlaDataCollector()
    collector.run()

if __name__ == "__main__":
    main()
