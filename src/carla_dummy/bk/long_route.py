import os
import cv2
import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleControl
from std_msgs.msg import String
import carla
import threading
import sys
import termios
import tty
import time

class VehicleController(Node):
    def __init__(self):
        super().__init__('vehicle_controller')
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town04')  # Carga el mapa Town04
        self.map = self.world.get_map()

        self.vehicle = None
        while self.vehicle is None:
            vehicles = self.world.get_actors().filter('vehicle.*')
            if vehicles:
                self.vehicle = vehicles[0]
            else:
                self.get_logger().info('Esperando a que se genere el vehículo...')
                time.sleep(1)

        self.vehicle.set_autopilot(True)  # Activa el piloto automático

        self.route = self.generate_route()
        self.current_waypoint_index = 0

        self.timer = self.create_timer(1.0, self.update_route)  # Actualiza cada segundo

        self.is_autopilot = True
        threading.Thread(target=self.keyboard_listener, daemon=True).start()

    def generate_route(self):
        start_location = carla.Location(x=230, y=195, z=40)
        end_location = carla.Location(x=230, y=250, z=40)  # Cambiar a la ruta deseada
        start_waypoint = self.map.get_waypoint(start_location)
        end_waypoint = self.map.get_waypoint(end_location)

        route = []
        current_waypoint = start_waypoint

        while current_waypoint.transform.location.distance(end_waypoint.transform.location) > 2.0:
            route.append(current_waypoint.transform.location)
            next_waypoints = current_waypoint.next(2.0)
            if next_waypoints:
                current_waypoint = next_waypoints[0]
            else:
                break

        route.append(end_waypoint.transform.location)
        return route

    def update_route(self):
        if not self.is_autopilot:
            return

        if self.current_waypoint_index >= len(self.route):
            self.current_waypoint_index = 0  # Reinicia el ciclo de waypoints

        target_location = self.route[self.current_waypoint_index]
        self.vehicle.set_transform(carla.Transform(target_location))

        distance_to_waypoint = self.vehicle.get_location().distance(target_location)
        if distance_to_waypoint < 5.0:  # Cambiar al siguiente waypoint si está cerca
            self.current_waypoint_index += 1

    def keyboard_listener(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(sys.stdin.fileno())
            print("Presiona 'm' para alternar el modo piloto automático.")
            while True:
                key = sys.stdin.read(1)
                if key.lower() == 'm':
                    self.toggle_autopilot()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def toggle_autopilot(self):
        self.is_autopilot = not self.is_autopilot
        self.vehicle.set_autopilot(self.is_autopilot)
        mode = "piloto automático" if self.is_autopilot else "manual"
        print(f"Cambiado a modo {mode}")

def main(args=None):
    rclpy.init(args=args)
    vehicle_controller = VehicleController()
    try:
        rclpy.spin(vehicle_controller)
    except KeyboardInterrupt:
        vehicle_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
