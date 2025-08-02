import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import carla
import csv
import os

class WaypointDrawer(Node):
    def __init__(self):
        super().__init__('waypoint_drawer')

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        self.subscription = self.create_subscription(
            String,
            '/curvature_status',
            self.curvature_callback,
            10
        )
        self.location_publisher = self.create_publisher(String, '/vehicle_location', 10)

        # Crear archivo CSV y escribir encabezado
        self.csv_file = 'vehicle_locations.csv'
        self.write_csv_header()

    def write_csv_header(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['x', 'y', 'z', 'status'])

    def curvature_callback(self, msg):
        curvatura = msg.data

        # Obtener el vehículo
        vehicle = self.world.get_actors().filter('vehicle.*')[0]
        location = vehicle.get_location()
        location_str = f"x: {location.x}, y: {location.y}, z: {location.z}"

        # Publicar la ubicación del vehículo
        location_msg = String()
        location_msg.data = location_str
        self.location_publisher.publish(location_msg)

        # Dibujar en el mapa de CARLA
        if curvatura == 'recta':
            self.world.debug.draw_string(
                location, 'O', draw_shadow=False, 
                color=carla.Color(r=0, g=255, b=0), 
                life_time=320, persistent_lines=True
                )
        else:
            self.world.debug.draw_string(
                location, 'O', draw_shadow=False, 
                color=carla.Color(r=255, g=0, b=0),
                  life_time=320, persistent_lines=True
                  )

        # Imprimir la ubicación del vehículo
        self.get_logger().info(f"Vehicle location: {location_str}")

        # Escribir la ubicación y el estado en el archivo CSV
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([location.x, location.y, location.z, curvatura])

def main(args=None):
    rclpy.init(args=args)
    node = WaypointDrawer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
