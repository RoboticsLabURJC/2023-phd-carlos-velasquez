import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import carla

class RoutePlanner(Node):
    def __init__(self):
        super().__init__('route_planner')

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town02')
        self.map = self.world.get_map()

        self.index_publisher = self.create_publisher(String, '/spawn_point_indices', 10)
        self.create_spawn_point_indices()

    def create_spawn_point_indices(self):
        spawn_points = self.map.get_spawn_points()

        for i, spawn_point in enumerate(spawn_points):
            self.world.debug.draw_string(spawn_point.location, str(i), life_time=1000)
            index_msg = String()
            index_msg.data = f"Index: {i}, Location: (x: {spawn_point.location.x}, y: {spawn_point.location.y}, z: {spawn_point.location.z})"
            self.index_publisher.publish(index_msg)

            self.world.debug.draw_arrow(spawn_point.location, spawn_point.location + spawn_point.get_forward_vector(), life_time=1000)
            if i == 117: 
                print(spawn_point.location)


        # En modo síncrono, necesitamos ejecutar la simulación para mover el espectador
        while True:
            self.world.tick()

def main(args=None):
    rclpy.init(args=args)
    node = RoutePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
