import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

class VehiclePositionVisualizer(Node):
    def __init__(self):
        super().__init__('vehicle_position_visualizer')
        
        # Suscriptor para obtener la posición del vehículo
        self.subscription = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odom_callback,
            10
        )
        
        # Publicador para visualizar la posición en RViz
        self.publisher = self.create_publisher(Marker, '/vehicle_position_marker', 10)
        
        self.marker = Marker()
        self.marker.header.frame_id = "map"
        self.marker.type = Marker.SPHERE
        self.marker.action = Marker.ADD
        self.marker.scale.x = 1.0
        self.marker.scale.y = 1.0
        self.marker.scale.z = 1.0
        self.marker.color.a = 1.0  # Transparencia
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0

    def odom_callback(self, msg):
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.marker.pose.position = msg.pose.pose.position
        self.marker.pose.orientation = msg.pose.pose.orientation
        self.publisher.publish(self.marker)

def main(args=None):
    rclpy.init(args=args)
    node = VehiclePositionVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
