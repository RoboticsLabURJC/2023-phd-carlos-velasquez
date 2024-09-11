import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from carla_msgs.msg import CarlaEgoVehicleStatus
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.image_callback,
            10)
        self.status_subscription = self.create_subscription(
            CarlaEgoVehicleStatus,
            '/carla/ego_vehicle/vehicle_status',
            self.status_callback,
            10)
        self.bridge = CvBridge()
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.speed = 0.0

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        # Draw vehicle status on the image
        info_text = f'Throttle: {self.throttle:.2f}, Brake: {self.brake:.2f}, Steer: {self.steer:.2f}, Speed: {self.speed:.2f} km/h'
        cv2.putText(cv_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("CARLA Manual Control", cv_image)
        cv2.waitKey(1)

    def status_callback(self, msg):
        self.throttle = msg.control.throttle
        self.brake = msg.control.brake
        self.steer = msg.control.steer
        self.speed = msg.velocity

def main(args=None):
    rclpy.init(args=args)

    image_subscriber = ImageSubscriber()

    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        pass

    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
