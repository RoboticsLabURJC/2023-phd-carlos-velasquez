import rclpy
from rclpy.node import Node
from carla_msgs.msg import CarlaEgoVehicleStatus
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pygame
import cv2
import numpy as np

from utils.LaneDetector import LaneDetector



class RadiusCurvature(Node):
    def __init__(self):
        super().__init__('detect_straight')

        self.bridge = CvBridge()
        self.detector = LaneDetector()      

        self.image_surface = None
        size = 640, 480
        self.screen = pygame.display.set_mode(size=size)
        pygame.display.set_caption('straight detection')

        self.speed_kmh = 0

        self.vehicle_status_subscriber = self.create_subscription(
            CarlaEgoVehicleStatus,
            '/carla/ego_vehicle/vehicle_status',
            self.status_callback,
            10
        )

        self.speed_value = 0.0

        self.subscription_img = self.create_subscription(
                Image,
                '/carla/ego_vehicle/rgb_front/image',
                self.image_callback,
                10
            )
        

    def status_callback(self, msg):
        self.speed_kmh = msg.velocity * 3.6
        # self.get_logger().info('Speed:   % 15.0f km/h' % (self.speed_kmh))

    def nothing(self, x):
        pass

    def image_callback(self, msg):

              
        camera_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_img = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)   
        
        frame = self.detector.detect_lane(rgb_img)  

        frame = np.transpose(frame, (1, 0, 2))
        frame_surface = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame_surface, (0, 0))
   
        speed_text = f'Speed: {self.speed_kmh:.1f} km/h'
        font = pygame.font.SysFont(None, 30)
        text_surface = font.render(speed_text, True, (255, 0, 0))
        self.screen.blit(text_surface, (10, 10))
        pygame.display.flip()


        # self.detector.image_thresholding(transformed_frame, l_h, l_s, l_v, u_h, u_s, u_v)
    
def main(args=None):
    pygame.init()
    rclpy.init(args=args)
    node = RadiusCurvature()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()