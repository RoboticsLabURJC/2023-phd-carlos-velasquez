import rclpy
from rclpy.node import Node
from carla_msgs.msg import CarlaEgoVehicleStatus
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pygame
import cv2
import numpy as np

from utils.LaneDetected import LaneDetector



class RadiusCurvature(Node):
    def __init__(self):
        super().__init__('detect_straight')

        self.bridge = CvBridge()
        self.detector = LaneDetector()      

        self.image_surface = None
        size = 1280, 960
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
        
        frame, transformed_frame, mask, msk = self.detector.run(rgb_img)  

        # self.get_logger().info(frame.shape)

        # self.get_thresholding(transformed_frame)   

        # frame = np.transpose(frame, (1, 0, 2))
        # self.get_logger().info(msk.shape)
        # transformed_frame = np.transpose(transformed_frame, (1, 0, 2))
        # mask = np.transpose(mask, (1, 0, 2))
        # msk = np.transpose(msk, (1, 0, 2))

        window_width, window_height = self.screen.get_size()

        # Calcular el tamaño de cada sección de la cuadrícula
        grid_width = window_width // 2
        grid_height = window_height // 2

        image_surface1 = pygame.Surface((grid_width, grid_height))
        image_surface2 = pygame.Surface((grid_width, grid_height))
        image_surface3 = pygame.Surface((grid_width, grid_height))
        image_surface4 = pygame.Surface((grid_width, grid_height))

        # Crear las superficies de las imágenes
        frame_surface = pygame.surfarray.make_surface(frame)
        transformed_surface = pygame.surfarray.make_surface(transformed_frame)
        mask_surface = pygame.surfarray.make_surface(mask)
        msk_surface = pygame.surfarray.make_surface(msk)  

        # Asignar cada imagen a su respectiva superficie
        image_surface1.blit(frame_surface, (0, 0))
        image_surface2.blit(transformed_surface, (0, 0))
        image_surface3.blit(mask_surface, (0, 0))
        image_surface4.blit(msk_surface, (0, 0))
 

        # Blit las superficies en la ventana principal
       # Blit las superficies de imagen en la ventana principal
        self.screen.blit(image_surface1, (0, 0))
        self.screen.blit(image_surface2, (grid_width, 0))
        self.screen.blit(image_surface3, (0, grid_height))
        self.screen.blit(image_surface4, (grid_width, grid_height))

        # Mostrar el contenido de los trackbars en la ventana de Pygame
        self.get_trackbars()
    
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