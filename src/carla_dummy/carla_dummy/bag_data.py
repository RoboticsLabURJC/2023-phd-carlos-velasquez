import os
import cv2
import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
# from ros2pkg.api import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from carla_msgs.msg import CarlaEgoVehicleStatus

class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder')

        self.subscription_img = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.image_callback,
            10
        )

        self.subscription_control = self.create_subscription(
            CarlaEgoVehicleStatus,
            '/carla/ego_vehicle/vehicle_status',
            self.control_callback,
            10  
        )

        self.image_timestamps = []
        self.image_names = []
        self.control_timestamps = []
        self.steer_values = []
        self.throttle_values = []
        self.brake_values = []
        self.speed_values = []

        self.bridge = CvBridge()

        self.path_output = '/home/canveo/carla_ros2_ws/carla_data/'

        
        self.image_path = os.path.join(self.path_output, 'images')
        self.image_path = os.path.join(self.path_output, 'images')
        os.makedirs(self.image_path, exist_ok=True) 
        print("datarecorder")

        self.output_image_file = os.path.join(self.path_output, 'image_data.npy')
        self.output_control_file = os.path.join(self.path_output, 'control_data.npy')

    
        self.output_data_file = os.path.join(self.path_output, 'carla_data.csv')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            print(e)
            return

     
        timestamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        image_name = f"frame{timestamp}.png"
        image_path = os.path.join(self.image_path, image_name)
        cv2.imwrite(image_path, cv_image)

        self.image_timestamps.append(timestamp)
        self.image_names.append(image_name)

    def control_callback(self, msg):
        # asociar las con la ultima imagen registrada
         control = msg.control
         self.control_timestamps.append(msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec)
         self.steer_values.append(control.steer)
         self.throttle_values.append(control.throttle)
         self.brake_values.append(control.brake)
         self.speed_values.append(msg.velocity)

    def save_data(self):
        np.save(self.output_image_file, {'timestamps': self.image_timestamps, 'names': self.image_names})
        np.save(self.output_control_file, {'timestamps': self.control_timestamps, 'steer': self.steer_values,
                                           'throttle': self.throttle_values, 'brake': self.brake_values,
                                           'speed': self.speed_values})

        image_data = np.load(self.output_image_file, allow_pickle=True).item()
        control_data = np.load(self.output_control_file, allow_pickle=True).item()

        image_df = pd.DataFrame({'timestamp': image_data['timestamps'], 'image_name': image_data['names']})
        control_df = pd.DataFrame({'timestamp': control_data['timestamps'], 'steer': control_data['steer'],
                                   'throttle': control_data['throttle'], 'brake': control_data['brake'],
                                   'speed': control_data['speed']})
        
        # image_df['timestamp'] = image_df['timestamp'].astype('float64')


        # Fusionar los DataFrames segun los timestamps
        merged_df = pd.merge_asof(image_df, control_df, on='timestamp', direction='forward')

        merged_df.to_csv(self.output_data_file, index=False)

        print("Datos guardados en archivos .npy y CSV")

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
