import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

import torch
import re
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
from torch import LongTensor
import albumentations as albu



class CarlaLanesDataset(Dataset):
    CLASSES = ['background', 'left_marker', 'right_marker']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        get_label_name = lambda fn: re.sub(".png", "_label.png", fn)
        self.masks_fps = [os.path.join(masks_dir, get_label_name(image_id)) for image_id in self.ids]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, LongTensor(mask)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def get_validation_augmentation():
        return None

    @staticmethod
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    @staticmethod
    def get_preprocessing(preprocessing_fn):
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=CarlaLanesDataset.to_tensor),
        ]
        return albu.Compose(_transform)


class CurvaturePredictor(Node):
    ENCODER = 'efficientnet-b0'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d'
    DEVICE = 'cuda'

    def __init__(self):
        super().__init__('curvature_predictor')

        self.bridge = CvBridge()

        self.positions: list = []
        self.a, self.b, self.c = 0.0, 0.0, 0.0

        self.subscription_img = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.image_callback,
            10
        )

        self.curvature_publisher = self.create_publisher(
            String,
            '/curvature_status',
            10
        )

        model_path = '/home/canveo/Documents/carla_laneddetection/lane_detection.pth'

        self.model = smp.FPN(
            encoder_name=self.ENCODER,
            encoder_weights=None,
            classes=len(CarlaLanesDataset.CLASSES),
            activation=self.ACTIVATION,
        ).to(self.DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.DEVICE)))
        self.model.eval()

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.ENCODER, self.ENCODER_WEIGHTS)

    def lane_center(self, centery: int, right_mask: np.ndarray, left_mask: np.ndarray, threshold: float) -> int:
        i: int = 1023

        while i > 0:
            if right_mask[centery, i] >= threshold:
                break
            i -= 1

        z: int = 1023
        while z > 0:
            if left_mask[centery, z] >= threshold:
                break
            z -= 1

        centerx: int = (i + z) // 2

        return centerx

    def first_derivative(self, y):
        return 2*self.a*y + self.b
    
    def second_derivative(self):
        return 2*self.a
    
    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 512))

        preprocessed_image = CarlaLanesDataset.get_preprocessing(self.preprocessing_fn)(image=img)["image"]
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        preprocessed_image = torch.tensor(preprocessed_image).to(self.DEVICE)

        start_time = time.time()

        with torch.no_grad():
            output = self.model(preprocessed_image)

        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000

        predicted_mask = np.argmax(output.squeeze().cpu().numpy(), axis=0)

        left_mask = (predicted_mask == CarlaLanesDataset.CLASSES.index('left_marker')).astype(np.uint8) * 255
        right_mask = (predicted_mask == CarlaLanesDataset.CLASSES.index('right_marker')).astype(np.uint8) * 255

        self.positions = []
        for i in [290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390]:
            centerx = self.lane_center(i, right_mask, left_mask, 0.5)
            self.positions.append((centerx, i))

        self.positions = np.array(self.positions)
        x = self.positions[:, 0]
        y = self.positions[:, 1] 

        coefficients = np.polyfit(y, x, 2)
        self.a, self.b, self.c = coefficients

        y_eval = 256

        first_deriv = self.first_derivative(y_eval)
        second_deriv = self.second_derivative()

        radius_of_curvature = ((1 + first_deriv**2)**(3/2)) / np.abs(second_deriv)
        straight_threshold = 1000  
        if radius_of_curvature > straight_threshold:
            curvature_text = 'recto'
        else:
            curvature_text = 'curvo'

        curvature_message = String()
        curvature_message.data = curvature_text
        self.curvature_publisher.publish(curvature_message)

        self.get_logger().info(f'Radio de curvatura: {radius_of_curvature:.2f}, {curvature_text}')

def main(args=None):
    rclpy.init(args=args)
    node = CurvaturePredictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
