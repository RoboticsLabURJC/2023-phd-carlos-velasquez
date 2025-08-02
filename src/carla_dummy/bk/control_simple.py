import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from carla_msgs.msg import CarlaEgoVehicleControl
import numpy as np
import cv2
from cv_bridge import CvBridge
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import asyncio

# Register the custom loss function
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Load the trained model
model_path = "/home/canveo/Projects/notebook/model/nuevo_dp05_monolitico_3dof_epoch_263.h5"
model = tf.keras.models.load_model(model_path, custom_objects={'mse': mse})

# Compile the model to avoid warnings
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

class CarlaInferenceNode(Node):
    def __init__(self):
        super().__init__('carla_inference_node')
        self.subscription = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)
        self.bridge = CvBridge()
        self.control = CarlaEgoVehicleControl()
        self.loop = asyncio.get_event_loop()

    async def process_image_async(self, frame):
        model_input = self.preprocess_for_model(frame)
        model_input = np.expand_dims(model_input, axis=0)  # Add batch dimension
        
        # Asynchronous inference using run_in_executor
        predictions = await self.loop.run_in_executor(None, model.predict, model_input)
        steer_prediction = predictions[0][0]
        throttle_prediction = predictions[1][0]
        brake_prediction = predictions[2][0]
        
        self.control.steer = float(steer_prediction)
        self.control.throttle = float(throttle_prediction)
        self.control.brake = float(brake_prediction)
        
        self.publisher.publish(self.control)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = self.process_image(frame)
        
        # Run the async process_image_async function
        asyncio.run_coroutine_threadsafe(self.process_image_async(frame), self.loop)

    def process_image(self, image):
        array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        return array

    def preprocess_for_model(self, image):
        img_cropped_resized = cv2.resize(image[200:-1, :], (200, 66))
        normalized_image = img_cropped_resized / 255.0  
        return normalized_image

def main(args=None):
    rclpy.init(args=args)
    node = CarlaInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
