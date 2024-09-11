import os
import time

import rclpy
from rclpy.node import Node
from carla_msgs.msg import CarlaEgoVehicleStatus
from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Configurar cuDNN
cudnn.benchmark = False
cudnn.deterministic = True


class LanePredict(Node):
    model_path = "/home/canveo/carla_ws/model/fastai_model.pth"
    MODEL = torch.load(model_path)
    MODEL.eval() 

    def __init__(self):
        super().__init__('detect_lane')

        self.bridge = CvBridge()

        cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Lane Detection", 1024, 512)

        self.speed_kmh = 0

        self.positions: list = []
        self.a, self.b, self.c = 0.0, 0.0, 0.0

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

        self.curvature_publisher = self.create_publisher(
            String,
            '/curvature_status',
            10
        )
        

    def get_prediction(self,  model, img_array):
        with torch.no_grad():
            image_tensor = img_array.transpose(2, 0, 1).astype('float32')/255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            model_output = torch.softmax(model.forward(x_tensor), dim=1).cpu().numpy()
        return model_output

    def lane_detection_overlay(self, img, left_mask, right_mask):
        res = np.copy(img)
        res[left_mask > 0.3, :] = [255, 0, 0]
        res[right_mask > 0.3, :] = [0, 0, 255]
        return res      
      
    def status_callback(self, msg):
        self.speed_kmh = msg.velocity * 3.6

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
    
    def third_derivative(self):
        pass
    
    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 512))        

        start_time = time.time()
        back, left, right = self.get_prediction(self.MODEL, img)[0]
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000 

        # overlaid_img  = img.copy()
        res = self.lane_detection_overlay(img, left, right)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        # cv2.imshow("Lane Detection", res)

        # left_line_x, right_line_x = self.draw_lane_lines(res, left, right)
        
        self.positions: list = []
        for i in [300, 310, 320, 330, 340, 350, 360, 370, 380]:
            centerx = self.lane_center(i, right, left, 0.5)
            # centerx = self.lane_center(i, right_line_x, left_line_x)             
            cv2.line(res, (0, i), (1024, i), (255, 255, 255), 1)
            cv2.circle(res, (centerx, i), 2, (0, 255, 0), -1)            
            self.positions.append((centerx, i))
        
        # reference_x: float = np.mean(np.array(self.positions))
        # x_values = [pos[0] for pos in self.positions]
        # reference_x = float(np.mean(self.positions))
        reference_x = float(np.mean([pos[0] for pos in self.positions]))
        cv2.circle(back, (int(reference_x), 330), 2, (0, 255, 255), -1) 

        
        # calculo de curvatura
        self.positions = np.array(self.positions)
        x = self.positions[:, 0]
        y = self.positions[:, 1] 

        coefficients = np.polyfit(y, x, 2)  # fit x = ay^2 + by +c
        self.a, self.b, self.c = coefficients

        # y_eval = 256

        # first_deriv = self.first_derivative(y_eval)
        # second_deriv = self.second_derivative()

        # if second_deriv != 0:
        #     radius_of_curvature = ((1 + first_deriv**2)**(3/2)) / np.abs(second_deriv)
        # else:
        #     radius_of_curvature = float('inf')  # Considerar como recto si la derivada segunda es cero

        # radius_of_curvature = ((1 + first_deriv**2)**(3/2)) / np.abs(second_deriv)

        # Calcular curvatura de cada marca de carril
        left_curvature = self.calculate_curvature(left)
        right_curvature = self.calculate_curvature(right)  

        straight_threshold = 2100

        left_curvature_value = left_curvature[0]  # Extraer el valor de curvatura
        right_curvature_value = right_curvature[0]  # Extraer el valor de curvatura

        left_status = 'recta' if left_curvature_value > straight_threshold else 'curva'
        right_status = 'recta' if right_curvature_value > straight_threshold else 'curva'

        average_curvature = (left_curvature_value + right_curvature_value) / 2
        average_status = 'recta' if average_curvature > straight_threshold else 'curva'


        
        # if average_curvature > straight_threshold:
        #     # print("El segmento es recto.")
        #     self.curvatura_text = 'recto'
        #     # r, g, b = 0, 0, 255
        # else:
        #     # print("El segmento es curvo.")
        #     self.curvatura_text = 'curvo'
            # r, g, b = 255, 0, 0
        #  Calcular promedio de curvatura
        # average_curvature = (left_curvature + right_curvature) / 2
        # if average_curvature > straight_threshold:
        #     self.curvatura_text = 'recta'
        # else:
        #     self.curvatura_text = 'curva'

        speed_text = f'Speed: {self.speed_kmh:.1f} km/h'
        inference_time_text = f'Tiempo de inferencia: {inference_time_ms:.2f} ms'
        curvature_text = f'Radio de curvatura: {average_curvature:.2f}'
        self.render_text(res, curvature_text, speed_text, inference_time_text, average_status)

        # Display the final image
        cv2.imshow("Lane Detection", res)
        cv2.waitKey(1)

        # # publisher data curvature
        curvature_message = String()
        curvature_message.data = average_status
        self.curvature_publisher.publish(curvature_message)


    def render_text(self, image, curvatura_text, speed_text, inference_time_text, curvature_text):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 255, 255)  # Red color in BGR
            thickness = 1

            # Position coordinates for each text string
            position_sp = (10, 30)
            position_tm = (10, 50)
            position_rc = (10, 70)
            position_cv = (512, 250)

            # Render the text onto the image
            cv2.putText(image, speed_text, position_sp, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(image, inference_time_text, position_tm, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(image, curvatura_text, position_rc, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(image, curvature_text, position_cv, font, font_scale, color, thickness, cv2.LINE_AA)

    # def draw_lane_lines(self, image, left_mask, right_mask):
    #     # Obtener las coordenadas de los píxeles que forman las marcas de carril
    #     left_y, left_x = np.where(left_mask > 0.5)
    #     right_y, right_x = np.where(right_mask > 0.5)

    #     # Ajustar una línea polinómica de primer grado (recta) a los puntos
    #     left_fit = np.polyfit(left_y, left_x, 1) if len(left_x) > 0 else None
    #     right_fit = np.polyfit(right_y, right_x, 1) if len(right_x) > 0 else None

    #     # Dibujar las líneas ajustadas sobre la imagen original
    #     y_max = image.shape[0]
    #     y_min = 0

    #     left_line_points = []
    #     right_line_points = []

    #     left_line_poly = []
    #     right_line_poly = []


    #     if left_fit is not None:
    #         left_line_y = np.linspace(y_min, y_max, y_max - y_min)
    #         left_line_x = np.polyval(left_fit, left_line_y).astype(int)
    #         left_line_poly = np.polyfit(left_line_y.astype(int), left_line_x, 1)
    #         left_line_points = list(zip(left_line_x, left_line_y.astype(int)))
    #         for x, y in left_line_points:
    #             cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # Línea roja

    #     if right_fit is not None:
    #         right_line_y = np.linspace(y_min, y_max, y_max - y_min)
    #         right_line_x = np.polyval(right_fit, right_line_y).astype(int)
    #         right_line_poly = np.polyfit(right_line_y.astype(int), right_line_x, 1)
    #         right_line_points = list(zip(right_line_x, right_line_y.astype(int)))
    #         for x, y in right_line_points:
    #             cv2.circle(image, (x, y), 2, (255, 0, 0), -1)  # Línea azul

    #     return left_line_poly, right_line_poly
    
    def calculate_curvature(self, mask):
        y, x = np.where(mask > 0.5)
        if len(x) < 2:
            return float('inf'), 0, 0, 0  # Sin suficiente puntos para calcular curvatura
        poly_fit = np.polyfit(y, x, 3)
        first_deriv = np.polyder(poly_fit, 1)
        second_deriv = np.polyder(poly_fit, 2)
        third_deriv = np.polyder(poly_fit, 3)
        
        y_eval = np.max(y)
        curvature = ((1 + (np.polyval(first_deriv, y_eval)) ** 2) ** 1.5) / np.abs(np.polyval(second_deriv, y_eval))
        
        return curvature, first_deriv, second_deriv, third_deriv


def main(args=None):
    rclpy.init(args=args)
    node = LanePredict()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
