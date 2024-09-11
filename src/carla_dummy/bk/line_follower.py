import pygame
import sys
import numpy as np
import rclpy
from rclpy.node import Node

from pygame.locals import K_ESCAPE
from pygame.locals import K_DOWN
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT
from pygame.locals import K_UP
from pygame.locals import KEYDOWN
from pygame.locals import KEYUP
from pygame.locals import K_a
from pygame.locals import K_s
from pygame.locals import K_w
from pygame.locals import K_d
from sensor_msgs.msg import Image
from threading import Thread
from carla_msgs.msg import CarlaEgoVehicleControl
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, Int8
from carla_msgs.msg import CarlaEgoVehicleStatus
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import cv2
from cv_bridge import CvBridge
import time
import matplotlib.pyplot as plt


class LaneDetector(Node):
    def __init__(self):
        super().__init__("Lane_detector")

        self.bridge = CvBridge()

        image_callback_group = MutuallyExclusiveCallbackGroup()
        self._default_callback_group = image_callback_group
      
        self.image_surface = None
        size = 800, 600
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("lane_detector")

        self.image_subscriber = self.create_subscription(
            Image, "/carla/ego_vehicle/rgb_front/image", 
            self.first_person_image_cb, 
            10
            )
        
        self.vehicle_status_subscriber = self.create_subscription(
            CarlaEgoVehicleStatus,
            '/carla/ego_vehicle/vehicle_status',
            self.status_callback,
            10
        )

        self.publisher = self.create_publisher(
            Int8,
            'lane_info',
            10
        )
        self.timer = self.create_timer(0.1, self.publish_info)
        self.speed_kmh = 0

        self.clock = pygame.time.Clock()
        self.fps = 0
        self.last_fps = 0
        self.start_time = 0

        self.role_name = "ego_vehicle"

        image_callback_group = MutuallyExclusiveCallbackGroup()
        self._default_callback_group = image_callback_group

        self.left_a = []
        self.left_b = []
        self.left_c = []
        self.right_a = []
        self.right_b = []
        self.right_c = []

        self.lane_info = 0

    def publish_info(self):
        value = self.get_lane_info()

        msg = Int8()
        msg.data = value
        self.publisher.publish(msg)

    def get_lane_info(self):
        return 1 if self.lane_info > 700 else 0

    def status_callback(self, msg):
        self.speed_kmh = msg.velocity * 3.6 


    def first_person_image_cb(self, image):

        if self.fps == 0:
            self.start_time = time.time()

        self.fps = self.fps + 1

        if time.time() - self.start_time >= 1:
            self.last_fps = self.fps
            self.fps = 0

        filter_img, lane_curve, curverad  = self.line_filter(image)
        array = np.frombuffer(filter_img.data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        self.lane_info = lane_curve

        pygame.init()

        if not hasattr(self, 'screen'):
            screen_width = image.width  
            screen_height = image.height  
            self.screen = pygame.display.set_mode((screen_width, screen_height))

        speed_text = f'Speed: {self.speed_kmh:.1f} km/h'
        font = pygame.font.SysFont(None, 30)
        text_surface = font.render(speed_text, True, (255, 0, 0))

        curvature_text = f'Lane Curvature: {lane_curve:.1f}'
        text_surface_lc = font.render(curvature_text, True, (255, 0, 0))

        curverad_text = f'Vehicle offset: {curverad:.1f}'
        text_surface_cv = font.render(curverad_text, True, (255, 0, 0))
     
        self.screen.fill((0, 0, 0))       

        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.screen.blit(image_surface, (0, 0))        
        self.screen.blit(text_surface, (10, 10))
        self.screen.blit(text_surface_lc, (10, 28))
        self.screen.blit(text_surface_cv, (10, 46))

        pygame.display.flip()


    # para detectar blanco utiliza valores  s_thresh=(100, 255), sx_thresh=(15, 255)
    def pipeline(self, img, s_thresh=(100, 255), sx_thresh=(15, 255)):
        img = np.copy(img)

        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(float)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        h_channel = hls[:, :, 0]

        # Sobel x detecta lor bordes en x
        # Take the derivative in x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)

        # Absolute x derivative to accentuate lines away from horizontal
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) &
                 (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        color_binary = np.dstack(
            (np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary

    def perspective_warp(self, img,
                         dst_size=(800, 600),
                         src=np.float32(
                             [(0.43, 0.65), (0.58, 0.65), (0.1, 1), (0.1, 1)]),
                         dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
        img_size = np.float32([(img.shape[1], img.shape[0])])
        src = src * img_size
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = dst * np.float32(dst_size)

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, dst_size)

        return warped

   
    def inv_perspective_warp(self, img,
                             dst_size=(800, 600),
                             src=np.float32(
                                 [(0, 0), (1, 0), (0, 1), (1, 1)]),
                             dst=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])):
        
        img_size = np.float32([(img.shape[1], img.shape[0])])
        src = src * img_size
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = dst * np.float32(dst_size)
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped

    def get_hist(sef, img):
        hist = np.sum(img[img.shape[0]//2:, :], axis=0)
        return hist

    def sliding_window(self, img, nwindows=9, margin=150, minpix=1, draw_windows=True):
        left_fit_ = np.empty(3)
        right_fit_ = np.empty(3)
        out_img = np.dstack((img, img, img))*255

        histogram = self.get_hist(img)

        # find peaks of left and right halves
        midpoint = int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = int(img.shape[0]/nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            if draw_windows == True:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                              (100, 255, 255), 3)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                              (100, 255, 255), 3)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(
                    np.mean(nonzerox[good_right_inds]))

    #        if len(good_right_inds) > minpix:
    #            rightx_current = np.int(np.mean([leftx_current +900, np.mean(nonzerox[good_right_inds])]))
    #        elif len(good_left_inds) > minpix:
    #            rightx_current = np.int(np.mean([np.mean(nonzerox[good_left_inds]) +900, rightx_current]))
    #        if len(good_left_inds) > minpix:
    #            leftx_current = np.int(np.mean([rightx_current -900, np.mean(nonzerox[good_left_inds])]))
    #        elif len(good_right_inds) > minpix:
    #            leftx_current = np.int(np.mean([np.mean(nonzerox[good_right_inds]) -900, leftx_current]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        if len(lefty) == 0 or len(leftx) == 0:
            print("Error: Los vectores lefty o leftx están vacíos.")
        else:
            left_fit = np.polyfit(lefty, leftx, 2)

        if len(righty) == 0 or len(rightx) == 0:
            print("Error: Los vectores righty o rightx están vacíos.")
        else:
            right_fit = np.polyfit(righty, rightx, 2)

        self.left_a.append(left_fit[0])
        self.left_b.append(left_fit[1])
        self.left_c.append(left_fit[2])

        self.right_a.append(right_fit[0])
        self.right_b.append(right_fit[1])
        self.right_c.append(right_fit[2])

        left_fit_[0] = np.mean(self.left_a[-10:])
        left_fit_[1] = np.mean(self.left_b[-10:])
        left_fit_[2] = np.mean(self.left_c[-10:])

        right_fit_[0] = np.mean(self.right_a[-10:])
        right_fit_[1] = np.mean(self.right_b[-10:])
        right_fit_[2] = np.mean(self.right_c[-10:])

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
        right_fitx = right_fit_[0]*ploty**2 + \
            right_fit_[1]*ploty + right_fit_[2]

        out_img[nonzeroy[left_lane_inds],
                nonzerox[left_lane_inds]] = [255, 0, 100]
        out_img[nonzeroy[right_lane_inds],
                nonzerox[right_lane_inds]] = [0, 100, 255]

        return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

    def get_curve(sef, img, leftx, rightx):
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        y_eval = np.max(ploty)
        ym_per_pix = 30.5/600  # meters per pixel in y dimension
        xm_per_pix = 3.7/800  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                         left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = (
            (1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        car_pos = img.shape[1]/2
        l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + \
            left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + \
            right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center = (car_pos - lane_center_position) * xm_per_pix / 10
        # Now our radius of curvature is in meters
        return (left_curverad, right_curverad, center)

    def draw_lanes(self, img, left_fit, right_fit):
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        color_img = np.zeros_like(img)

        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))

        cv2.fillPoly(color_img, [points.astype(int)], (0, 200, 255))
        inv_perspective = self.inv_perspective_warp(color_img)
        inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
        return inv_perspective


    def line_filter(self, ros_img):

        img = self.bridge.imgmsg_to_cv2(
            ros_img, 
            desired_encoding='passthrough'
            )

        img_ = self.pipeline(img)
        img_ = self.perspective_warp(img_)
        out_img, curves, lanes, ploty = self.sliding_window(
            img_, 
            draw_windows=False
            )

        curverad = self.get_curve(img, curves[0], curves[1])

        lane_curve = np.mean([curverad[0], curverad[1]])
        img = self.draw_lanes(img, curves[0], curves[1])

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontColor = (0, 0, 0)
        fontSize = 0.5
        cv2.putText(img, 'Lane Curvature: {:.0f} m'.format(
            lane_curve), (570, 620), font, fontSize, fontColor, 2)
        cv2.putText(img, 'Vehicle offset: {:.4f} m'.format(
            curverad[2]), (570, 650), font, fontSize, fontColor, 2)

        ros_image = self.bridge.cv2_to_imgmsg(img, encoding="passthrough")

        return ros_image, lane_curve, curverad[2]

    def show_fps(self, img):
        # fps = int(self.clock.get_fps())
        image = cv2.putText(img, 'FPS: ' + str(self.last_fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1, cv2.LINE_AA)
        self.clock.tick(60)

        return image


def main(args=None):

    pygame.init()
    rclpy.init(args=args)
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
