import cv2
import numpy as np


class LaneDetector:
    def __init__(self):
        # self.vidcap = cv2.VideoCapture(video_path)
        self.frame = None
        self.transformed_frame = None
        self.mask = None
        self.histogram = None
        self.left_base = None
        self.right_base = None

        # Create trackbars for tuning thresholds
        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("L - H", "Trackbars", 0, 255, self.nothing)
        cv2.createTrackbar("L - S", "Trackbars", 0, 255, self.nothing)
        cv2.createTrackbar("L - V", "Trackbars", 200, 255, self.nothing)
        cv2.createTrackbar("U - H", "Trackbars", 255, 255, self.nothing)
        cv2.createTrackbar("U - S", "Trackbars", 50, 255, self.nothing)
        cv2.createTrackbar("U - V", "Trackbars", 255, 255, self.nothing)


    def nothing(self, x):
        pass

    def image_thresholding(self, transformed_frame):
        # Implement logic to choose points for perspective transformation
        hsv_transformed_frame = cv2.cvtColor(
                                    transformed_frame, 
                                    cv2.COLOR_BGR2HSV
                                    )

        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])
        self.mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    def perspective_transform(self, frame):
        # Choosing points for perspective transformation
        tl = (175, 387)
        bl = (60, 472)
        tr = (450, 380)
        br = (558, 472)

        cv2.circle(frame, tl, 5, (0, 0, 255), -1)
        cv2.circle(frame, bl, 5, (0, 0, 255), -1)
        cv2.circle(frame, tr, 5, (0, 0, 255), -1)
        cv2.circle(frame, br, 5, (0, 0, 255), -1)

        # Aplying perspective transformation
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

        # Matrix to warp the image for birdseye window
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

        return transformed_frame

    def detect_lanes(self):
        # Detect lanes using image thresholding and sliding windows
        # Implement lane detection logic here
        # Sliding Window
        histogram = np.sum(self.mask[self.mask.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0]/2)
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        y = 472
        lx = []
        rx = []

        self.msk = self.mask.copy()

        while y > 0:
            # Left threshold
            img = self.mask[y-40:y, left_base-50:left_base+50]
            contours, _ = cv2.findContours(
                img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    lx.append(left_base-50 + cx)
                    left_base = left_base-50 + cx

            # Right threshold
            img = self.mask[y-40:y, right_base-50:right_base+50]
            contours, _ = cv2.findContours(
                img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    lx.append(right_base-50 + cx)
                    right_base = right_base-50 + cx

            cv2.rectangle(self.msk, (left_base-50, y),
                          (left_base+50, y-40), (255, 255, 255), 2)
            cv2.rectangle(self.msk, (right_base-50, y),
                          (right_base+50, y-40), (255, 255, 255), 2)
            y -= 40

    def run(self, image):
        # while True:
        # success, self.frame = self.vidcap.read()
        # if not success:
        #     break

        self.frame = image

        # Resize frame
        self.frame = cv2.resize(self.frame[:, :, :3], (640, 480))

        # Perspective transformation
        self.transformed_frame = self.perspective_transform(self.frame)

        # Image thresholding
        self.image_thresholding(self.transformed_frame)

        # Detect lanes
        self.detect_lanes()

        # cv2.imshow("Original", self.frame)
        # cv2.imshow("Bird's Eye View", self.transformed_frame)
        # cv2.imshow("Lane Detection - Image Thresholding", self.mask)
        # cv2.imshow("Lane Detection - Sliding Windows", self.msk)

        frame = np.transpose(self.frame, (1, 0, 2))
        transformed_frame = np.transpose(self.transformed_frame, (1, 0, 2))
        mask = np.transpose(self.mask)
        msk = np.transpose(self.msk)

        return frame, transformed_frame, mask, msk
    # , self.transformed_frame, self.mask, self.msk

        # return self.frame, self.transformed_frame, self.mask, self.msk

        # if cv2.waitKey(10) == 27:
        #     break

        # self.vidcap.release()
        # cv2.destroyAllWindows()


# if __name__ == "__main__":
#     video_path = "LaneVideo.mp4"
#     lane_detector = LaneDetector(video_path)
#     lane_detector.run()
