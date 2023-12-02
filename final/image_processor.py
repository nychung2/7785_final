# recieves image from raspberry pi cam (subscribe)
# publishes processed image/data to classifer to identify
# Also recieves control query from drive_controller to initiate
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int64, Int64MultiArray
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

import time
import cv2
import numpy as np 

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')

        self.image = None

        img_qos_profile = QoSProfile(depth=5)
        img_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        img_qos_profile.durability = QoSDurabilityPolicy.VOLATILE
        img_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        local_qos_profile = QoSProfile(depth=5)
        local_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        local_qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        local_qos_profile.reliability = QoSReliabilityPolicy.RELIABLE

        self.img_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed', 
            #'/simulated_camera/image_raw/compressed',
            self.img_callback,
            img_qos_profile
        )

        self.request_subscriber = self.create_subscription(
            Int64,
            '/image_request', 
            self.request_callback,
            local_qos_profile
        )

        self.pro_img_publisher = self.create_publisher(
            Int64MultiArray,
            '/processed_image',
            local_qos_profile
        )

    def img_callback(self, image):
        self.image = image

    def request_callback(self, msg):
        if msg.data == 1:
            img = CvBridge().compressed_imgmsg_to_cv2(self.image)
            to_send = Int64MultiArray()
            to_send.data = self.process_image(img)
            self.pro_img_publisher.publish(to_send)

    def get_object_location(self, contours):
        # def contour_area(a):
        #     return cv2.contourArea(a)
        # areas = np.apply_along_axis(contour_area, axis = 0)
        # filt_max = np.argmax(areas[areas <= 100])
        # c = contours[filt_max]
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        return x, y, w ,h
        
    
    def crop_image(self, img, x, y, w, h, margin=15):
        if x - margin >= 0:
            x1 = x - margin
        else:
            x1 = x
        if x + w + margin <= img.shape[1]:
            x2 = x + w + margin
        else:
            x2 = x + w
        if y - margin >= 0:
            y1 = y - margin
        else:
            y1 = y
        if y + h + margin >= img.shape[0]:
            y2 = y + h + margin
        else:
            y2 = y + h

        crop_img = img[y1:y2, x1:x2]

        return crop_img

    def process_images(self, img):
    # initalize kernel and threshold values
        kernel = np.ones((5,5), np.uint16)
        lower_range = np.array([0, 95, 45])
        upper_range = np.array([180, 255, 255])
        width, height = 80, 60
            # convert2hsv, close and open (maybe need to convert color/process a lil more)
        
        original_img = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.inRange(img, lower_range, upper_range)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        # find contour
        adjust = 0
        contours, hierarchy = cv2.findContours(img[adjust:-1,:], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            x, y, w, h = self.get_object_location(contours)
            y = y + adjust
            cropped_img = cv2.resize(self.crop_image(original_img, x, y, w, h), (width,height))
            cropped_mask = cv2.resize(self.crop_image(img, x, y, w, h), (width,height))
            return np.append(cropped_img.flatten(), cropped_mask.flatten())
        else:
            shaped = cv2.resize(original_img, (width,height)).flatten()
            return np.append(shaped, np.zeros((width*height), dtype=int))

def main():
    rclpy.init()
    image_processor = ImageProcessor()
    try:
        rclpy.spin(image_processor)
    except SystemExit:
        rclpy.get_logger("Image Processor Node").info("Shutting Down")
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
