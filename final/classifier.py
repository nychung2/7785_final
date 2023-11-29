import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int64MultiArray, Int64
import time
import numpy as np
from joblib import load
from sklearn import metrics

class Classifier(Node):
    def __init__(self):
        super().__init__('classifier')
        self.model = load('/home/nchung/somewhere/sign_classifier.joblib')

        qos_profile = QoSProfile(depth=5)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos_profile.reliability = QoSReliabilityPolicy.RELIABLE

        self.img_subscriber = self.create_subscription(
            Int64MultiArray,
            '/proccessed_image',
            self.img_callback,
            qos_profile
        )

        self.classifier_publisher = self.create_publisher(
            Int64,
            '/classifier',
            qos_profile
        )
    
    def img_callback(self, data):
        msg = Int64()
        msg.data = self.model.predict(data)
        self.classifier_publisher.publish(msg)

def main():
    rclpy.init()
    classifier = Classifier()
    try:
        rclpy.spin(classifier)
    except SystemExit:
        rclpy.get_logger("Classifier Node").info("Shutting Down")
    classifier.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()