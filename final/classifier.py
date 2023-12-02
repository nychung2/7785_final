import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int64MultiArray, Int64
from joblib import load

class Classifier(Node):
    def __init__(self):
        super().__init__('classifier')
        self.model = load('/home/nchung/Desktop/Chung_Lab6/sign_classifier.joblib')

        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos_profile.reliability = QoSReliabilityPolicy.RELIABLE

        self.img_subscriber = self.create_subscription(
            Int64MultiArray,
            '/processed_image',
            self.img_callback,
            qos_profile
        )

        self.classifier_publisher = self.create_publisher(
            Int64,
            '/classifier',
            qos_profile
        )
    
    def img_callback(self, data):
        self.get_logger().info("Recieved Image to Classify")
        self.get_logger().info(str(int(self.model.predict(data.data))))
        msg = Int64()
        msg.data = int(self.model.predict(data.data))
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