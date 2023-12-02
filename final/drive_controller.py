# controls movement of turtlebot. 
# subscribes to classifier, odom (orientation eg 90 degrees etc),
# and lidar
# publishes to cmd_Vel and image_processor

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int64
import time
import math
import numpy as np

class DriveController(Node):
    def __init__(self):
        super().__init__('drive_controller')

        self.target_distance = 0.5 # meters
        self.wall_distance = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.target_angle = 0.0
        self.classifier_old = 0
        self.same_grid = False

        #timer_period = 3 # seconds (2 Hz)
        #self.timer = self.create_timer(timer_period, self.timer_callback)

        self.Init = True
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()
        self.globalAng = 0

        bot_qos_profile = QoSProfile(depth=1)
        bot_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        bot_qos_profile.durability = QoSDurabilityPolicy.VOLATILE
        bot_qos_profile.reliability = QoSReliabilityPolicy.SYSTEM_DEFAULT

        local_qos_profile = QoSProfile(depth=5)
        local_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        local_qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        local_qos_profile.reliability = QoSReliabilityPolicy.RELIABLE

        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            bot_qos_profile
        )

        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            bot_qos_profile
        )

        self.classifier_subscriber = self.create_subscription(
            Int64,
            '/classifier',
            self.classifier_callback,
            local_qos_profile
        )

        self.request_publisher = self.create_publisher(
            Int64,
            '/image_request',
            local_qos_profile
        )
        
        self.vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            bot_qos_profile
        )

    # start by reading sign in front, then performing task e.g if robot sees right -> set target orientation to 90 degrees (first) and lidar distance to xm
    # then drive towards x+n m of a sign and then set that as the new target to center on? once the robot is oriented properly, request classifier. 
    # first try without the centering on the target. Might not need?

    def odom_callback(self, data):
        self.update_Odometry(data)

    def lidar_callback(self, data):
        dist = data.ranges[0]
        #self.get_logger().info(str(dist))
        msg = Twist()
        e = self.target_angle - self.globalAng
        self.get_logger().info(str(e))
        if e > math.pi:
            e -= 2*math.pi
        elif e < -math.pi:
            e += 2*math.pi
        if abs(e) > 0.02: # original: 0.01 maybe need to increase? 
            kpa = 2
            ua = kpa * e
            if ua > 1.0:
                ua = 1.0
            if ua < -1.0:
                ua = -1.0
            msg.angular.z = ua
            self.vel_publisher.publish(msg)
        elif dist < self.target_distance and not self.same_grid:
            # request wall classification
            #self.target_angle = 0
            msg.angular.x = 0.0
            self.vel_publisher.publish(msg)
            self.request_classifier()
            #self.update_angle(self.target_angle)
        elif dist >= self.target_distance:
            # move up until wall dist < 0.6 
            self.same_grid = False
            le = self.target_distance - dist
            kpl = -10
            ul = kpl * le
            if ul > 0.1:
                ul = 0.1
            if ul < -0.1:
                ul = -0.1
            msg.linear.x = ul
            self.vel_publisher.publish(msg)

    def classifier_callback(self, prediction):
        #wait for a callback to set a new orientation goal or to stay stationary
        # stop means search? turn 90 until another target reached? 
        # 0-> nothing, 1-> left, 2-> right, 3-> backwards (180), 4-> stop, 5-> goal
        #if prediction.data != self.classifier_old:
        if prediction.data == 1:
            self.get_logger().info("Turning Left")
            self.update_angle(math.pi/2)
        elif prediction.data == 2:
            self.get_logger().info("Turning Right")
            self.update_angle(-math.pi/2)
        elif prediction.data == 3:
            self.get_logger().info("Turning Around")
            self.update_angle(math.pi)
        elif prediction.data == 4:
            self.get_logger().info("Stop Sign - Turning Around")
            self.update_angle(math.pi)
        elif prediction.data == 5:
            self.get_logger().info("Reached Goal")
            self.goal_reached()
        else:
            pass
        #self.classifier_old = prediction.data

    def update_angle(self, angle):
        # turn turtlebot according to angle given
        target_angle = self.target_angle
        target_angle += angle
        if target_angle > math.pi:
            target_angle -= 2*math.pi
        elif target_angle < -math.pi:
            target_angle += 2*math.pi
        self.target_angle = target_angle
        return

    def goal_reached(self):
        raise SystemExit

    def request_classifier(self):
        self.same_grid = True
        msg = Int64()
        msg.data = 1
        self.request_publisher.publish(msg)
        return

    def update_Odometry(self, Odom):
        position = Odom.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z + q.x*q.y),1 - 2*(q.y*q.y + q.z*q.z))

        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        #We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang
        if self.globalAng < -math.pi:
            self.globalAng += 2*math.pi
        elif self.globalAng >= math.pi:
            self.globalAng -= 2*math.pi


def main():
    rclpy.init()
    drive_controller = DriveController()
    try:
        rclpy.spin(drive_controller)
    except SystemExit:
        rclpy.get_logger("Drive Controller Node").info("Shutting Down")
    drive_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
