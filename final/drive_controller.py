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
        self.wall_distance = [0.5, 0.5, 0.53, 0.5, 0.5]
        self.target_angle = 0.0

        timer_period = 0.5 # seconds (2 Hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.Init = True
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()
        self.globalAng = 0

        bot_qos_profile = QoSProfile(depth=5)
        bot_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        bot_qos_profile.durability = QoSDurabilityPolicy.VOLATILE
        bot_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

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
        #average distance +/- 0.5 deg
        self.distance.pop(0)
        meas_dist = data.ranges[0]
        # if meas_dist == np.nan:
        #     meas_dist = 5
        self.distance.append(meas_dist)

    def classifier_callback(self, prediction):
        #wait for a callback to set a new orientation goal or to stay stationary
        # stop means search? turn 90 until another target reached? 
        # 0-> nothing, 1-> left, 2-> right, 3-> backwards (180), 4-> stop, 5-> goal
        if prediction.data == 1:
            self.target_angle = math.pi/2
            self.get_logger().info("Turning Left")
        elif prediction.data == 2:
            self.target_angle = -math.pi/2
            self.get_logger().info("Turning Right")
        elif prediction.data == 3:
            self.target_angle = math.pi
            self.get_logger().info("Turning Around")
        elif prediction.data == 4:
            self.target_angle = math.pi
            self.get_logger().info("Stop Sign - Turning Around")
        elif prediction.data == 5:
            self.get_logger().info("Reached Goal")
            self.goal_reached()
        else:
            self.move_next()

    def timer_callback(self):
        mean = np.mean(self.wall_distance)
        if mean < 0.6:
            # request wall classification
            self.request_classifier()
            time.sleep(1.0)
            self.update_angle(self.target_angle)
        elif mean >= 0.6:
            # move up until wall dist < 0.6 
            self.move_next(mean)

    def update_angle(self, angle):
        # turn turtlebot according to angle given
        msg = Twist()
        initial_orientation = self.globalAng
        target_angle = initial_orientation + angle
        e = target_angle - initial_orientation
        while abs(e) > 0.01: # roughly 0.5 degrees
            kpa = 2
            ua = kpa * e
            if ua > 1.0:
                ua = 1.0
            if ua < -1.0:
                ua = -1.0
            msg.angular.z = ua
            self.vel_publisher.publish(msg)
            time.sleep(0.05)
            e = target_angle - self.globalAng
        msg.angular.z = 0
        self.vel_publisher.publish(msg)
        return

    def move_next(self, mean):
        # move to next grid distance as given by move_dist
        msg = Twist()
        target_distance = self.target_distance
        initial_distance = mean
        e = self.target_distance - initial_distance
        while abs(e) > 0.0001:
            kpl = 50
            ul = kpl * e
            if ul > 0.1:
                ul = 0.1
            if ul < -0.1:
                ul = -0.1
            msg.linear.x = ul
            self.vel_publisher.publish(msg)
            time.sleep(0.05)
            e = target_distance - np.mean(self.wall_distance)
        msg.angular.x = 0
        self.vel_publisher.publish(msg)
        return

    def goal_reached(self):
        raise SystemExit

    def request_classifier(self):
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
