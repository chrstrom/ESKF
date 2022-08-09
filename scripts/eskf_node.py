#!/usr/bin/python3

import rospy
import numpy as np

from eskf import ESKF

from sensors.imu import ImuData
from sensors.dvl import DVL
from sensors.depth import DepthSensor
from eskf_types.state import NominalState
from eskf_types.quaternion import Quaternion

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped, PoseWithCovarianceStamped


class ESKF_NODE:
    """Workflow:
    run imu.predict for every incoming IMU measurement
    run dvl.update for every incoming DVL measurement

    publish state estimates with a given frequency.
    Running the ESKF at a higher frequency than the imu inputs leads to repeat
    measurements

    The IMU is not part of the sensors/ since it is the driving unit of everything in the ESKF
    """

    def __init__(self, frequency) -> None:
        dvl = DVL()
        depth = DepthSensor()
        acc_std = acc_bias_std = acc_bias_rate = gyro_std = gyro_bias_std =  gyro_bias_rate = 0
        acc_correction = np.eye(3)
        gyro_correction = np.eye(3)
        lever_arm = np.array((0, 0, 0))

        self.eskf = ESKF(acc_std, acc_bias_std, acc_bias_rate,
            gyro_std, gyro_bias_std, gyro_bias_rate,
            acc_correction, gyro_correction, lever_arm
        )

        pos = vel = acc_bias = gyro_bias = np.array((0, 0, 0))
        ori = Quaternion(0, np.array((0, 0, 1)))
        self.x_nom_prev = NominalState(pos, vel, ori, acc_bias, gyro_bias, rospy.Time.now().to_sec())

        self.odom_pub = rospy.Publisher("/odometry/ned", Odometry, queue_size=10)

        rospy.Subscriber("/imu/data_raw", Imu, self.imu_cb, queue_size=10)
        rospy.Subscriber("/dvl/dvl_data", TwistWithCovarianceStamped, self.dvl_cb, queue_size=10)
        rospy.Subscriber("/dvl/ahrs_pose", PoseWithCovarianceStamped, self.depth_cb, queue_size=10)

        self.rate = rospy.Rate(frequency)

    def imu_cb(self, msg):
        ts = msg.header.stamp.to_sec()
        av = msg.angular_velocity
        la = msg.linear_acceleration
        avel = np.array((av.x, av.y, av.z))
        acc = np.array((la.x, la.y, la.z))

        u_imu = ImuData(ts, acc, avel)
        imu_body = self.eskf.imu_to_body(self.x_nom_prev, u_imu)
        x_nom_pred = self.eskf.predict_x_nom(self.x_nom_prev, imu_body)
        
        # predict imu measurement
        # propagate error state

    def dvl_cb(self, msg):
        ts = msg.header.stamp.to_sec()
        vel = msg.twist.twist.linear
        # Map to body
        # Predict dvl measurement
        # Correct covariances
        # Inject error state
        # Reset error state
        pass

    def depth_cb(self, msg):
        ts = msg.header.stamp.to_sec()
        depth = -msg.pose.pose.position.z # NED

    def spin(self):

        while not rospy.is_shutdown():
            odometry = Odometry()
            # TODO: Get estimates from self
            self.odom_pub.publish(odometry)

            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("eskf")
    eskf = ESKF_NODE(30)
    eskf.spin()
