#!/usr/bin/python3

import rospy
import numpy as np

from eskf import ESKF
from utilities.matrix import block_3x3

from sensors.imu import ImuData
from sensors.dvl import DVL, DvlData
from sensors.depth import DepthSensor
from eskf_types.state import ErrorStateGauss, NominalState
from eskf_types.quaternion import Quaternion

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from geometry_msgs.msg import TwistWithCovarianceStamped, PoseWithCovarianceStamped, Pose, Twist, Point
from geometry_msgs.msg import Quaternion as ROSQuaternion


class ESKF_NODE:
    """Workflow:
    run imu.predict for every incoming IMU measurement
    run dvl.update for every incoming DVL measurement

    publish state estimates with a given frequency.
    Running the ESKF at a higher frequency than the imu inputs leads to repeat
    measurements
    """

    def __init__(self, frequency) -> None:
        self.dvl = DVL()
        self.depth = DepthSensor()

        acc_std = 0.001
        acc_bias_std = 0.0005
        acc_bias_rate = 10e-12

        gyro_std = 0.0002
        gyro_bias_std = 0.00002
        gyro_bias_rate = 10e-12

        acc_correction = np.eye(3)
        gyro_correction = np.eye(3)
        lever_arm = np.array((0, 0, 0))

        self.eskf = ESKF(acc_std, acc_bias_std, acc_bias_rate,
            gyro_std, gyro_bias_std, gyro_bias_rate,
            acc_correction, gyro_correction, lever_arm
        )

        pos = vel = acc_bias = gyro_bias = np.zeros(3)
        ori = Quaternion(1, np.array((0, 0, 0)))
        self.x_nom_prev = NominalState(pos, vel, ori, acc_bias, gyro_bias, rospy.Time.now().to_sec())

        self.odom_pub = rospy.Publisher("/odometry/ned", Odometry, queue_size=10)

        rospy.Subscriber("/imu/data_raw", Imu, self.imu_cb, queue_size=10)
        rospy.Subscriber("/dvl/dvl_data", TwistWithCovarianceStamped, self.dvl_cb, queue_size=10)
        rospy.Subscriber("/dvl/ahrs_pose", PoseWithCovarianceStamped, self.depth_cb, queue_size=10)

        self.rate = rospy.Rate(frequency)

        # TODO: Parametrize
        init_std = np.repeat(repeats=3, a=[0, 0, np.deg2rad(0), 0.001, 0.001]) 
        self.x_err_prev = ErrorStateGauss(np.zeros(15), np.diag(init_std**2), rospy.Time.now().to_sec())

    def imu_cb(self, msg):
        ts = msg.header.stamp.to_sec()
        av = msg.angular_velocity
        la = msg.linear_acceleration

        acc = np.array((la.x, la.y, la.z))
        avel = np.array((av.x, av.y, av.z))

        u_imu = ImuData(ts, acc, avel)
        x_nom_pred, x_err_pred = self.eskf.predict(self.x_nom_prev, self.x_err_prev, u_imu)

        self.x_nom_prev = x_nom_pred
        self.x_err_prev = x_err_pred
        self.x_nom_prev.ts = rospy.Time.now().to_sec()


    def dvl_cb(self, msg):
        ts = msg.header.stamp.to_sec()
        v = msg.twist.twist.linear

        vel = np.array((v.x, v.y, v.z))
        cov = np.reshape(msg.twist.covariance, (6, 6))
        cov = cov[block_3x3(0, 0)]
        
        z_dvl = DvlData(ts, vel, cov)

        x_err_upd = self.dvl.update(self.x_nom_prev, self.x_err_prev, z_dvl)

        x_nom_inj, x_err_inj = self.eskf.inject(self.x_nom_prev, x_err_upd)

        self.x_nom_prev = x_nom_inj
        self.x_err_prev = x_err_inj
        self.x_nom_prev.ts = rospy.Time.now().to_sec()


    def depth_cb(self, msg):
        ts = msg.header.stamp.to_sec()
        depth = -msg.pose.pose.position.z # NED

    def spin(self):

        # TODO: Add angular velocity
        # TODO: Add covariances
        seq = 0
        while not rospy.is_shutdown():

            eskf_pos = self.x_nom_prev.pos
            eskf_ori = self.x_nom_prev.ori
            eskf_vel = self.x_nom_prev.vel


            pos = Point()
            pos.x = eskf_pos[0]
            pos.y = eskf_pos[1]
            pos.z = eskf_pos[2]

            ori = ROSQuaternion()
            ori.w = eskf_ori.real_part
            ori.x = eskf_ori.vec_part[0]
            ori.y = eskf_ori.vec_part[1]
            ori.z = eskf_ori.vec_part[2]

            vel = Twist()
            vel.linear.x = eskf_vel[0]
            vel.linear.y = eskf_vel[1]
            vel.linear.z = eskf_vel[2]
            vel.angular.x = 0
            vel.angular.y = 0
            vel.angular.z = 0

            odometry = Odometry()
            odometry.pose.pose = Pose(position=pos, orientation=ori)
            #odometry.pose.covariance = ...
            odometry.twist.twist = vel
            #odometry.twist.covariance = ...

            odometry.header.seq = seq
            odometry.header.stamp = rospy.Time.now()
            odometry.header.frame_id = "ned"
            odometry.child_frame_id = "base_link"

            self.odom_pub.publish(odometry)

            seq += 1
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("eskf")
    eskf = ESKF_NODE(30)
    eskf.spin()
