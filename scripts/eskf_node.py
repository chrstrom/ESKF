#!/usr/bin/python3

import queue
import rospy

from eskf import ESKF
from sensors.dvl import DVL
from sensors.pressure import PressureSensor

from nav_msgs.msg import Odometry
from sensor_msgs.msg import ImuSensor
from geometry_msgs.msg import TwistStamped
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
        pressure = PressureSensor()
        eskf = ESKF()


        self.odom_pub = rospy.Publisher("/odometry/ned", Odometry, queue_size=10)

        rospy.Subscriber("/imu/data_raw", ImuSensor, self.imu_cb, queue_size=10)
        rospy.Subscriber("/dvl/dvl_data", TwistStamped, self.dvl_callback, queue_size=10)
        # TODO: Pressure sensor callback

        self.rate = rospy.Rate(frequency)


    def imu_cb(self, msg):
        # Map to body
        # predict imu measurement
        # propagate error state
        pass

    def dvl_callback(self, msg):
        # Map to body
        # Predict dvl measurement
        # Correct covariances
        # Inject error state
        # Reset error state
        pass


    def spin(self):

        while not rospy.is_shutdown():
            odometry = Odometry()
            # TODO: Get estimates from self
            self.odom_pub.publish(odometry)

            self.rate.sleep()



if __name__ == '__main__':
    rospy.init_node("eskf")
    eskf = ESKF_NODE(30)
    eskf.spin()