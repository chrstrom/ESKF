#!/usr/bin/env python3

import numpy as np
import scipy.linalg as la
import time

import rospy

from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist

# For every IMU measurement:
#     propagate state
# For every DVL measurement:
#     correct state


class ESKF:

    def __init__(self, p, p_ba, p_bv, n, r):
        """
        p: initial covariances for P
        p_ba = inverse time constant for the accelerometer bias process
        b_bv = inverse time constant for the gyro bias process
        n = process noise covariance
        r = measurement noise covariance

        """

        rospy.init_node("eskf")
        
        self.N_nom_state = 16
        self.N_err_state = 15
        self.N_noise_terms = 12
        self.N_meas_terms = 3

        # Initial states and covariances
        self.nom_state = np.zeros((self.N_nom_state , 1)) #[pos, vel, orientation (quat), bias_accel, bias_ang_vel]
        self.err_state = np.zeros((self.N_err_state, 1)) #[pos, vel, orientation (eulr), bias_accel, bias_ang_vel] (errors)

        self.P = p * np.eye(self.N_nom_state)

        # Inverse time constants for the bias processes
        self.p_bias_accel = p_ba
        self.p_bias_ang_vel = p_bv

        # Noise terms
        self.process_noise_covariance = n * np.eye(self.N_noise_terms)
        self.measurement_noise_covariance = r * np.eye(self.N_meas_terms)

        rospy.Subscriber("/imu/data_raw", Imu, self.update, queue_size=100)
        rospy.Subscriber("/dvl/data_raw", Twist, self.correct, queue_size=3)


        # Wait for imu data to publish here, to avoid dt in propagate growing too high
        self.last_time = time.time

    def n(self):
        # Sample 12-dim zero mean process noise vector
        return np.random.multivariate_normal(np.zeros(self.N_noise_terms), self.process_noise_covariance)

    def w(self):
        # Sample 3-dim zero mean measurement noise vector
        return np.random.multivariate_normal(np.zeros(self.N_meas_terms), self.measurement_noise_covariance)

    def predict_measurement(self):
        z_pred = self.H @ self.nom_state + self.w()
        return z_pred

    def compose_state(self):
        x = np.zeros_like(self.nom_state)

        # "Normal addition"
        x[:3] = self.nom_state[:3] + self.err_state[:3]
        x[3:6] = self.nom_state[3:6] + self.err_state[3:6]
        x[10:13] = self.nom_state[10:13] + self.err_state[9:12]
        x[13:16] = self.nom_state[13:16] + self.err_state[12:15]

        # Quaternion multiplication
        qa = self.nom_state[6:10]
        eta_a = qa[:1].ravel()
        eps_a = qa[1:].ravel()

        qb = np.r_[np.array([1]).reshape(1, 1), 0.5*self.err_state[6:9, :]]
        QA = np.zeros((4, 4))
        QA[:3, 0] = eps_a
        QA[0, 1:4] = -eps_a.T
        QA[1:4, 1:4] = self.S(eps_a)

        x[6:10] = (eta_a * np.eye(4) + QA ) @ qb

        return x



    def update(self, imu_data):
        measured_accel = imu_data.linear_acceleration
        measured_ang_vel = imu_data.angular_velocity

        # TODO: transfer imu data to numpy arrays

        T = time.time - self.last_time

        A, G = self.discretize(measured_accel, measured_ang_vel, T)

        err_state_dot = A @ self.err_state + G @ self.n()


        self.err_state +=  T * err_state_dot # simple euler integration
        self.last_time = time.time


    def correct(self, dvl_data):
        velocity = dvl_data.linear
        # TODO: transfer dvl data to numpy arrays


        z = self.H @ velocity       

        # EKF update
        W = self.P @ self.H.T @ la.inv(self.H@self.P@self.H.T + self.process_noise_covariance)
        self.err_state = W @ (z - self.h())
        self.P = (np.eye(self.N_nom_state) - W @ self.H()) @ self.P
        self.nom_state = self.compose_state()

        # Reset covariance
        G = np.eye(self.N_err_state)
        G[6:9, 6:9] = np.eye(3) - self.S(0.5 * self.err_state[6:9])
        self.P = G @ self.P @ G.T

        # Reset error state
        self.err_state = np.zeros((self.N_err_state, 1))


    def H(self):
        # Measurement model for velocity inputs
        # Naive: only considers 2D movement

        HX = np.zeros((3, self.N_nom_state))
        HX[:3, 3:6] = np.eye(3)
        
        #q = self.nom_state[6:10].T[0]
        #eta = q[:1]
        #eps = q[1:]

        #Q_dtheta = 0.5 * np.r_[-eps.reshape((1, 3)), eta*np.eye(3) + self.S(eps)]
        #X_dx = la.block_diag(np.eye(6), Q_dtheta, np.eye(6))
        
        #H = HX @ X_dx

        return HX

    def R(self, q):
        eta = q[:1]
        eps = q[1:]
        S = self.S(eps)
        R = np.eye(3) + 2*eta*S + 2*S@S
        return R

    def S(self, v):
        S = np.array([0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0], dtype=np.float64).reshape((3, 3))

        return S

    def A_ERR_CONT(self, measured_accel, measured_ang_vel):

        q = self.nom_state[6:10]
        bias_accel = self.nom_state[10:13]
        bias_ang_vel = self.nom_state[13:16]

        A = np.zeros((self.N_err_state, self.N_err_state))

        A[:3, 3:6] = np.eye(3)
        A[3:6, 6:9] = -self.R(q) @ self.S(measured_accel - bias_accel)
        A[3:6, 9:12] = -self.R(q)
        A[6:9, 6:9] = -self.S(measured_ang_vel - bias_ang_vel)
        A[6:9, 12:15] = -np.eye(3)
        A[9:12, 9:12] = -self.p_bias_accel * np.eye(3)
        A[12:15, 12:15] = -self.p_bias_ang_vel * np.eye(3)

        return A

    def G_ERR_CONT(self):
        q = self.nom_state[6:10]

        G = np.zeros((self.N_err_state, self.N_noise_terms))

        G[3:6, :3] = -self.R(q)
        G[6:9, 3:6] = -np.eye(3)
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)

        return G

    def discretize(self, measured_accel, measured_ang_vel, T):
        AC = self.A_ERR_CONT(measured_accel, measured_ang_vel)
        GC = self.G_ERR_CONT()

        AD = la.expm(T * AC)
        GD = T * GC
        return AD, GD
        

if __name__ == "__main__":


    try:
        eskf = ESKF(p=0.001, p_ba=0.001, p_bv=0.001, n=0.001, r=0.001)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    