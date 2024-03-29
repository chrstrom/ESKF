import numpy as np
from numpy import ndarray
import threading
import scipy
from dataclasses import dataclass, field
from typing import Tuple

from datetime import datetime as dt

from sensors.imu import ImuData

from eskf_types.state import NominalState, ErrorStateGauss
from eskf_types.quaternion import Quaternion
from utilities.matrix import skew, block_3x3

@dataclass
class ESKF:
    acc_std: float
    acc_bias_std: float
    acc_bias_rate: float  # rate = p in Brekke

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_rate: float  # rate = p in Brekke

    acc_correction: "ndarray[3,3]"
    gyro_correction: "ndarray[3,3]"
    lever_arm: "ndarray[3]"

    Q_err: "ndarray[12,12]" = field(init=False, repr=False)
    g: "ndarray[3]" = np.array([0, 0, 9.82])

    def __post_init__(self):
        # Brekke 10.70
        V = self.acc_std**2 * self.acc_correction @ self.acc_correction.T
        T = self.gyro_std**2 * self.gyro_correction @ self.gyro_correction.T
        A = self.acc_bias_std**2 * np.eye(3)
        O = self.gyro_bias_std**2 * np.eye(3)

        self.Q_err = scipy.linalg.block_diag(V, T, A, O)

        # Initialize the variables holding the posteriors
        pos = vel = acc_bias = gyro_bias = np.zeros(3)
        ori = Quaternion(1, np.array((0, 0, 0)))
        self.x_nom_prev = NominalState(pos, vel, ori, acc_bias, gyro_bias, 0)
        init_std = np.repeat(repeats=3, a=[0, 0, np.deg2rad(0), 0.0001, 0.0001]) 
        self.x_err_prev = ErrorStateGauss(np.zeros(15), np.diag(init_std**2), 0)

        self.gauss_mutex = threading.Lock()
        self.start_time = dt.now()

    def A_err_cont(
        self,
        x_nom_prev: NominalState,
        u_imu_body: ImuData,
    ) -> "ndarray[15,15]":
        """Get the transition matrix, A, in Brekke (10.68)

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (ImuMeasurement): corrected IMU measurement
        Returns:
            A (ndarray[15,15]): A
        """
        A = np.zeros((15, 15))
        A[block_3x3(0, 1)] = np.eye(3)
        A[block_3x3(1, 2)] = -x_nom_prev.ori.R @ skew(
            u_imu_body.acc - x_nom_prev.acc_bias
        )
        A[block_3x3(1, 3)] = -x_nom_prev.ori.R
        A[block_3x3(2, 2)] = -skew(u_imu_body.avel - x_nom_prev.acc_bias)
        A[block_3x3(2, 4)] = -np.eye(3)
        A[block_3x3(3, 3)] = -self.acc_bias_rate * self.acc_correction
        A[block_3x3(4, 4)] = -self.gyro_bias_rate * self.gyro_correction

        return A

    def GQGT_err_cont(self, x_nom_prev: NominalState) -> "ndarray[15, 12]":
        """The noise covariance matrix, GQGT, in (10.68)

        From (Theorem 3.2.2) we can see that (10.68) can be written as
        d/dt x_err = A@x_err + G@n == A@x_err + m
        where m is gaussian with mean 0 and covariance G @ Q @ G.T. Thats why
        we need GQGT.

        Hint: you can use block_3x3 to simplify indexing if you want to.
        The first I element in G can be set as G[block_3x3(2, 1)] = -np.eye(3)

        Args:
            x_nom_prev (NominalState): previous nominal state
        Returns:
            GQGT (ndarray[15, 15]): G @ Q @ G.T
        """
        G = np.zeros((15, 12))

        G[block_3x3(1, 0)] = -x_nom_prev.ori.R
        G[block_3x3(2, 1)] = -np.eye(3)
        G[block_3x3(3, 2)] = np.eye(3)
        G[block_3x3(4, 3)] = np.eye(3)

        GQGT = G @ self.Q_err @ G.T

        return GQGT

    def get_van_loan_matrix(self, V: "ndarray[30, 30]"):
        """Use this funciton in get_discrete_error_diff to get the van loan
        matrix. See (4.63)

        All the tests are ran with do_approximations=False

        Args:
            V (ndarray[30, 30]): [description]

        Returns:
            VLM (ndarray[30, 30]): VanLoanMatrix
        """
        # second order approcimation of matrix exponential which is faster
        # VLM = np.eye(*V.shape) + V + (V@V) / 2

        VLM = scipy.linalg.expm(V)
        return VLM

    def discretize(
        self,
        x_nom_prev: NominalState,
        u_imu_body: ImuData,
    ) -> Tuple["ndarray[15, 15]", "ndarray[15, 15]"]:
        """Get the discrete equivalents of A and GQGT in (4.63)

        Hint: you should use get_van_loan_matrix to get the van loan matrix

        See (4.5 Discretization) and (4.63) for more information.
        Or see "Discretization of process noise" in
        https://en.wikipedia.org/wiki/Discretization

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (ImuMeasurement): corrected IMU measurement

        Returns:
            Ad (ndarray[15, 15]): discrede transition matrix
            GQGTd (ndarray[15, 15]): discrete noise covariance matrix
        """
        Ts = 0.004 #abs(x_nom_prev.ts - u_imu_body.ts) # TODO: How to handle difference in ROSTIME and now?

        A = self.A_err_cont(x_nom_prev, u_imu_body)

        GQGT = self.GQGT_err_cont(x_nom_prev)

        V = np.block([[-A, GQGT], [np.zeros((15, 15)), A.T]])

        VL = self.get_van_loan_matrix(V * Ts)

        V2 = VL[:15, 15:]
        V1 = VL[15:, 15:]

        Ad = V1.T
        GQGTd = V1.T @ V2

        return Ad, GQGTd

    def imu_to_body(
        self,
        x_nom_prev: NominalState,
        u_imu: ImuData,
    ) -> ImuData:
        """Correct IMU measurement so it gives a measurement of acceleration
        and angular velocity in body.

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_imu (ImuMeasurement): raw IMU measurement

        Returns:
            ImuMeasurement: corrected IMU measurement
        """

        acc_imu = u_imu.acc #- x_nom_prev.acc_bias
        avel_imu = u_imu.avel #- x_nom_prev.gyro_bias

        acc_body = self.acc_correction @ acc_imu
        avel_body = self.gyro_correction @ avel_imu

        imu_body = ImuData(u_imu.ts, acc_body, avel_body)

        return imu_body

    def predict_x_nom(
        self,
        x_nom_prev: NominalState,
        u_imu: ImuData,
    ) -> NominalState:
        """Predict the nominal state, given a corrected IMU measurement

        Hint: Discrete time prediction of equation (10.58)
        See the assignment description for more hints

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (ImuMeasurement): corrected IMU measuremnt

        Returns:
            x_nom_pred (NominalState): predicted nominal state
        """

        if x_nom_prev.ts is None:
            x_nom_prev.ts = 0

        # Catch NaN's
        if x_nom_prev.ori.real_part is np.nan or any(x_nom_prev.ori.vec_part) is np.nan:
            x_nom_prev.ori = Quaternion(1, np.zeros((3, 1)))

        h = 0.004 #float(abs(x_nom_prev.ts - u_imu.ts)) # TODO: Fix time stepping

        # Previous state
        pos_prev = x_nom_prev.pos
        vel_prev = x_nom_prev.vel
        ori_prev = x_nom_prev.ori
        acc_bias_prev = x_nom_prev.acc_bias
        gyro_bias_prev = x_nom_prev.gyro_bias

        # Measurements
        z_acc = u_imu.acc
        z_avel = u_imu.avel

        # State derivatives from Brekke (10.58)
        pos_dot = vel_prev
        vel_dot = x_nom_prev.ori.R @ (z_acc - acc_bias_prev) + self.g
        acc_bias_dot = -self.acc_bias_rate * np.eye(3) @ acc_bias_prev
        gyro_bias_dot = -self.gyro_bias_std * np.eye(3) @ gyro_bias_prev

        # Euler step to get predictions from
        pos = pos_prev + h * pos_dot
        vel = vel_prev + h * vel_dot
        acc_bias = acc_bias_prev + h * acc_bias_dot
        gyro_bias = gyro_bias_prev + h * gyro_bias_dot

        # Orientation
        omega = np.array(z_avel - gyro_bias_prev)
        nu = ori_prev.real_part
        eta = np.array(ori_prev.vec_part)
        q_dot_real = -0.5 * omega @ eta.T
        q_dot_vec = (nu * np.eye(3) + skew(eta)) @ omega.T

        q_pred_real = nu + h * q_dot_real
        q_pred_vec = eta + h * q_dot_vec

        norm = np.sqrt(q_pred_real**2 + sum(e * e for e in q_pred_vec))

        q = Quaternion(q_pred_real / norm, q_pred_vec / norm)

        x_nom_pred = NominalState(pos, vel, q, acc_bias, gyro_bias)

        return x_nom_pred

    def predict_x_err(
        self,
        x_nom_prev: NominalState,
        x_err_prev_gauss: ErrorStateGauss,
        u_imu: ImuData,
    ) -> ErrorStateGauss:
        """Predict the error state by doing a discrete step of Brekke (10.68)

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_prev_gauss (ErrorStateGauss): previous error state gaussian
            z_corr (ImuMeasurement): corrected IMU measuremnt

        Returns:
            x_err_pred (ErrorStateGauss): predicted error state
        """
        Ad, GQGTd = self.discretize(x_nom_prev, u_imu)

        P_prev = x_err_prev_gauss.cov
        Q = Ad @ P_prev @ Ad.T + GQGTd

        x_err_pred = ErrorStateGauss(x_err_prev_gauss.mean, Q, x_nom_prev.ts)

        return x_err_pred

    def predict(
        self,
        u_imu: ImuData,
    ) -> Tuple[NominalState, ErrorStateGauss]:
        """Run a prediction step for every IMU input

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_gauss (ErrorStateGauss): previous error state gaussian
            z_imu (ImuMeasurement): raw IMU measurement

        Returns:
            x_nom_pred (NominalState): predicted nominal state
            x_err_pred (ErrorStateGauss): predicted error state
        """

        u_imu_body = self.imu_to_body(self.x_nom_prev, u_imu)

        self.gauss_mutex.acquire()
        self.x_nom_prev = self.predict_x_nom(self.x_nom_prev, u_imu_body)
        self.x_nom_prev.ts = dt.now().timestamp()
        self.x_err_prev = self.predict_x_err(self.x_nom_prev, self.x_err_prev, u_imu_body)
        self.x_err_prev.ts = dt.now().timestamp()
        self.gauss_mutex.release()
        #return x_nom_pred, x_err_pred

    def inject(
        self, x_err_upd: ErrorStateGauss
    ) -> Tuple[NominalState, ErrorStateGauss]:
        """Perform the injection step, an implementation of Brekke
        (10.72), (10.85) and (10.86)

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_upd (ErrorStateGauss): updated error state gaussian

        Returns:
            x_nom_inj (NominalState): nominal state after injection
            x_err_inj (ErrorStateGauss): error state gaussian after injection
        """
        mean = np.zeros(15)
        G = np.eye(15)
        G[6:9, 6:9] = np.eye(3) - skew(0.5 * x_err_upd.avec)
        cov = G @ x_err_upd.cov @ G.T

        self.gauss_mutex.acquire()
        self.x_nom_prev = NominalState(
            self.x_nom_prev.pos + x_err_upd.pos,
            self.x_nom_prev.vel + x_err_upd.vel,
            self.x_nom_prev.ori.multiply(Quaternion(1, 0.5 * x_err_upd.avec)),
            self.x_nom_prev.acc_bias + x_err_upd.acc_bias,
            self.x_nom_prev.gyro_bias + x_err_upd.gyro_bias,
            self.x_nom_prev.ts,
        )
        self.x_nom_prev.ts = dt.now().timestamp()
        self.x_err_prev = ErrorStateGauss(mean, cov, x_err_upd.ts)
        self.x_err_prev.ts = dt.now().timestamp()
        self.gauss_mutex.release()

        #return x_nom_inj, x_err_inj
