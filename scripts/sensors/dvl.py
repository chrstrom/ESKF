#!/usr/bin/python3

from eskf_types.multivariate_gaussian import MultiVariateGaussian

from eskf_types.state import NominalState, ErrorStateGauss

from numpy import ndarray
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from utilities.matrix import block_3x3

@dataclass
class DvlData:
    """Data as it comes from the DVL
    Args:
        ts (float): IMU measurement timestamp
        vel (ndarray[3]): velocity measurement
    """
    ts: float
    vel: 'ndarray[3]'
    cov: 'ndarray[3, 3]'


# TODO: replace GNSS with DVL
class DVL:
    def __init_(self):
        pass

    def predict(
        self,
        x_nom: NominalState,
        x_err: ErrorStateGauss,
        z_dvl: DvlData,
    ) -> MultiVariateGaussian:
        """Predict the dvl measurement

        Hint: z_gnss is only used in get_gnss_cov and to get timestamp for
        the predicted measurement

        Args:
            x_nom (NominalState): previous nominal state
            x_err (ErrorStateGauss): previous error state gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            z_gnss_pred_gauss (MultiVarGaussStamped): gnss prediction gaussian
        """

        mean = x_nom.vel
        H = self.measurment_jac(x_nom)
        cov = H@x_err.cov@H.T + z_dvl.cov

        z_gnss_pred_gauss = MultiVariateGaussian(mean, cov, z_dvl.ts)

        return z_gnss_pred_gauss

    def update(
        self,
        x_nom: NominalState,
        x_err: ErrorStateGauss,
        z_dvl: DvlData
    ) -> ErrorStateGauss:
        """Update the error state from a dvl measurement

        Args:
            x_nom (NominalState): previous nominal state
            x_err (ErrorStateGauss): previous error state gaussian
            z_gnss_pred_gauss (MultiVarGaussStamped): gnss prediction gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            x_err_upd_gauss (ErrorStateGauss): updated error state gaussian
        """

        z_dvl_pred = self.predict(x_nom, x_err, z_dvl)

        P = x_err.cov
        R = z_dvl.cov
        H = self.measurment_jac(x_nom)

        W = P@H.T@np.linalg.inv(H@P@H.T + R)
        I_WH = np.eye(*P.shape) - W @ H

        P_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)
        mean = W@(z_dvl.vel - z_dvl_pred.mean)

        x_err_upd_gauss = ErrorStateGauss(mean, P_upd, z_dvl.ts)

        return x_err_upd_gauss

    def measurment_jac(self, x_nom: NominalState) -> "ndarray[3,15]":
        """Get the measurement jacobian, H.

        Hint: the gnss antenna has a relative position to the center given by
        self.lever_arm. How will the gnss measurement change if the drone is
        rotated differently? Use get_cross_matrix and some other stuff :)

        Returns:
            H (ndarray[3, 15]): [description]
        """

        H = np.zeros((3, 15))
        H[block_3x3(0, 1)] = np.eye(3)
        
        return H 
