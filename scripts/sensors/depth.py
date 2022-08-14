#!/usr/bin/python3

import numpy as np
from numpy import ndarray
from dataclasses import dataclass
from typing import Tuple

from eskf_types.multivariate_gaussian import MultiVariateGaussian
from eskf_types.state import NominalState, ErrorStateGauss


@dataclass
class DepthData:
    """Data as it comes from the pressure sensor
    Args:
        ts (float): IMU measurement timestamp
        pressure (float): Pressure sensor measurement
        cov (float): (Co)variance for the measurement
    """
    ts: float
    depth: float
    cov: float



class DepthSensor:
    def __init_(self):
        pass

    def predict(
        self,
        x_nom: NominalState,
        x_err: ErrorStateGauss,
        z_depth: DepthData,
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

        mean = x_nom.pos[2]
        H = self.measurement_jac(x_nom)
        cov = H@x_err.cov@H.T + z_depth.cov

        z_gnss_pred_gauss = MultiVariateGaussian(mean, cov, z_depth.ts)

        return z_gnss_pred_gauss

    def update(
        self,
        x_nom: NominalState,
        x_err: ErrorStateGauss,
        z_depth: DepthData
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
        z_depth_pred = self.predict(x_nom, x_err, z_depth)

        P = x_err.cov
        R = z_depth.cov
        H = self.measurement_jac(x_nom)

        W = P@H.T@np.linalg.inv(H@P@H.T + R)
        I_WH = np.eye(*P.shape) - W @ H

        P_upd = (I_WH @ P @ I_WH.T + R * W @ W.T)
        mean = W * (z_depth.depth - z_depth_pred.mean)
        mean = mean.ravel()

        x_err_upd_gauss = ErrorStateGauss(mean, P_upd, z_depth.ts)

        return x_err_upd_gauss

    def measurement_jac(self, x_nom: NominalState) -> "ndarray[3,15]":
        """Get the measurement jacobian, H.
        Returns:
            H (ndarray[1, 15]): [description]
        """

        H = np.zeros((1, 15))
        H[0][2] = 1
        
        return H 