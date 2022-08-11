from numpy import ndarray
from dataclasses import dataclass
from typing import Optional

from eskf_types.multivariate_gaussian import MultiVariateGaussian
from eskf_types.quaternion import Quaternion

@dataclass
class NominalState:
    """Class representing a nominal state as in Brekke (Table 10.1)
    Args:
        pos (ndarray[3]): position in NED
        vel (ndarray[3]): velocity in NED
        ori (Quaternion): orientation as a quaternion in NED
        accm_bias (ndarray[3]): accelerometer bias
        gyro_bias (ndarray[3]): gyro bias
    """

    pos: "ndarray[3]"
    vel: "ndarray[3]"
    ori: "Quaternion"
    acc_bias: "ndarray[3]"
    gyro_bias: "ndarray[3]"

    ts: Optional[float] = None


@dataclass
class ErrorStateGauss(MultiVariateGaussian):
    """A multivariate gaussian representing the error state."""

    # def __post_init__(self):
    #     super().__post_init__()

    @property
    def pos(self):
        """position"""
        return self.mean[0:3]

    @property
    def vel(self):
        """velocity"""
        return self.mean[3:6]

    @property
    def avec(self):
        """angles vector
        this is often called a rotation vector
        """
        return self.mean[6:9]

    @property
    def acc_bias(self):
        """accelerometer bias"""
        return self.mean[9:12]

    @property
    def gyro_bias(self):
        """gyro bias"""
        return self.mean[12:15]

