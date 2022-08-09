from numpy import ndarray
from dataclasses import dataclass

@dataclass
class ImuData:
    """Data as it comes from the IMU
    Args:
        ts (float): IMU measurement timestamp
        acc (ndarray[3]): accelerometer measurement
        avel (ndarray[3]): gyro measurement
    """
    ts: 'float'
    acc: 'ndarray[3]'
    avel: 'ndarray[3]'
