#!/usr/bin/python3

from mv_gaussian import MultiVariateGaussian
from datatypes.measurements import (
    ImuMeasurement,
    CorrectedImuMeasurement,
    GnssMeasurement,
)
from datatypes.eskf_states import NominalState, ErrorStateGauss


class PressureSensor:
    def __init_(self):
        pass

    def predict(
        self,
        x_nom: NominalState,
        x_err: ErrorStateGauss,
        z_gnss: GnssMeasurement,
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

        # TODO replace this with your own code
        z_dvl_pred_gauss = solution.eskf.ESKF.predict_gnss_measurement(
            self, x_nom, x_err, z_gnss
        )

        return z_dvl_pred_gauss

    def update(
        self,
        x_nom_prev: NominalState,
        x_err_prev: NominalState,
        z_gnss: GnssMeasurement,
    ) -> Tuple[NominalState, ErrorStateGauss, MultiVarGaussStamped]:
        """Method called every time an dvl measurement is received.


        Args:
            x_nom_prev (NominalState): [description]
            x_nom_prev (NominalState): [description]
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            x_nom_inj (NominalState): previous nominal state
            x_err_inj (ErrorStateGauss): previous error state
            z_gnss_pred_gauss (MultiVarGaussStamped): predicted gnss
                measurement, used for NIS calculations.
        """

        # TODO replace this with your own code
        x_nom_inj, x_err_inj, z_gnss_pred_gauss = solution.eskf.ESKF.update_from_gnss(
            self, x_nom_prev, x_err_prev, z_gnss
        )

        return x_nom_inj, x_err_inj, z_gnss_pred_gauss

    def update_x_err(
        self,
        x_nom: NominalState,
        x_err: ErrorStateGauss,
        z_gnss_pred_gauss: MultiVariateGaussian,
        z_gnss: GnssMeasurement,
    ) -> ErrorStateGauss:
        """Update the error state from a gnss measurement

        Hint: see (10.75)
        Due to numerical error its recomended use the robust calculation of
        posterior covariance.

        I_WH = np.eye(*P.shape) - W @ H
        P_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)

        Args:
            x_nom (NominalState): previous nominal state
            x_err (ErrorStateGauss): previous error state gaussian
            z_gnss_pred_gauss (MultiVarGaussStamped): gnss prediction gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            x_err_upd_gauss (ErrorStateGauss): updated error state gaussian
        """

        # TODO replace this with your own code
        x_err_upd_gauss = solution.eskf.ESKF.get_x_err_upd(
            self, x_nom, x_err, z_gnss_pred_gauss, z_gnss
        )

        return x_err_upd_gauss

    def measurment_jac(self, x_nom: NominalState) -> "ndarray[3,15]":
        """Get the measurement jacobian, H.

        Hint: the gnss antenna has a relative position to the center given by
        self.lever_arm. How will the gnss measurement change if the drone is
        rotated differently? Use get_cross_matrix and some other stuff :)

        Returns:
            H (ndarray[3, 15]): [description]
        """

        # TODO replace this with your own code
        H = solution.eskf.ESKF.get_gnss_measurment_jac(self, x_nom)

        return H

    def cov(self, z_gnss: GnssMeasurement) -> "ndarray[3,3]":
        """Use this function in predict_gnss_measurement to get R.
        Get gnss covariance estimate based on gnss estimated accuracy.

        All the test data has self.use_gnss_accuracy=False, so this does not
        affect the tests.

        There is no given solution to this function, feel free to play around!

        Returns:
            gnss_cov (ndarray[3,3]): the estimated gnss covariance
        """
        if self.use_gnss_accuracy and z_gnss.accuracy is not None:
            # play around with this part, the suggested way is not optimal
            gnss_cov = (z_gnss.accuracy / 3) ** 2 * self.gnss_cov

        else:
            # dont change this part
            gnss_cov = self.gnss_cov
        return gnss_cov
