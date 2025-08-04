import numpy as np
import h5py
from scipy.interpolate import RectBivariateSpline


class GridLikelihood:
    """Evaluate likelihood from a pre-computed interpolation table."""

    def __init__(self, table_file):
        with h5py.File(table_file, "r") as f:
            self.gamma = f["gamma_dm"][:]
            self.logM = f["logM_dm"][:]
            self.detJ = np.nan_to_num(f["detJ"][:])
            self.muA = np.nan_to_num(f["muA"][:])
            self.muB = np.nan_to_num(f["muB"][:])

    def likelihood(self, eta):
        """Return the likelihood value for hyper-parameters ``eta``.

        Parameters
        ----------
        eta : (mu0, beta, sigma)
            Hyper-parameters defining the weight function.
        """
        mu0, beta, sigma = eta
        if sigma <= 0:
            return 0.0
        weight = np.exp(
            -0.5
            * ((self.logM[None, :] - (mu0 + beta * self.gamma[:, None])) / sigma) ** 2
        )
        integrand = self.detJ * weight
        spline = RectBivariateSpline(self.gamma, self.logM, integrand)
        return float(
            spline.integral(
                self.gamma.min(),
                self.gamma.max(),
                self.logM.min(),
                self.logM.max(),
            )
        )
