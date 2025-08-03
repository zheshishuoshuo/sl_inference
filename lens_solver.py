from .lens_model import LensModel
from scipy.interpolate import splrep, splint
from .sl_cosmology import Dang, Mpc, c, G, M_Sun, rhoc
import numpy as np
from .sl_profiles import nfw, deVaucouleurs as deV
from scipy.optimize import brentq


def precompute_sigma_spline(logM_halo, zl, rs=10.0):
    """Pre-compute the NFW surface-density grid for a halo mass.

    Parameters
    ----------
    logM_halo : float
        Base-10 logarithm of the halo mass in solar masses.
    zl : float
        Lens redshift.
    rs : float, optional
        NFW scale radius in kpc.  Defaults to ``10.0`` to match the previous
        implementation.

    Returns
    -------
    tuple of ``(Rkpc, Sigma, sigmaR_spline)`` where

    ``Rkpc``
        Radius grid in kpc.
    ``Sigma``
        Surface mass density evaluated on ``Rkpc``.
    ``sigmaR_spline``
        Spline representation of ``Sigma(R) * R`` suitable for fast
        integration in the pure Python fallback.
    """

    M_halo = 10 ** logM_halo  # [Msun]
    rhoc_z = rhoc(zl)  # [Msun / Mpc^3]
    r200 = (M_halo * 3.0 / (4 * np.pi * 200 * rhoc_z)) ** (1.0 / 3.0) * 1000.0
    nfw_norm = M_halo / nfw.M3d(r200, rs)

    R2d = np.logspace(-3, 2, 1001)
    Rkpc = R2d * rs  # [kpc]
    Sigma = nfw_norm * nfw.Sigma(Rkpc, rs)  # [Msun / kpc^2]
    sigmaR_spline = splrep(Rkpc, Sigma * Rkpc)

    return Rkpc, Sigma, sigmaR_spline

# Try to import the Cython-accelerated routines.  If the extension is not
# available (e.g., on platforms without a C compiler), fall back to the pure
# Python implementations below.
try:  # pragma: no cover - optional acceleration
    from .lens_solver_cy import (
        solve_single_lens as _solve_single_lens_cy,
        alpha_star_unit as alpha_star_unit_cy,
        alpha_halo as alpha_halo_cy,
    )
    _CYTHON_AVAILABLE = True
except Exception:  # pragma: no cover - extension not compiled
    _CYTHON_AVAILABLE = False


def solve_single_lens(model, beta_unit):
    """Solve the lens equation for a given ``LensModel``.

    If the optional Cython extension is available, the root finding and
    deflection calculations are performed in C for speed.  Otherwise a pure
    Python implementation using :func:`scipy.optimize.brentq` is used.
    """
    caustic_max_at_lens_plane = model.solve_xradcrit()  # [kpc]
    caustic_max_at_source_plane = model.solve_ycaustic()  # [kpc]
    beta = beta_unit * caustic_max_at_source_plane  # [kpc]
    einstein_radius = model.einstein_radius()  # [kpc]

    if _CYTHON_AVAILABLE:
        # Use the compiled implementation.  Pass in the pre-computed NFW grid
        # stored on the model to avoid Python overhead inside the solver.
        return _solve_single_lens_cy(
            model.Rkpc,
            model._Sigma,
            model.M_star,
            model.Re,
            model.s_cr,
            beta,
            einstein_radius,
            caustic_max_at_lens_plane,
        )

    # Fallback Python implementation using SciPy's brentq root finder.
    def lens_equation(x):
        return model.alpha(x) - x + beta

    xA = brentq(lens_equation, einstein_radius, 100 * einstein_radius)
    xB = brentq(lens_equation, -einstein_radius, -caustic_max_at_lens_plane)
    return xA, xB

def solve_lens_parameters_from_obs(
    xA_obs, xB_obs, logRe_obs, logM_halo, zl, zs, precomputed=None
):

    Re = 10 ** logRe_obs  # [kpc]

    # Pre-compute or unpack the NFW grid for this halo mass.
    if precomputed is None:
        Rkpc, Sigma, sigmaR_spline = precompute_sigma_spline(logM_halo, zl)
    else:
        Rkpc, Sigma, sigmaR_spline = precomputed

    dd = Dang(zl)  # [Mpc]
    ds = Dang(zs)  # [Mpc]
    dds = Dang(zs, zl)  # [Mpc]
    kpc = Mpc / 1000.0  # [kpc/Mpc]
    s_cr = c**2 / (4 * np.pi * G) * ds / dds / dd / Mpc / M_Sun * kpc**2

    if _CYTHON_AVAILABLE:
        def alpha_star_unit(x):
            return alpha_star_unit_cy(x, Re, s_cr)

        def alpha_halo(x):
            return alpha_halo_cy(x, Rkpc, Sigma, s_cr)
    else:  # Python fallbacks using spline integration
        def alpha_star_unit(x):
            m2d_star = deV.fast_M2d(abs(x) / Re)
            return m2d_star / (np.pi * x * s_cr)

        def alpha_halo(x):
            m2d_halo = 2 * np.pi * splint(0.0, abs(x), sigmaR_spline)
            return m2d_halo / (np.pi * x * s_cr)

    M_star_solved = ((xA_obs -xB_obs) + alpha_halo(xB_obs)-alpha_halo(xA_obs))/ (alpha_star_unit(xA_obs) - alpha_star_unit(xB_obs))  # [Msun]
    beta_solved = -(alpha_star_unit(xA_obs)*(xB_obs-alpha_halo(xB_obs)) -alpha_star_unit(xB_obs)*(xA_obs-alpha_halo(xA_obs))) / (alpha_star_unit(xB_obs) - alpha_star_unit(xA_obs))  # [kpc]

    another_solution_beta = M_star_solved*alpha_star_unit(xA_obs) +alpha_halo(xA_obs) - xA_obs  # [kpc]
    
    # logbeta_solved = np.log10(beta_solved)  # [kpc]
    if M_star_solved <= 0:
        # print(f"Warning at Mh = {logM_halo}, xA = {xA_obs}, xB = {xB_obs}, Re = {Re}, beta = {beta_solved}")
        raise ValueError(f"Invalid M_star_solved = {M_star_solved}, must be > 0")
    logM_star_solved = np.log10(M_star_solved)  # [Msun]
    model = LensModel(logM_star=logM_star_solved, logM_halo=logM_halo, logRe=np.log10(Re), zl=zl, zs=zs)
    caustic_max_at_lens_plane = model.solve_xradcrit()  # [kpc]
    caustic_max_at_source_plane = model.solve_ycaustic()  # [kpc]
    # beta = beta_unit * caustic_max_at_source_plane  # [kpc]
    beta_unit = beta_solved / caustic_max_at_source_plane  # [kpc]
    # print('truebeta',beta_solved, another_solution_beta,beta_unit)

    return logM_star_solved, beta_unit  


def compute_detJ(theta1_obs, theta2_obs, logRe_obs, logMh, zl=0.3, zs=2.0):
    delta = 1e-4

    precomputed = precompute_sigma_spline(logMh, zl)

    logM0, beta0 = solve_lens_parameters_from_obs(
        theta1_obs, theta2_obs, logRe_obs, logMh, zl, zs, precomputed
    )
    logM1, beta1 = solve_lens_parameters_from_obs(
        theta1_obs + delta, theta2_obs, logRe_obs, logMh, zl, zs, precomputed
    )
    logM2, beta2 = solve_lens_parameters_from_obs(
        theta1_obs, theta2_obs + delta, logRe_obs, logMh, zl, zs, precomputed
    )

    dlogM_dtheta1 = (logM1 - logM0) / delta
    dlogM_dtheta2 = (logM2 - logM0) / delta
    dbeta_dtheta1 = (beta1 - beta0) / delta
    dbeta_dtheta2 = (beta2 - beta0) / delta

    J = np.array([[dlogM_dtheta1, dlogM_dtheta2],
                  [dbeta_dtheta1, dbeta_dtheta2]])
    return np.abs(np.linalg.det(J))
