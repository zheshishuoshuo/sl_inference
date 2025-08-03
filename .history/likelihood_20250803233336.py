from .utils import selection_function, mag_likelihood
from .lens_model import LensModel
from .lens_solver import solve_lens_parameters_from_obs, compute_detJ
from .cached_A import cached_A_interp
from .norm_computer.compute_norm_grid import logRe_of_logMsps
from scipy.stats import norm
from functools import lru_cache
import numpy as np
# def log_prior(eta): ...
# def log_likelihood(...): ...
# def log_posterior(...): ...
# def likelihood_single_fast_optimized(...): ...


# === 全局缓存变量 ===
# likelihood.py 顶部
_context = {
    "data_df": None,
    "logMstar_interp_list": None,
    "detJ_interp_list": None,
    "use_interp": False
}


def set_context(data_df, logMstar_interp_list, detJ_interp_list, use_interp=False):
    _context["data_df"] = data_df
    _context["logMstar_interp_list"] = logMstar_interp_list
    _context["detJ_interp_list"] = detJ_interp_list
    _context["use_interp"] = use_interp


def initializer_for_pool(data_df_, logMstar_list_, detJ_list_, use_interp_):
    set_context(
        data_df=data_df_,
        logMstar_interp_list=logMstar_list_,
        detJ_interp_list=detJ_list_,
        use_interp=use_interp_
    )


@lru_cache(maxsize=None)
def _solve_lens_parameters_cached(xA_obs, xB_obs, logRe_obs, logMh, zl, zs):
    return solve_lens_parameters_from_obs(xA_obs, xB_obs, logRe_obs, logMh, zl, zs)


@lru_cache(maxsize=None)
def _compute_detJ_cached(xA_obs, xB_obs, logRe_obs, logMh, zl, zs):
    return compute_detJ(xA_obs, xB_obs, logRe_obs, logMh, zl, zs)


def log_prior(eta):
    """Flat prior for the five inference parameters.

    The parameter ``xi`` is treated as a known hyper-parameter during
    inference and is fixed to zero.  Consequently it is not part of the
    ``eta`` vector any more.  The remaining parameters are checked only for
    broad physical bounds.
    """
    mu0, beta, sigma, mu_alpha, sigma_alpha = eta
    if not (
        12.5 < mu0 < 13.5
        and  < sigma < 5
        and 0 < sigma_alpha < 1
        and -0.2 < mu_alpha < 0.3
        and 0 < beta < 5
    ):
        return -np.inf
    return 0.0  # flat prior




def likelihood_single_fast_optimized(
    di, eta, gridN=35, zl=0.3, zs=2.0, ms=26.0, sigma_m=0.1, m_lim=26.5,
    logMstar_interp=None, detJ_interp=None, use_interp=False
):
    xA_obs, xB_obs, logM_sps_obs, logRe_obs, m1_obs, m2_obs = di
    mu0, beta, sigma, mu_alpha, sigma_alpha = eta
    xi = 0.0

    mu_DM_i = mu0 + beta * ((logM_sps_obs + mu_alpha) - 11.4)
    logMh_grid = np.linspace(mu_DM_i - 4 * sigma, mu_DM_i + 4 * sigma, gridN)
    logalpha_grid = np.linspace(
        mu_alpha - 4 * sigma_alpha, mu_alpha + 4 * sigma_alpha, gridN
    )

    logM_star_arr = np.empty(gridN)
    detJ_arr = np.empty(gridN)
    selA_arr = np.empty(gridN)
    selB_arr = np.empty(gridN)
    p_magA_arr = np.empty(gridN)
    p_magB_arr = np.empty(gridN)
    valid_mask = np.ones(gridN, dtype=bool)

    for i, logMh in enumerate(logMh_grid):
        try:
            if use_interp:
                logM_star = float(logMstar_interp(logMh))
                detJ = float(detJ_interp(logMh))
            else:
                logM_star, _ = _solve_lens_parameters_cached(
                    xA_obs, xB_obs, logRe_obs, logMh, zl, zs
                )
                detJ = _compute_detJ_cached(
                    xA_obs, xB_obs, logRe_obs, logMh, zl, zs
                )
            model = LensModel(
                logM_star=logM_star, logM_halo=logMh, logRe=logRe_obs, zl=zl, zs=zs
            )
            muA = model.mu_from_rt(xA_obs)
            muB = model.mu_from_rt(xB_obs)
            selA = selection_function(muA, m_lim, ms, sigma_m)
            selB = selection_function(muB, m_lim, ms, sigma_m)
            p_magA = mag_likelihood(m1_obs, muA, ms, sigma_m)
            p_magB = mag_likelihood(m2_obs, muB, ms, sigma_m)
        except Exception:
            valid_mask[i] = False
            logM_star = 0.0
            detJ = 0.0
            selA = selB = 0.0
            p_magA = p_magB = 0.0

        logM_star_arr[i] = logM_star
        detJ_arr[i] = detJ
        selA_arr[i] = selA
        selB_arr[i] = selB
        p_magA_arr[i] = p_magA
        p_magB_arr[i] = p_magB

    p_logalpha_arr = norm.pdf(logalpha_grid, loc=mu_alpha, scale=sigma_alpha)

    mu_DM_local_arr = mu0 + beta * (logM_star_arr - 11.4)
    p_logMh_arr = np.where(
        valid_mask,
        norm.pdf(logMh_grid, loc=mu_DM_local_arr, scale=sigma),
        0.0,
    )

    p_Mstar_arr = norm.pdf(
        logM_sps_obs,
        loc=logM_star_arr[:, None] - logalpha_grid[None, :],
        scale=0.1,
    )

    coeff = (
        detJ_arr
        * selA_arr
        * selB_arr
        * p_magA_arr
        * p_magB_arr
        * p_logMh_arr
    )[:, None]

    Z = p_Mstar_arr * p_logalpha_arr[None, :] * coeff

    integral = np.trapz(np.trapz(Z, logalpha_grid, axis=1), logMh_grid)
    return max(integral, 1e-300)


def log_likelihood(eta, **kwargs):
    # global _data_df, _logMstar_interp_list, _detJ_interp_list, _use_interp
    _data_df = _context["data_df"]
    _logMstar_interp_list = _context["logMstar_interp_list"]
    _detJ_interp_list = _context["detJ_interp_list"]
    _use_interp = _context["use_interp"]

    mu0, beta, sigma, mu_alpha, sigma_alpha = eta
    xi = 0.0

    if sigma <= 0 or sigma_alpha <= 0 or sigma > 2.0 or sigma_alpha > 2.0:
        return -np.inf

    try:
        A_eta = cached_A_interp(mu0, sigma, beta, xi)
        if not np.isfinite(A_eta) or A_eta <= 0:
            return -np.inf
    except Exception:
        return -np.inf

    logL = 0.0
    for i, (_, row) in enumerate(_data_df.iterrows()):
        try:
            di = tuple(row[col] for col in [
                'xA', 'xB', 'logM_star_sps_observed', 'logRe',
                'magnitude_observedA', 'magnitude_observedB'
            ])
        except:
            return -np.inf

        logMstar_interp = _logMstar_interp_list[i] if _use_interp else None
        detJ_interp = _detJ_interp_list[i] if _use_interp else None

        try:
            L_i = likelihood_single_fast_optimized(
                di, eta,
                logMstar_interp=logMstar_interp,
                detJ_interp=detJ_interp,
                use_interp=_use_interp,
                **kwargs
            )
            if not np.isfinite(L_i) or L_i <= 0:
                return -np.inf
            logL += np.log(L_i / A_eta)
        except:
            return -np.inf

    return logL


def log_posterior(eta):
    lp = log_prior(eta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(eta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll



