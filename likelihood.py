from .utils import selection_function, mag_likelihood
from .lens_model import LensModel
from .cached_A import cached_A_interp
from .norm_computer.compute_norm_grid import logRe_of_logMsps
from scipy.stats import norm
from pathlib import Path
import numpy as np
# def log_prior(eta): ...
# def log_likelihood(...): ...
# def log_posterior(...): ...
# def likelihood_single_fast_optimized(...): ...


# === 全局缓存变量 ===
# likelihood.py 顶部
_context = {
    "data_df": None,
    "precomputed_tables": None,
}


def set_context(data_df, precomputed_tables):
    _context["data_df"] = data_df
    _context["precomputed_tables"] = precomputed_tables


def initializer_for_pool(data_df_, tables_):
    set_context(
        data_df=data_df_,
        precomputed_tables=tables_,
    )


def load_precomputed_tables(sim_id):
    """Load pre-computed lensing tables for a given simulation id.

    Parameters
    ----------
    sim_id : str
        Identifier of the simulation run. The function expects files of the
        form ``tables/<sim_id>/lens_*_grid.npz`` to be present.

    Returns
    -------
    list[dict]
        A list with one entry per lens. Each entry is a dictionary containing
        ``logMh_grid``, ``logM_star`` and ``detJ`` arrays.

    Raises
    ------
    FileNotFoundError
        If the directory or required ``npz`` files are missing.
    """

    tables_dir = Path(__file__).resolve().parent / "tables" / sim_id
    if not tables_dir.exists():
        raise FileNotFoundError(
            f"Precomputed tables for sim_id '{sim_id}' not found at {tables_dir}"
        )

    npz_files = sorted(tables_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No npz files found in precomputed table directory {tables_dir}"
        )

    tables = []
    for file in npz_files:
        with np.load(file) as data:
            tables.append(
                {
                    "logMh_grid": data["logMh_grid"],
                    "logM_star": data["logM_star"],
                    "detJ": data["detJ"],
                }
            )

    return tables


def log_prior(eta):
    """Flat prior for the five inference parameters.

    The parameter ``xi`` is treated as a known hyper-parameter during
    inference and is fixed to zero.  Consequently it is not part of the
    ``eta`` vector any more.  The remaining parameters are checked only for
    broad physical bounds.
    """
    mu0, beta, sigma, mu_alpha, sigma_alpha = eta
    if not (
        10 < mu0 < 15
        and 0 < sigma < 1
        and 0 < sigma_alpha < 1
        and 0 < mu_alpha < 1
        and 0 < beta < 5
    ):
        return -np.inf
    return 0.0  # flat prior


    # if not (
    #     12 < mu0 < 14
    #     and 0 < sigma < 0.7
    #     and 0 < sigma_alpha < 0.3
    #     and 0 < mu_alpha < 0.3
    #     and 0 < beta < 5
    # ):
    #     return -np.inf
    # return 0.0  # flat prior




def likelihood_single_fast_optimized(
    di,
    eta,
    table,
    zl=0.3,
    zs=2.0,
    ms=26.0,
    sigma_m=0.1,
    m_lim=26.5,
):
    xA_obs, xB_obs, logM_sps_obs, logRe_obs, m1_obs, m2_obs = di
    mu0, beta, sigma, mu_alpha, sigma_alpha = eta

    logMh_grid = table["logMh_grid"]
    logM_star_arr = table["logM_star"]
    detJ_arr = table["detJ"]
    gridN = len(logMh_grid)

    logalpha_grid = np.linspace(
        mu_alpha - 4 * sigma_alpha, mu_alpha + 4 * sigma_alpha, gridN
    )

    selA_arr = np.empty(gridN)
    selB_arr = np.empty(gridN)
    p_magA_arr = np.empty(gridN)
    p_magB_arr = np.empty(gridN)
    valid_mask = np.ones(gridN, dtype=bool)

    for i, (logMh, logM_star) in enumerate(zip(logMh_grid, logM_star_arr)):
        try:
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
            selA = selB = 0.0
            p_magA = p_magB = 0.0

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
    _data_df = _context["data_df"]
    _tables = _context["precomputed_tables"]

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

        try:
            table = _tables[i] if _tables is not None else None
            L_i = likelihood_single_fast_optimized(
                di,
                eta,
                table=table,
                **kwargs,
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



