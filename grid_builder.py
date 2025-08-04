import numpy as np
import h5py

from .lens_solver import (
    solve_lens_parameters_from_obs,
    compute_detJ,
    solve_single_lens,
)
from .lens_model import LensModel


def build_grid(
    xA_obs,
    xB_obs,
    logRe_obs,
    gamma_dm_grid,
    logM_dm_grid,
    zl=0.3,
    zs=2.0,
    outfile="data/tables/grid.h5",
):
    """Pre-compute lensing observables on a 2-D grid.

    Parameters
    ----------
    xA_obs, xB_obs : float
        Observed image positions in kpc.
    logRe_obs : float
        Observed effective radius in kpc (log10).
    gamma_dm_grid, logM_dm_grid : array_like
        Grids of dark-matter slope ``gamma_dm`` and halo mass ``logM_dm``.
    zl, zs : float
        Lens and source redshifts.
    outfile : str
        Path to the HDF5 file where the table will be stored.

    Returns
    -------
    str
        Path to the generated HDF5 file.
    """

    gamma_dm_grid = np.asarray(gamma_dm_grid)
    logM_dm_grid = np.asarray(logM_dm_grid)

    ng, nm = len(gamma_dm_grid), len(logM_dm_grid)
    shape = (ng, nm)
    logM_star = np.empty(shape)
    detJ = np.empty(shape)
    muA = np.empty(shape)
    muB = np.empty(shape)
    xA_model = np.empty(shape)
    xB_model = np.empty(shape)

    for i, g in enumerate(gamma_dm_grid):
        for j, logM in enumerate(logM_dm_grid):
            logMh = logM + g  # simple mapping for demo purposes
            try:
                logM_s, beta_unit = solve_lens_parameters_from_obs(
                    xA_obs, xB_obs, logRe_obs, logMh, zl, zs
                )
                detJ_val = compute_detJ(xA_obs, xB_obs, logRe_obs, logMh, zl, zs)
                model = LensModel(
                    logM_star=logM_s, logM_halo=logMh, logRe=logRe_obs, zl=zl, zs=zs
                )
                xA_pred, xB_pred = solve_single_lens(model, beta_unit)
                muA[i, j] = model.mu_from_rt(xA_pred)
                muB[i, j] = model.mu_from_rt(xB_pred)
                logM_star[i, j] = logM_s
                detJ[i, j] = detJ_val
                xA_model[i, j] = xA_pred
                xB_model[i, j] = xB_pred
            except Exception:
                logM_star[i, j] = np.nan
                detJ[i, j] = np.nan
                muA[i, j] = np.nan
                muB[i, j] = np.nan
                xA_model[i, j] = np.nan
                xB_model[i, j] = np.nan

    with h5py.File(outfile, "w") as f:
        f.create_dataset("gamma_dm", data=gamma_dm_grid)
        f.create_dataset("logM_dm", data=logM_dm_grid)
        f.create_dataset("logM_star", data=logM_star)
        f.create_dataset("detJ", data=detJ)
        f.create_dataset("muA", data=muA)
        f.create_dataset("muB", data=muB)
        f.create_dataset("xA_model", data=xA_model)
        f.create_dataset("xB_model", data=xB_model)
        f.attrs["xA_obs"] = xA_obs
        f.attrs["xB_obs"] = xB_obs
        f.attrs["logRe_obs"] = logRe_obs
        f.attrs["zl"] = zl
        f.attrs["zs"] = zs

    return outfile
