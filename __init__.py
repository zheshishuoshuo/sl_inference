"""sl_inference package

Tools for modelling strong gravitational lenses and performing inference on their physical properties."""

from .lens_model import LensModel
from .lens_solver import (
    solve_single_lens,
    solve_lens_parameters_from_obs,
    compute_detJ,
    precompute_sigma_spline,
)
from .mass_sampler import (
    mstar_gene,
    logRe_given_logM,
    logMh_given_logM_logRe,
    generate_samples,
)
from .mock_generator import run_mock_simulation
from .grid_builder import build_grid
from .likelihood import GridLikelihood

__all__ = [
    "LensModel",
    "solve_single_lens",
    "solve_lens_parameters_from_obs",
    "compute_detJ",
    "precompute_sigma_spline",
    "mstar_gene",
    "logRe_given_logM",
    "logMh_given_logM_logRe",
    "generate_samples",
    "run_mock_simulation",
    "build_grid",
    "GridLikelihood",
]

__version__ = "0.2"
