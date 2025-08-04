#!/usr/bin/env python
"""Precompute lensing tables for simulated lenses.

This script generates mock lensed systems and tabulates the lensing
solutions on a grid of halo masses. The resulting arrays are saved under
``tables/<sim_id>/`` as ``lens_<id>_grid.npz`` files together with a
``metadata.json`` description of the simulation.

The script skips already existing ``npz`` files to avoid redundant
computation.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import internal modules without requiring the full package to be installed.
# We create a lightweight ``sl_inference`` package so that modules using
# relative imports (e.g. ``from .lens_properties import ...``) can be loaded.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if "sl_inference" not in sys.modules:
    pkg = types.ModuleType("sl_inference")
    pkg.__path__ = [str(ROOT)]
    sys.modules["sl_inference"] = pkg

import importlib

mock_generator = importlib.import_module("sl_inference.mock_generator")
interpolator = importlib.import_module("sl_inference.interpolator")


# ---------------------------------------------------------------------------
# Main functionality
# ---------------------------------------------------------------------------

def build_tables(
    n_samples: int,
    logmh_min: float,
    logmh_max: float,
    logmh_step: float,
    sim_id: str,
    mag_source: float,
    maximum_magnitude: float,
    zl: float,
    zs: float,
    process: int,
) -> None:
    """Generate mock lenses and tabulate lensing solutions."""

    logmh_grid = np.arange(logmh_min, logmh_max + logmh_step, logmh_step)

    out_dir = ROOT / "tables" / sim_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mock_lens_data, _ = mock_generator.run_mock_simulation(
        n_samples,
        mag_source=mag_source,
        maximum_magnitude=maximum_magnitude,
        zl=zl,
        zs=zs,
        process=process,
    )

    for lens_id, row in mock_lens_data.reset_index(drop=True).iterrows():
        file_path = out_dir / f"lens_{lens_id}_grid.npz"
        if file_path.exists():
            continue

        xA, xB, logRe = row["xA"], row["xB"], row["logRe"]

        logmstar_interp = interpolator.solve_lens_tabulate(
            logmh_grid, xA, xB, logRe, zl=zl, zs=zs
        )
        detj_interp = interpolator.detJ_tabulate(
            logmh_grid, xA, xB, logRe, zl=zl, zs=zs
        )

        logmstar_vals = logmstar_interp(logmh_grid)
        detj_vals = detj_interp(logmh_grid)

        np.savez(
            file_path,
            logMh_grid=logmh_grid,
            logM_star=logmstar_vals,
            detJ=detj_vals,
        )

    metadata = {
        "n_samples": int(n_samples),
        "n_lenses": int(len(mock_lens_data)),
        "logmh_min": float(logmh_min),
        "logmh_max": float(logmh_max),
        "logmh_step": float(logmh_step),
        "mag_source": float(mag_source),
        "maximum_magnitude": float(maximum_magnitude),
        "zl": float(zl),
        "zs": float(zs),
        "process": int(process),
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mock lensing tables"
    )
    parser.add_argument("--n-samples", type=int, default=10, help="number of simulated lenses")
    parser.add_argument("--logmh-min", type=float, default=11.0)
    parser.add_argument("--logmh-max", type=float, default=14.0)
    parser.add_argument("--logmh-step", type=float, default=0.1)
    parser.add_argument(
        "--sim-id",
        type=str,
        default=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        help="identifier for this simulation run",
    )
    parser.add_argument("--mag-source", type=float, default=26.0)
    parser.add_argument("--maximum-magnitude", type=float, default=26.5)
    parser.add_argument("--zl", type=float, default=0.3)
    parser.add_argument("--zs", type=float, default=2.0)
    parser.add_argument("--process", type=int, default=0, help="number of processes for simulation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_tables(
        n_samples=args.n_samples,
        logmh_min=args.logmh_min,
        logmh_max=args.logmh_max,
        logmh_step=args.logmh_step,
        sim_id=args.sim_id,
        mag_source=args.mag_source,
        maximum_magnitude=args.maximum_magnitude,
        zl=args.zl,
        zs=args.zs,
        process=args.process,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
