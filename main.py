from .run_mcmc import run_mcmc
from .likelihood import log_posterior
from .mock_generator import run_mock_simulation
import multiprocessing as mp
from pathlib import Path
import numpy as np

def main():
    mock_lens_data, mock_observed_data = run_mock_simulation(n_samples=100)
    data_dir = Path(__file__).resolve().parent / "data" / "tables" / "sim_example"
    data_dir.mkdir(parents=True, exist_ok=True)
    mock_lens_data.to_csv(data_dir / "mock_lens_data.csv", index=True)
    mock_observed_data.to_csv(data_dir / "mock_observed_data.csv", index=True)
    sampler = run_mcmc(
        data_df=mock_observed_data,
        sim_id="sim_example",
        log_posterior_func=log_posterior,
        backend_file="mcmc_chain1000.h5",
        nwalkers=18,
        nsteps=5000,
        ndim=5,
        initial_guess=np.array([12.5, 2.0, 0.3, 0.05, 0.05]),
        processes=mp.cpu_count()
        )


if __name__ == "__main__":
    main()
