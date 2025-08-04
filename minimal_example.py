import os
import numpy as np
import emcee

from .mock_generator import run_mock_simulation
from .grid_builder import build_grid
from .likelihood import GridLikelihood


def main():
    mock_lens_data, mock_obs = run_mock_simulation(n_samples=100, process=0)
    if len(mock_obs) == 0:
        raise RuntimeError("no lensed systems found in mock data")

    os.makedirs("data/mock", exist_ok=True)
    os.makedirs("data/tables", exist_ok=True)
    os.makedirs("data/inference", exist_ok=True)

    mock_obs.to_csv("data/mock/mock_observed.csv", index=False)
    row = mock_obs.iloc[0]

    gamma_grid = np.linspace(-0.1, 0.1, 5)
    logM_grid = np.linspace(11.0, 13.0, 5)
    table_file = "data/tables/example_table.h5"
    build_grid(row["xA"], row["xB"], row["logRe"], gamma_grid, logM_grid, outfile=table_file)

    grid = GridLikelihood(table_file)

    def logprob(theta):
        mu0, beta, sigma = theta
        if sigma <= 0:
            return -np.inf
        L = grid.likelihood(theta)
        return np.log(L + 1e-300)

    sampler = emcee.EnsembleSampler(6, 3, logprob)
    p0 = np.array([12.0, 1.0, 0.5]) + 1e-4 * np.random.randn(6, 3)
    sampler.run_mcmc(p0, 1, progress=False)
    np.save("data/inference/chain.npy", sampler.get_chain())
    print("Example log probabilities:", sampler.get_log_prob())


if __name__ == "__main__":
    main()
