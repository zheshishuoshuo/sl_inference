"""Not a Simple test module."""
from .run_mcmc import run_mcmc
from .likelihood import log_posterior
from .mock_generator import run_mock_simulation
import numpy as np
from .utils import fit_alphasps
from .utils import fit_halo_model_crossfit_fixed_xi
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    mock_lens_data, mock_observed_data = run_mock_simulation(n_samples=20)
    test_filename = "chains_eta4.h5"

    sampler = run_mcmc(
        data_df=mock_observed_data,
        sim_id="sim_example",
        log_posterior_func=log_posterior,
        backend_file=test_filename,
        nwalkers=10,
        nsteps=500,
        ndim=5,
        initial_guess=np.array([12, 2.2, 0.5, 0.2, 0.1]),
        processes=min(2, mp.cpu_count()),
    )

    samples = sampler.get_chain(flat=True, discard=100)
    df = pd.DataFrame(samples, columns=[
        "mu0", "beta", "sigma", "mu_alpha", "sigma_alpha"
    ])
    print(f"[INFO] 采样完成，样本数量: {len(samples)}")


    true_values = {
    "mu0": 12.91,
    "beta": 2.04,
    "sigma": 0.37,
    "mu_alpha": 0.1,
    "sigma_alpha": 0.05
    }

    # === 画图 ===
    g = sns.pairplot(df,
                    diag_kind="kde",
                    markers=".",
                    plot_kws={"alpha": 0.5, "s": 10},
                    corner=True,
                    )

    # === 添加真值线 ===
    for i, param1 in enumerate(g.x_vars):
        ax = g.axes[i, i]
        ax.axvline(true_values[param1], color="red", linestyle="--", linewidth=1.2)

        for j in range(i):
            ax = g.axes[i, j]
            ax.axvline(true_values[g.x_vars[j]], color="red", linestyle="--", linewidth=1)
            ax.axhline(true_values[g.x_vars[i]], color="red", linestyle="--", linewidth=1)
            ax.plot(true_values[g.x_vars[j]], true_values[g.x_vars[i]],
                    "ro", markersize=3)
            
    # 用蓝色线画出数据拟合结果



    fitted = fit_halo_model_crossfit_fixed_xi(mock_lens_data=mock_lens_data)
    mu_h0 = fitted['mu_h0']
    beta_h = fitted['beta_h']
    xi_h = fitted['xi_h']
    sigma_h = fitted['sigma_h']

    mualpha, sigmaalpha = fit_alphasps(mock_lens_data=mock_lens_data)

    fitted_data_values = {
        "mu0": mu_h0,
        "beta": beta_h,
        "sigma": sigma_h,
        "mu_alpha": mualpha,
        "sigma_alpha": sigmaalpha
    }

    for i, param1 in enumerate(g.x_vars):
        ax = g.axes[i, i]
        ax.axvline(fitted_data_values[param1], color="blue", linestyle="--", linewidth=1.2)

        for j in range(i):
            ax = g.axes[i, j]
            ax.axvline(fitted_data_values[g.x_vars[j]], color="blue", linestyle="--", linewidth=1)
            ax.axhline(fitted_data_values[g.x_vars[i]], color="blue", linestyle="--", linewidth=1)
            ax.plot(
                fitted_data_values[g.x_vars[j]],
                fitted_data_values[g.x_vars[i]],
                "bo",
                markersize=3,
            )

    plt.show()


if __name__ == "__main__":
    run()
