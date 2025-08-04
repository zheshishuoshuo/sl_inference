"""Not a Simple test module."""
from .run_mcmc import run_mcmc
from .likelihood import log_posterior
from .interpolator import build_interp_list_for_lenses
from .mock_generator import run_mock_simulation
import numpy as np
from .utils import fit_alphasps
from .utils import fit_halo_model_crossfit_fixed_xi
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    mock_lens_data, mock_observed_data = run_mock_simulation(n_samples=)
    logMh_grid = np.linspace(12.5, 13.5, 100)

    logMstar_list, detJ_list = build_interp_list_for_lenses(
        mock_observed_data, logMh_grid, zl=0.3, zs=2.0
    )
    test_filename = "chains_eta4.h5"
    if os.path.exists(os.path.join(os.path.dirname(__file__),'chains', test_filename)):
        print(f"[INFO] 继续采样：读取已有文件 {test_filename}")
    
    sampler = run_mcmc(
        data_df=mock_observed_data,
        logMstar_interp_list=logMstar_list,
        detJ_interp_list=detJ_list,
        use_interp=True,
        log_posterior_func=log_posterior,
        backend_file=test_filename,
        nwalkers=10,
        nsteps=10000,
        ndim=5,
        initial_guess=np.array([12.5, 2.2, 0.5, 0.2, 0.1]),
        processes=10
        )

    samples = sampler.get_chain(flat=True, discard=500)
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

    fited_data_values = {
        "mu0": mu_h0,
        "beta": beta_h,
        "sigma": sigma_h,
        "mu_alpha": mualpha,
        "sigma_alpha": sigmaalpha
    }   

    for i, param1 in enumerate(g.x_vars):
        ax = g.axes[i, i]
        ax.axvline(fited_data_values[param1], color="blue", linestyle="--", linewidth=1.2)

        for j in range(i):
            ax = g.axes[i, j]
            ax.axvline(fited_data_values[g.x_vars[j]], color="blue", linestyle="--", linewidth=1)
            ax.axhline(fited_data_values[g.x_vars[i]], color="blue", linestyle="--", linewidth=1)
            ax.plot(fited_data_values[g.x_vars[j]], fited_data_values[g.x_vars[i]],
                    "bo", markersize=3)

    plt.show()


if __name__ == "__main__":
    run()
