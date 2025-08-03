from scipy.special import erf
import numpy as np
from scipy.stats import norm

# def selection_function(mu, m_lim, m_source, sigma_m): ...
# def mag_likelihood(m_obs, mu, m_source, sigma_m): ...


def selection_function(mu, m_lim, ms, sigma_m):
    """选择函数（单个图像）"""
    return 0.5 * (1 + erf((m_lim - ms + 2.5 * np.log10(np.abs(mu))) / (np.sqrt(2) * sigma_m)))

def mag_likelihood(m_obs, mu, ms, sigma_m):
    """星等的似然函数"""
    mag_model = ms - 2.5 * np.log10(np.abs(mu))
    return norm.pdf(m_obs, loc=mag_model, scale=sigma_m)



# fit tools

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

def fit_halo_model_simple_fixed_xi(mock_lens_data, params, center_mass=11.4):
    """
    已知 mu_R0, beta_R 时，固定 xi_h = 0，只拟合 logMh ~ (logM - 11.4)
    """
    logM  = np.asarray(mock_lens_data['logM_star_sps'])
    logMh = np.asarray(mock_lens_data['logMh'])

    x1 = logM - center_mass
    X = x1.reshape(-1, 1)

    reg = LinearRegression().fit(X, logMh)
    resid = logMh - reg.predict(X)
    sigma_h = float(resid.std(ddof=2))  # 截距+1系数 ⇒ ddof=2

    return {
        "mu_h0": float(reg.intercept_),
        "beta_h": float(reg.coef_[0]),
        "xi_h":   0.0,  # 固定
        "sigma_h": sigma_h,
        "n": len(logMh),
    }


def fit_halo_model_crossfit_fixed_xi(mock_lens_data, center_mass=11.4):
    """
    不知道 mu_R0, beta_R 时，固定 xi_h = 0 的交叉验证版本：
    仅用 logMh ~ (logM - 11.4) 线性拟合，忽略 logRe 的贡献。
    """
    logM  = np.asarray(mock_lens_data['logM_star_sps'])
    logMh = np.asarray(mock_lens_data['logMh'])

    x1 = logM - center_mass
    X = x1.reshape(-1, 1)

    reg = LinearRegression().fit(X, logMh)
    resid = logMh - reg.predict(X)
    sigma_h = float(resid.std(ddof=2))

    return {
        "mu_h0": float(reg.intercept_),
        "beta_h": float(reg.coef_[0]),
        "xi_h":   0.0,  # 固定
        "sigma_h": sigma_h,
        "n": len(logMh),
    }

def fit
