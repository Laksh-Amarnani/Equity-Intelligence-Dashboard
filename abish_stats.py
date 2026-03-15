import numpy as np
import pandas as pd
import scipy.stats as stats 
from scipy.optimize import minimize_scalar
from scipy.stats import(
    norm,t as t_dist,chi2,
    ttest_1samp, ttest_ind,ttest_rel,
    f_oneway,mannwhitneyu, kruskal, wilcoxon,
    shapiro,jarque_bera
)

import warnings
warnings.filterwarnings('ignore')
#Confidence Intervals
def confidence_interval_mean(data: np.ndarray, confidence:float=0.95) -> dict:
    n=len(data)
    mean=np.mean(data)
    se= stats.sem(data)

    if n > 30:
        z = norm.ppf((1 + confidence) / 2)
        margin = z * se
        method = "Z-interval (large sample)"
    else:
        df=n-1
        t_val=t_dist.ppf((1 + confidence) / 2, df)
        margin = t_val * se
        method = "t-interval (small sample)"
    return {
        "mean": round(mean,6),
        "lower": round(mean - margin,6),
        "upper": round(mean + margin,6),
        "std_error": round(se,6),
        "n": n,
        "confidence": confidence,
        "method": method,
    }

def confidence_interval_volatility(data: np.ndarray, confidence:float=0.95) -> dict:
    n=len(data)
    var=np.var(data, ddof=1)
    df=n-1
    alpha  = 1 - confidence

    chi2_lower = chi2.ppf(alpha / 2, df)
    chi2_upper = chi2.ppf(1 - alpha / 2, df)

    var_lower=(df * var) / chi2_lower
    var_upper=(df * var) / chi2_upper

    return {
        "sample_variance": round(var,8),
        "sample_std":      round(np.sqrt(var),6),
        "variance_lower":  round(var_lower,8),
        "variance_upper":  round(var_upper,8),
        "std_lower":       round(np.sqrt(var_lower), 6),
        "std_upper":       round(np.sqrt(var_upper), 6),
        "confidence":      confidence,
    }
#Maximum Likelihood Estimation MLE

def mle_normal(data: np.ndarray) -> dict:
    mu_mle = np.mean(data)
    sigma_mle = np.std(data, ddof=0)
    log_likelihood=np.sum(norm.logpdf(data, loc=mu_mle, scale=sigma_mle))
    return {
        "mu_mle":         round(mu_mle,6),
        "sigma_mle":      round(sigma_mle,6),
        "log_likelihood": round(log_likelihood,4),
        "aic":            round(2 * 2 - 2* log_likelihood,4),
        "interpretation": (
            f"The MLE estimates the stock's daily return as "
            f"μ = {mu_mle*100:.3f}% with σ = {sigma_mle*100:.3f}%"

        ) 
    }
def plot_likelihood_surface(data: np.ndarray, mu_range: tuple = None) -> dict:
    sigma_mle = np.std(data, ddof=0)
    if mu_range is None:
        mu_mle=np.mean(data)
        mu_range = (mu_mle - 4 * sigma_mle, mu_mle + 4 * sigma_mle)
    
    mu_grid = np.linspace(mu_range[0], mu_range[1], 300)
    log_likes=[np.sum(norm.logpdf(data, loc=mu, scale=sigma_mle)) for mu in mu_grid]
    return {
        "mu_grid":        mu_grid,
        "log_likelihoods": np.array(log_likes),
        "mle_mu":         np.mean(data),
        "peak_ll":        max(log_likes),
    }