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