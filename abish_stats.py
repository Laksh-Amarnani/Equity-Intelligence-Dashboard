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

