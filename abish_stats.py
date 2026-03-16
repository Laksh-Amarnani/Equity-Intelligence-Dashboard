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

#Hypothesis Testing
def z_test_mean(data: np.ndarray, pop_mean: float,pop_std:float, alpha: float = 0.05, tail: str = "two") -> dict:
    n = len(data)
    x_bar = np.mean(data)
    se = pop_std / np.sqrt(n)
    z_stat = (x_bar - pop_mean) / se
    if tail == "two":
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        z_crit = norm.ppf(1 - alpha / 2)
    elif tail == "right":
        p_value = 1 - norm.cdf(z_stat)
        z_crit = norm.ppf(1 - alpha)
    else:
        p_value = norm.cdf(z_stat)
        z_crit = -norm.ppf(1-alpha)

    reject = p_value < alpha
    return {
        "test":         "Z-Test (One Sample)",
        "z_statistic":  round(z_stat, 4),
        "z_critical":   round(z_crit, 4),
        "p_value":      round(p_value, 6),
        "alpha":        alpha,
        "reject_H0":    reject,
        "conclusion":   (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"The mean return {'IS' if reject else 'IS NOT'} "
            f"significantly different from {pop_mean:.4%}."
        ),
    }
def t_test_one_sample(data: np.ndarray, pop_mean: float, alpha: float = 0.05) -> dict:
    t_stat, p_value = ttest_1samp(data, pop_mean)
    df      = len(data) - 1
    t_crit  = t_dist.ppf(1 - alpha / 2, df)
    reject  = p_value < alpha

    return {
        "test":        "One-Sample t-Test",
        "t_statistic": round(t_stat, 4),
        "t_critical":  round(t_crit, 4),
        "p_value":     round(p_value, 6),
        "df":          df,
        "alpha":       alpha,
        "reject_H0":   reject,
        "conclusion":  (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"Mean return {'IS' if reject else 'IS NOT'} "
            f"significantly different from {pop_mean:.4%}."
        ),
    }
def t_test_two_sample(data1: np.ndarray, data2: np.ndarray,label1: str="Stock A", label2: str="Stock B", alpha: float = 0.05) -> dict:
    t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
    reject = p_value < alpha

    return {
        "test":          "Two-Sample Welch's t-Test",
        "label1":        label1,
        "label2":        label2,
        "mean1":         round(np.mean(data1), 6),
        "mean2":         round(np.mean(data2), 6),
        "t_statistic":   round(t_stat, 4),
        "p_value":       round(p_value, 6),
        "alpha":         alpha,
        "reject_H0":     reject,
        "conclusion":    (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"Returns of {label1} and {label2} are "
            f"{'SIGNIFICANTLY DIFFERENT' if reject else 'NOT significantly different'}."
        ),
    }
def t_test_paired(data1: np.ndarray, data2: np.ndarray, label1: str="Before", label2: str="After", alpha: float = 0.05) -> dict:
    n=min(len(data1), len(data2))
    t_stat, p_value = ttest_rel(data1[:n], data2[:n])
    reject = p_value < alpha

    return {
        "test":          "Paired t-Test",
        "label1":        label1,
        "label2":        label2,
        "mean_diff":     round(np.mean(data1[:n] - data2[:n]), 6),
        "t_statistic":   round(t_stat, 4),
        "p_value":       round(p_value, 6),
        "alpha":         alpha,
        "reject_H0":     reject,
        "conclusion":    (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"There {'IS' if reject else 'IS NOT'} a significant difference "
            f"between {label1} and {label2}."
        ),
    }
#Anova
def one_way_anova(*groups, labels: list = None, alpha: float = 0.05) -> dict:
    if labels is None:
        labels = [f"Group {i+1}" for i in range(len(groups))]
    
    f_stat, p_value = f_oneway(*groups)
    reject = p_value < alpha
    #Summary stats per group
    group_stats = []
    for label,grp in zip(labels, groups):
        group_stats.append({
            "sector": label,
            "n": len(grp),
            "mean": round(np.mean(grp), 6),
            "std": round(np.std(grp, ddof=1), 6),
            
        })
    #Effect size (eta squared)
    grand_mean  = np.mean(np.concatenate(groups))
    ss_between  = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total    = sum(np.sum((g - grand_mean)**2) for g in groups)
    eta_squared = ss_between / ss_total if ss_total != 0 else 0
    return {
        "test":         "One-Way ANOVA",
        "f_statistic":  round(f_stat, 4),
        "p_value":      round(p_value, 6),
        "alpha":        alpha,
        "eta_squared":  round(eta_squared, 4),
        "effect_size":  "Large" if eta_squared > 0.14 else "Medium" if eta_squared > 0.06 else "Small",
        "group_stats":  group_stats,
        "reject_H0":    reject,
        "conclusion":   (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"Sector returns {'ARE' if reject else 'ARE NOT'} significantly different."
        ),
    }
