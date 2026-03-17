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
#Chi-Square Tests

def chi_square_normality(data: np.ndarray, bins:int=10, alpha: float = 0.05) -> dict:
    observed_freq, bin_edges = np.histogram(data, bins=bins)
    mu,sigma                 =np.mean(data), np.std(data, ddof=1)
    expected= []
    for i in range(len(bin_edges)-1):
        p_low  = norm.cdf(bin_edges[i],   mu, sigma)
        p_high = norm.cdf(bin_edges[i+1], mu, sigma)
        expected.append((p_high - p_low) * len(data))
    expected = np.array(expected)

    mask     = expected >= 5
    obs_filt = observed[mask]
    exp_filt = expected[mask]

    chi2_stat = np.sum((obs_filt - exp_filt)**2 / exp_filt)
    df        = len(obs_filt) - 1 - 2          # -2 for estimated μ, σ
    df        = max(df, 1)
    p_value   = 1 - chi2.cdf(chi2_stat, df)
    reject    = p_value < alpha


    sw_stat, sw_p = shapiro(data[:5000])        # shapiro works best on <5000

    return {
        "test":         "Chi-Square Goodness of Fit (Normality)",
        "chi2_stat":    round(chi2_stat, 4),
        "p_value":      round(p_value, 6),
        "df":           df,
        "alpha":        alpha,
        "reject_H0":    reject,
        "shapiro_p":    round(sw_p, 6),
        "conclusion":   (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"Returns {'DO NOT' if reject else 'DO'} follow a Normal distribution. "
            f"(Shapiro-Wilk p = {sw_p:.4f})"
        ),
    }
def chi_square_independence(data: pd.DataFrame, col1: str, col2: str,
                            alpha: float = 0.05) -> dict:

    contingency = pd.crosstab(data[col1], data[col2])
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
    reject = p_value < alpha

    n = contingency.values.sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))

    return {
        "test": "Chi-Square Test of Independence",
        "chi2_stat": round(chi2_stat, 4),
        "p_value": round(p_value, 6),
        "df": dof,
        "alpha": alpha,
        "cramers_v": round(cramers_v, 4),
        "effect_size": "Large" if cramers_v > 0.5 else "Medium" if cramers_v > 0.3 else "Small",
        "contingency": contingency,
        "reject_H0": reject,
        "conclusion": (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"{col1} and {col2} are "
            f"{'DEPENDENT' if reject else 'INDEPENDENT'}."
        ),
    }
#Non-Parametric Tests
def mann_whitney_u_test(data1: np.ndarray, data2: np.ndarray, label1: str="Stock A", label2: str="Stock B", alpha: float = 0.05) -> dict:
    u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    reject = p_value < alpha
    # Effect size r=Z/√N
    n      = len(data1) + len(data2)
    z_val  = (u_stat - (len(data1) * len(data2) / 2)) / np.sqrt(
              len(data1) * len(data2) * (n + 1) / 12)
    r      = abs(z_val) / np.sqrt(n)
    return {
        "test":        "Mann-Whitney U Test",
        "label1":      label1,
        "label2":      label2,
        "u_statistic": round(u_stat, 4),
        "p_value":     round(p_value, 6),
        "effect_r":    round(r, 4),
        "alpha":       alpha,
        "reject_H0":   reject,
        "conclusion":  (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"Distributions of {label1} and {label2} are "
            f"{'SIGNIFICANTLY DIFFERENT' if reject else 'NOT significantly different'}."
        ),
    }

def kruskal_wallis_test(*groups, labels: list = None, alpha: float = 0.05) -> dict:
    if labels is None:
        labels = [f"Group {i+1}" for i in range(len(groups))]
    
    h_stat, p_value = kruskal(*groups)
    reject = p_value < alpha
    # Effect size: eta squared equivalent
    n       = sum(len(g) for g in groups)
    eta_sq  = (h_stat - len(groups) + 1) / (n - len(groups))
    return {
        "test":        "Kruskal-Wallis Test",
        "h_statistic": round(h_stat, 4),
        "p_value":     round(p_value, 6),
        "alpha":       alpha,
        "labels":      labels,
        "group_medians": {l: round(float(np.median(g)), 6)
                          for l, g in zip(labels, groups)},
        "eta_squared": round(max(eta_sq, 0), 4),
        "reject_H0":   reject,
        "conclusion":  (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"Sector return distributions "
            f"{'ARE' if reject else 'ARE NOT'} significantly different."
        ),
    }
def friedman_test(*groups, labels: list = None, alpha: float = 0.05) -> dict:
    if labels is None:
        labels = [f"Group {i+1}" for i in range(len(groups))]

    # Trim to equal length
    min_len = min(len(g) for g in groups)
    groups  = [g[:min_len] for g in groups]

    f_stat, p_value = friedmanchisquare(*groups)
    reject = p_value < alpha
    return {
        "test":        "Friedman Test",
        "statistic":   round(f_stat, 4),
        "p_value":     round(p_value, 6),
        "alpha":       alpha,
        "labels":      labels,
        "reject_H0":   reject,
        "conclusion":  (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"Repeated measures are "
            f"{'SIGNIFICANTLY DIFFERENT' if reject else 'NOT significantly different'}."
        ),
    }
# Bayesian Volatility Estimation
def bayesian_volatility(returns: np.ndarray, prior_mean: float = 0.015, prior_strength: float = 30) -> dict:
    n = len(returns)
    obs_var = np.var(returns, ddof=1)
    obs_std = np.sqrt(obs_var)
    posterior_std = (prior_strength * prior_mean + n * obs_std) / (prior_strength + n)

    alpha_post   = (prior_strength + n) / 2
    beta_post    = (prior_strength * prior_mean**2 + (n - 1) * obs_var) / 2

    ci_lower = np.sqrt(beta_post / chi2.ppf(0.975, 2 * alpha_post))
    ci_upper = np.sqrt(beta_post / chi2.ppf(0.025, 2 * alpha_post))

    std_grid  = np.linspace(max(0.001, posterior_std - 4 * obs_std / np.sqrt(n)),
                             posterior_std + 4 * obs_std / np.sqrt(n), 300)
    
    post_std_of_std = obs_std / np.sqrt(2 * (n - 1))
    posterior_pdf   = norm.pdf(std_grid, posterior_std, post_std_of_std)
    return {
        "prior_volatility":     round(prior_mean, 6),
        "observed_volatility":  round(obs_std, 6),
        "posterior_volatility": round(posterior_std, 6),
        "ci_lower_95":          round(ci_lower, 6),
        "ci_upper_95":          round(ci_upper, 6),
        "annualized_posterior": round(posterior_std * np.sqrt(252), 4),
        "std_grid":             std_grid,
        "posterior_pdf":        posterior_pdf,
        "interpretation": (
            f"After observing {n} days of returns, our belief about "
            f"daily volatility updated from {prior_mean*100:.2f}% (prior) "
            f"to {posterior_std*100:.2f}% (posterior). "
            f"95% Credible Interval: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]"
        ),
    }
#Full Sector Comparison Report

def full_sector_report(sector_returns: dict, alpha: float = 0.05) -> dict:
    labels = list(sector_returns.keys())
    groups = [np.array(sector_returns[l].dropna()) for l in labels]

    report = {}

    report["descriptive"] = {
        l: {
            "mean": round(float(np.mean(g)), 6),
            "std": round(float(np.std(g, ddof=1)), 6),
            "skew": round(float(pd.Series(g).skew()), 4),
            "kurtosis": round(float(pd.Series(g).kurtosis()), 4),
            "n": len(g),
        }
        for l, g in zip(labels, groups)
    }

    report["normality"] = {
        l: chi_square_normality(g, alpha=alpha)
        for l, g in zip(labels, groups)
    }

    report["anova"] = one_way_anova(*groups, labels=labels, alpha=alpha)

    report["kruskal_wallis"] = kruskal_wallis_test(*groups, labels=labels, alpha=alpha)

    from itertools import combinations
    report["pairwise_t"] = {}
    for (l1, g1), (l2, g2) in combinations(zip(labels, groups), 2):
        key = f"{l1} vs {l2}"
        report["pairwise_t"][key] = t_test_two_sample(g1, g2, l1, l2, alpha)

    report["bayesian_vol"] = {
        l: bayesian_volatility(g)
        for l, g in zip(labels, groups)
    }

    return report
