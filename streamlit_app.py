import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_loader   import (
    get_price_data, get_daily_returns, get_log_returns,
    get_sector_returns, get_stock_info, engineer_features,
    SECTORS, ALL_TICKERS,
)
from abish_stats   import (
    confidence_interval_mean,
    confidence_interval_volatility,
    mle_normal,
    plot_likelihood_surface,
    z_test_mean,
    t_test_one_sample,
    t_test_two_sample,
    one_way_anova,
    chi_square_normality,
    kruskal_wallis_test,
    friedman_test,
    bayesian_volatility,
    full_sector_report,
)
from laksh_ml      import (
    train_logistic_model,
    train_ridge_model,
    train_lasso_model,
    compare_ridge_lasso,
    predict_tomorrow,
    glm_logistic_summary,
    FEATURE_COLS,
)

st.set_page_config(
    page_title="Equity Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #00d4aa; }
    .metric-label { font-size: 0.8rem; color: #8892a4; margin-top: 4px; }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        border-left: 4px solid #00d4aa;
        padding-left: 12px;
        margin: 24px 0 12px 0;
    }
    .result-box {
        background: #1a1f2e;
        border-radius: 8px;
        padding: 14px 18px;
        border: 1px solid #2d3250;
        margin: 8px 0;
    }
    .reject { color: #ff6b6b; font-weight: 600; }
    .accept { color: #51cf66; font-weight: 600; }
    .tag-selected {
        background: #00d4aa22;
        color: #00d4aa;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
    }
    .tag-zeroed {
        background: #ff6b6b22;
        color: #ff6b6b;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
    }
    div[data-testid="stMetricValue"] { color: #00d4aa !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=60)
    st.title("Equity Intelligence")
    st.caption("Statistical & ML Analysis Dashboard")
    st.divider()

    st.subheader("⚙️ Settings")
    selected_sector = st.selectbox("Primary Sector", list(SECTORS.keys()))
    ticker1 = st.selectbox("Stock 1", SECTORS[selected_sector])

    other_sectors = [s for s in SECTORS.keys() if s != selected_sector]
    compare_sector = st.selectbox("Compare Sector", other_sectors)
    ticker2 = st.selectbox("Stock 2", SECTORS[compare_sector])

    period  = st.select_slider("Data Period", ["6mo", "1y", "2y", "3y", "5y"], value="2y")
    alpha   = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)

    st.divider()
    st.caption("Built by **Abish** (Stats) & **Laksh** (ML)\nMPSTME NMIMS — SSDI Project")

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(t1, t2, period):
    prices1 = get_price_data([t1], period)
    prices2 = get_price_data([t2], period)
    ret1    = get_daily_returns(prices1)[t1].dropna()
    ret2    = get_daily_returns(prices2)[t2].dropna()
    log1    = get_log_returns(prices1)[t1].dropna()
    sector_rets = get_sector_returns(period)
    info1   = get_stock_info(t1)
    info2   = get_stock_info(t2)
    df1     = engineer_features(t1, period)
    return prices1, prices2, ret1, ret2, log1, sector_rets, info1, info2, df1

with st.spinner("Fetching live NSE data..."):
    try:
        prices1, prices2, ret1, ret2, log1, \
        sector_rets, info1, info2, df_ml = load_data(ticker1, ticker2, period)
        data_ok = True
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        data_ok = False

if not data_ok:
    st.stop()

st.title("📊 Equity Intelligence Dashboard")
st.caption(f"Live analysis for **{ticker1}** vs **{ticker2}** | Period: {period}")
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview",
    "📐 Risk Profile  (Abish)",
    "🧪 Hypothesis Tests  (Abish)",
    "🌐 Sector Analysis  (Abish)",
    "🤖 ML Predictions  (Laksh)",
    "📉 Ridge & Lasso  (Laksh)",
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)

    def safe_get(d, k): return d.get(k, "N/A")

    with c1:
        st.metric("Company",     safe_get(info1, "name")[:20])
    with c2:
        st.metric("Avg Daily Return",  f"{ret1.mean()*100:.3f}%")
    with c3:
        st.metric("Daily Volatility",  f"{ret1.std()*100:.2f}%")
    with c4:
        ann = ret1.mean() * 252
        st.metric("Annualised Return", f"{ann*100:.1f}%")

    st.divider()


    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f'<div class="section-header">Price History — {ticker1}</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prices1.index, y=prices1[ticker1],
            mode="lines", name=ticker1,
            line=dict(color="#00d4aa", width=1.5)
        ))
        fig.update_layout(template="plotly_dark", height=300,
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown(f'<div class="section-header">Price History — {ticker2}</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=prices2.index, y=prices2[ticker2],
            mode="lines", name=ticker2,
            line=dict(color="#f7971e", width=1.5)
        ))
        fig2.update_layout(template="plotly_dark", height=300,
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)


    st.markdown('<div class="section-header">Return Distribution Comparison</div>',
                unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=ret1, name=ticker1, opacity=0.7,
        marker_color="#00d4aa", nbinsx=60
    ))
    fig3.add_trace(go.Histogram(
        x=ret2, name=ticker2, opacity=0.7,
        marker_color="#f7971e", nbinsx=60
    ))
    fig3.update_layout(
        barmode="overlay", template="plotly_dark",
        xaxis_title="Daily Return", yaxis_title="Frequency",
        height=350, margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown("### 📐 Risk Profile — Abish's Statistical Analysis")

    col1, col2 = st.columns(2)


    with col1:
        st.markdown('<div class="section-header">Confidence Interval — Mean Return</div>',
                    unsafe_allow_html=True)
        ci  = confidence_interval_mean(ret1.values, confidence=1 - alpha)
        civ = confidence_interval_volatility(ret1.values, confidence=1 - alpha)

        st.markdown(f"""
        <div class="result-box">
            <b>Estimated Mean Daily Return:</b> {ci['mean']*100:.4f}%<br>
            <b>{int((1-alpha)*100)}% CI:</b> [{ci['lower']*100:.4f}%, {ci['upper']*100:.4f}%]<br>
            <b>Std Error:</b> {ci['std_error']*100:.5f}%<br>
            <b>Method:</b> {ci['method']}<br>
            <b>n:</b> {ci['n']} observations
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Confidence Interval — Volatility</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-box">
            <b>Sample Std Dev:</b> {civ['sample_std']*100:.3f}%<br>
            <b>{int((1-alpha)*100)}% CI for σ:</b> [{civ['std_lower']*100:.3f}%, {civ['std_upper']*100:.3f}%]<br>
            <small>Based on Chi-Square distribution</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Maximum Likelihood Estimation</div>',
                    unsafe_allow_html=True)
        mle = mle_normal(ret1.values)

        st.markdown(f"""
        <div class="result-box">
            <b>MLE μ̂:</b> {mle['mu_mle']*100:.4f}%<br>
            <b>MLE σ̂:</b> {mle['sigma_mle']*100:.4f}%<br>
            <b>Log-Likelihood:</b> {mle['log_likelihood']}<br>
            <b>AIC:</b> {mle['aic']}<br>
            <br><i>{mle['interpretation']}</i>
        </div>
        """, unsafe_allow_html=True)

        ll_data = plot_likelihood_surface(ret1.values)
        fig_ll  = go.Figure()
        fig_ll.add_trace(go.Scatter(
            x=ll_data["mu_grid"] * 100,
            y=ll_data["log_likelihoods"],
            mode="lines", line=dict(color="#00d4aa")
        ))
        fig_ll.add_vline(
            x=ll_data["mle_mu"] * 100,
            line_color="#f7971e", line_dash="dash",
            annotation_text="MLE"
        )
        fig_ll.update_layout(
            template="plotly_dark",
            xaxis_title="μ (%)", yaxis_title="Log-Likelihood",
            title="Likelihood Surface", height=280,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_ll, use_container_width=True)

    st.markdown('<div class="section-header">Bayesian Volatility Estimation</div>',
                unsafe_allow_html=True)

    prior_vol = st.slider("Prior Belief on Daily Volatility (%)", 0.5, 4.0, 1.5, 0.1)
    bv        = bayesian_volatility(ret1.values, prior_mean=prior_vol / 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prior Volatility",      f"{bv['prior_volatility']*100:.2f}%")
    c2.metric("Observed Volatility",   f"{bv['observed_volatility']*100:.2f}%")
    c3.metric("Posterior Volatility",  f"{bv['posterior_volatility']*100:.2f}%")
    c4.metric("Annualised (Posterior)",f"{bv['annualized_posterior']*100:.1f}%")

    st.markdown(f'<div class="result-box">{bv["interpretation"]}</div>',
                unsafe_allow_html=True)

    fig_bay = go.Figure()
    fig_bay.add_trace(go.Scatter(
        x=bv["std_grid"] * 100,
        y=bv["posterior_pdf"],
        mode="lines", fill="tozeroy",
        line=dict(color="#00d4aa"),
        name="Posterior Distribution"
    ))
    fig_bay.add_vline(x=bv["posterior_volatility"] * 100,
                      line_color="#f7971e", line_dash="dash",
                      annotation_text="Posterior Mean")
    fig_bay.update_layout(
        template="plotly_dark",
        xaxis_title="Daily Volatility (%)",
        yaxis_title="Posterior Density",
        height=300, margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_bay, use_container_width=True)



with tab3:
    st.markdown("### 🧪 Hypothesis Testing Suite — Abish")

    test_type = st.radio(
        "Select Test",
        ["Z-Test vs Benchmark", "One-Sample t-Test",
         "Two-Sample t-Test", "Normality (Chi-Square)",
         "Paired t-Test"],
        horizontal=True,
    )

    st.divider()

    def result_badge(reject):
        if reject:
            return '<span class="reject">❌ REJECT H₀</span>'
        return '<span class="accept">✅ FAIL TO REJECT H₀</span>'

    if test_type == "Z-Test vs Benchmark":
        bench = st.number_input("Benchmark Daily Return (%)", value=0.05, step=0.01) / 100
        pop_std = st.number_input("Known Population Std (%)", value=1.0, step=0.1) / 100
        res = z_test_mean(ret1.values, bench, pop_std, alpha=alpha)
        col1, col2, col3 = st.columns(3)
        col1.metric("Z-Statistic",  res["z_statistic"])
        col2.metric("Z-Critical",   f"±{res['z_critical']}")
        col3.metric("p-value",      res["p_value"])
        st.markdown(f"""<div class="result-box">
            {result_badge(res['reject_H0'])}<br><br>
            <b>H₀:</b> μ = {bench*100:.3f}%<br>
            <b>H₁:</b> μ ≠ {bench*100:.3f}%<br><br>
            {res['conclusion']}
        </div>""", unsafe_allow_html=True)

    elif test_type == "One-Sample t-Test":
        null_mean = st.number_input("Null Hypothesis Mean (%)", value=0.0, step=0.01) / 100
        res = t_test_one_sample(ret1.values, null_mean, alpha=alpha)
        col1, col2, col3 = st.columns(3)
        col1.metric("t-Statistic", res["t_statistic"])
        col2.metric("t-Critical",  f"±{res['t_critical']}")
        col3.metric("p-value",     res["p_value"])
        st.markdown(f"""<div class="result-box">
            {result_badge(res['reject_H0'])}<br><br>
            {res['conclusion']}
        </div>""", unsafe_allow_html=True)

    elif test_type == "Two-Sample t-Test":
        res = t_test_two_sample(ret1.values, ret2.values,
                                ticker1, ticker2, alpha=alpha)
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Mean {ticker1}", f"{res['mean1']*100:.4f}%")
        col2.metric(f"Mean {ticker2}", f"{res['mean2']*100:.4f}%")
        col3.metric("p-value",         res["p_value"])
        st.markdown(f"""<div class="result-box">
            {result_badge(res['reject_H0'])}<br><br>
            <b>Test:</b> {res['test']}<br>
            <b>t-Statistic:</b> {res['t_statistic']}<br><br>
            {res['conclusion']}
        </div>""", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Box(y=ret1.values * 100, name=ticker1,
                              marker_color="#00d4aa"))
        fig.add_trace(go.Box(y=ret2.values * 100, name=ticker2,
                              marker_color="#f7971e"))
        fig.update_layout(
            template="plotly_dark",
            yaxis_title="Daily Return (%)",
            height=350, margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Normality (Chi-Square)":
        res = chi_square_normality(ret1.values, alpha=alpha)
        col1, col2, col3 = st.columns(3)
        col1.metric("χ² Statistic",  res["chi2_stat"])
        col2.metric("p-value",        res["p_value"])
        col3.metric("Shapiro-Wilk p", res["shapiro_p"])
        st.markdown(f"""<div class="result-box">
            {result_badge(res['reject_H0'])}<br><br>
            {res['conclusion']}
        </div>""", unsafe_allow_html=True)

        from scipy.stats import probplot
        theoretical_q, sample_q = zip(*probplot(ret1.values)[0])
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical_q, y=sample_q,
            mode="markers", marker=dict(color="#00d4aa", size=4), name="QQ"
        ))
        lim = max(abs(min(theoretical_q)), abs(max(theoretical_q)))
        fig_qq.add_trace(go.Scatter(
            x=[-lim, lim], y=[-lim, lim],
            mode="lines", line=dict(color="#f7971e", dash="dash"), name="Normal Line"
        ))
        fig_qq.update_layout(
            template="plotly_dark", title="Q-Q Plot (Normal)",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=350, margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    elif test_type == "Paired t-Test":
        n    = min(len(ret1), len(ret2))
        mid  = n // 2
        res  = t_test_two_sample(ret1.values[:mid], ret1.values[mid:n],
                                   "First Half", "Second Half", alpha)
        col1, col2 = st.columns(2)
        col1.metric("t-Statistic", res["t_statistic"])
        col2.metric("p-value",     res["p_value"])
        st.markdown(f"""<div class="result-box">
            {result_badge(res['reject_H0'])}<br><br>
            Comparing <b>first half</b> vs <b>second half</b> of {ticker1} returns.<br><br>
            {res['conclusion']}
        </div>""", unsafe_allow_html=True)

with tab4:
    st.markdown("### 🌐 Cross-Sector Statistical Analysis — Abish")

    with st.spinner("Running sector-wide statistical tests..."):
        labels  = list(sector_rets.keys())
        groups  = [np.array(sector_rets[l].dropna()) for l in labels]

        anova_res  = one_way_anova(*groups, labels=labels, alpha=alpha)
        kw_res     = kruskal_wallis_test(*groups, labels=labels, alpha=alpha)

    st.markdown('<div class="section-header">Sector Return Distributions</div>',
                unsafe_allow_html=True)
    colors = ["#00d4aa", "#f7971e", "#a855f7", "#3b82f6", "#ef4444"]
    fig_sec = go.Figure()
    for (label, grp), color in zip(zip(labels, groups), colors):
        fig_sec.add_trace(go.Violin(
            y=grp * 100, name=label,
            box_visible=True, meanline_visible=True,
            marker_color=color, opacity=0.8
        ))
    fig_sec.update_layout(
        template="plotly_dark", yaxis_title="Daily Return (%)",
        height=400, margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_sec, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">One-Way ANOVA</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""<div class="result-box">
            <b>F-Statistic:</b> {anova_res['f_statistic']}<br>
            <b>p-value:</b> {anova_res['p_value']}<br>
            <b>Effect Size (η²):</b> {anova_res['eta_squared']} ({anova_res['effect_size']})<br><br>
            {result_badge(anova_res['reject_H0'])}<br><br>
            {anova_res['conclusion']}
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Kruskal-Wallis (Nonparametric)</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""<div class="result-box">
            <b>H-Statistic:</b> {kw_res['h_statistic']}<br>
            <b>p-value:</b> {kw_res['p_value']}<br>
            <b>Group Medians:</b><br>
            {"<br>".join(f"  {k}: {v*100:.4f}%" for k,v in kw_res['group_medians'].items())}
            <br><br>
            {result_badge(kw_res['reject_H0'])}<br>
            {kw_res['conclusion']}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Descriptive Statistics by Sector</div>',
                unsafe_allow_html=True)
    desc_data = []
    for label, grp in zip(labels, groups):
        desc_data.append({
            "Sector":        label,
            "Mean (%)":      round(np.mean(grp) * 100, 4),
            "Std Dev (%)":   round(np.std(grp, ddof=1) * 100, 4),
            "Skewness":      round(float(pd.Series(grp).skew()), 4),
            "Kurtosis":      round(float(pd.Series(grp).kurtosis()), 4),
            "Observations":  len(grp),
        })
    st.dataframe(pd.DataFrame(desc_data), use_container_width=True)

with tab5:
    st.markdown("### 🤖 ML Direction Prediction — Laksh")

    if len(df_ml) < 100:
        st.warning("Not enough data for ML training. Try a longer period.")
    else:
        with st.spinner("Training Logistic Regression model..."):
            log_res = train_logistic_model(df_ml)

        st.markdown('<div class="section-header">Tomorrow\'s Signal</div>',
                    unsafe_allow_html=True)
        pred      = log_res["tomorrow"] if "tomorrow" in log_res else \
                    predict_tomorrow(log_res, df_ml.iloc[-1])
        
        pred = predict_tomorrow(log_res, df_ml.iloc[-1])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal",       pred["direction"])
        c2.metric("P(Up)",        f"{pred['prob_up']*100:.1f}%")
        c3.metric("P(Down)",      f"{pred['prob_down']*100:.1f}%")
        c4.metric("Confidence",   f"{pred['confidence']}%")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            # Model metrics
            st.markdown('<div class="section-header">Model Performance</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""<div class="result-box">
                <b>Test Accuracy:</b> {log_res['accuracy']*100:.1f}%<br>
                <b>ROC-AUC:</b> {log_res['roc_auc']}<br>
                <b>CV Accuracy:</b> {log_res['cv_accuracy_mean']*100:.1f}%
                ± {log_res['cv_accuracy_std']*100:.1f}%<br>
                <b>CV AUC:</b> {log_res['cv_auc_mean']:.3f}<br>
                <br><i>{log_res['interpretation']}</i>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">ROC Curve</div>',
                        unsafe_allow_html=True)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=log_res["roc_fpr"], y=log_res["roc_tpr"],
                mode="lines", name=f"ROC (AUC={log_res['roc_auc']})",
                line=dict(color="#00d4aa", width=2)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="#888", dash="dash"), name="Random"
            ))
            fig_roc.update_layout(
                template="plotly_dark",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=300, margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Confusion Matrix</div>',
                        unsafe_allow_html=True)
            cm = log_res["confusion_matrix"]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual",
                            color="Count"),
                x=["Down", "Up"], y=["Down", "Up"],
                color_continuous_scale="teal",
                text_auto=True,
            )
            fig_cm.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown('<div class="section-header">Feature Coefficients (Log-Odds)</div>',
                        unsafe_allow_html=True)
            coef_df = log_res["coef_df"]
            colors  = ["#00d4aa" if c > 0 else "#ff6b6b"
                       for c in coef_df["coefficient"]]
            fig_coef = go.Figure(go.Bar(
                x=coef_df["coefficient"],
                y=coef_df["feature"],
                orientation="h",
                marker_color=colors,
            ))
            fig_coef.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Coefficient (Log-Odds)"
            )
            st.plotly_chart(fig_coef, use_container_width=True)


with tab6:
    st.markdown("### 📉 Ridge & Lasso Regression — Laksh")

    with st.spinner("Training Ridge and Lasso models..."):
        ridge_res = train_ridge_model(df_ml)
        lasso_res = train_lasso_model(df_ml)
        comp_df   = compare_ridge_lasso(ridge_res, lasso_res)

    st.markdown('<div class="section-header">Model Comparison</div>',
                unsafe_allow_html=True)
    st.dataframe(comp_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Ridge Coefficients</div>',
                    unsafe_allow_html=True)
        rc = ridge_res["coef_df"]
        fig_r = go.Figure(go.Bar(
            x=rc["coefficient"],
            y=rc["feature"],
            orientation="h",
            marker_color=["#00d4aa" if c > 0 else "#ff6b6b"
                          for c in rc["coefficient"]]
        ))
        fig_r.update_layout(
            template="plotly_dark", height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=f"Coefficient (α={ridge_res['best_alpha']})"
        )
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown(f'<div class="result-box"><b>R²:</b> {ridge_res["r2_score"]} &nbsp;|&nbsp; '
                    f'<b>RMSE:</b> {ridge_res["rmse"]*100:.4f}%<br>'
                    f'{ridge_res["interpretation"]}</div>',
                    unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Lasso Feature Selection</div>',
                    unsafe_allow_html=True)
        lc = lasso_res["coef_df"]

        fig_l = go.Figure(go.Bar(
            x=lc["coefficient"],
            y=lc["feature"],
            orientation="h",
            marker_color=["#00d4aa" if s else "#444"
                          for s in lc["selected"]]
        ))
        fig_l.update_layout(
            template="plotly_dark", height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=f"Coefficient (α={lasso_res['best_alpha']:.5f})"
        )
        st.plotly_chart(fig_l, use_container_width=True)

        n_sel = lasso_res["n_selected"]
        n_zer = lasso_res["n_zeroed"]
        st.markdown(f"""<div class="result-box">
            <span class="tag-selected">✓ {n_sel} Selected</span> &nbsp;
            <span class="tag-zeroed">✗ {n_zer} Zeroed Out</span><br><br>
            <b>Kept:</b> {", ".join(lasso_res['selected_features'])}<br><br>
            {lasso_res['interpretation']}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Lasso Regularization Path</div>',
                unsafe_allow_html=True)
    fig_path = go.Figure()
    colors_path = px.colors.qualitative.Plotly
    for i, feat in enumerate(FEATURE_COLS):
        fig_path.add_trace(go.Scatter(
            x=np.log10(lasso_res["alpha_path"]),
            y=lasso_res["coef_path"][:, i],
            mode="lines",
            name=feat,
            line=dict(color=colors_path[i % len(colors_path)], width=1.5)
        ))
    fig_path.add_vline(
        x=np.log10(lasso_res["best_alpha"]),
        line_color="white", line_dash="dash",
        annotation_text="Best α (CV)"
    )
    fig_path.update_layout(
        template="plotly_dark",
        xaxis_title="log₁₀(α)",
        yaxis_title="Coefficient Value",
        title="How features get zeroed as regularization increases",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_path, use_container_width=True)