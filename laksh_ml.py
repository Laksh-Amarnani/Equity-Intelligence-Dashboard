import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import (
    LogisticRegression,
    Ridge, Lasso,
    RidgeCV, LassoCV
)
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection   import (
    train_test_split, cross_val_score,
    StratifiedKFold, TimeSeriesSplit
)
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.pipeline import Pipeline
import statsmodels.api as sm


FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_5",
    "rolling_mean_5", "rolling_std_5",
    "rolling_mean_20", "rolling_std_20",
    "rsi", "volume_ratio",
    "dist_ma20", "dist_ma50",
]


def _split_features(df: pd.DataFrame):
    """Extract X, y from engineered feature DataFrame."""
    X = df[FEATURE_COLS].values
    y = df["target"].values
    return X, y


def train_logistic_model(df: pd.DataFrame,
                          test_size: float = 0.2,
                          random_state: int = 42) -> dict:
    """
    Train Logistic Regression to predict next-day direction (Up / Down).

    Uses TimeSeriesSplit for cross-validation (prevents look-ahead bias).

    Business use: "Given today's price, RSI, and volume — will this stock
                   go up or down tomorrow?" (like a signal generation engine)
    """
    X, y = _split_features(df)

    # Time-series aware split (NO random shuffle for financial data!)
    split    = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Pipeline: scale → logistic regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",    # handles class imbalance
            solver="lbfgs",
            random_state=random_state,
        ))
    ])

    pipe.fit(X_train, y_train)

    # Predictions
    y_pred       = pipe.predict(X_test)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]

    # Cross-validation (time-series splits)
    tscv   = TimeSeriesSplit(n_splits=5)
    cv_acc = cross_val_score(pipe, X, y, cv=tscv, scoring="accuracy")
    cv_auc = cross_val_score(pipe, X, y, cv=tscv, scoring="roc_auc")

    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Coefficients (after scaling — from the logistic model)
    coefs = pipe.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "coefficient": coefs,
        "abs_coef":   np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    # Log-odds interpretation
    odds_df = coef_df.copy()
    odds_df["odds_ratio"] = np.exp(odds_df["coefficient"])

    return {
        "model":           pipe,
        "accuracy":        round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":         round(roc_auc_score(y_test, y_pred_proba), 4),
        "cv_accuracy_mean": round(cv_acc.mean(), 4),
        "cv_accuracy_std":  round(cv_acc.std(), 4),
        "cv_auc_mean":      round(cv_auc.mean(), 4),
        "report":           classification_report(y_test, y_pred,
                                                   target_names=["Down", "Up"],
                                                   output_dict=True),
        "confusion_matrix": cm,
        "roc_fpr":          fpr,
        "roc_tpr":          tpr,
        "y_test":           y_test,
        "y_pred":           y_pred,
        "y_pred_proba":     y_pred_proba,
        "coef_df":          coef_df,
        "odds_df":          odds_df,
        "feature_cols":     FEATURE_COLS,
        "X_test":           X_test,
        "dates_test":       df.index[split:],
        "interpretation": (
            f"The model achieves {accuracy_score(y_test, y_pred)*100:.1f}% accuracy "
            f"and AUC = {roc_auc_score(y_test, y_pred_proba):.3f}. "
            f"Top signal: {coef_df.iloc[0]['feature']} "
            f"(coef = {coef_df.iloc[0]['coefficient']:.4f})"
        ),
    }


def predict_tomorrow(model_result: dict, latest_row: pd.Series) -> dict:
    """
    Predict tomorrow's direction for a single latest observation.
    Returns probability and direction label.
    """
    pipe   = model_result["model"]
    x      = latest_row[FEATURE_COLS].values.reshape(1, -1)
    prob   = pipe.predict_proba(x)[0]
    label  = "UP 📈" if prob[1] > 0.5 else "DOWN 📉"

    return {
        "direction":    label,
        "prob_up":      round(prob[1], 4),
        "prob_down":    round(prob[0], 4),
        "confidence":   round(max(prob) * 100, 1),
    }


def train_ridge_model(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Ridge Regression to predict the magnitude of next-day return.
    Adds L2 regularization to handle multicollinearity among features.

    Business use: "Not just direction — how much might this stock move tomorrow?"
    Coefficient plot shows which features matter after regularization.
    """
    X = df[FEATURE_COLS].values
    y = df["return"].shift(-1).dropna().values   # actual next-day return
    X = X[:len(y)]                               # align lengths

    split    = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Auto-select best alpha (regularization strength) via cross-validation
    alphas    = np.logspace(-3, 4, 100)
    ridge_cv  = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5))
    ridge_cv.fit(X_train_s, y_train)

    best_alpha = ridge_cv.alpha_
    ridge      = Ridge(alpha=best_alpha)
    ridge.fit(X_train_s, y_train)

    y_pred  = ridge.predict(X_test_s)
    mse     = mean_squared_error(y_test, y_pred)
    rmse    = np.sqrt(mse)
    r2      = r2_score(y_test, y_pred)
    mae     = mean_absolute_error(y_test, y_pred)

    # Coefficient path over alphas (for visualization)
    coef_path = []
    for a in alphas:
        r = Ridge(alpha=a)
        r.fit(X_train_s, y_train)
        coef_path.append(r.coef_)
    coef_path = np.array(coef_path)

    coef_df = pd.DataFrame({
        "feature":     FEATURE_COLS,
        "coefficient": ridge.coef_,
        "abs_coef":    np.abs(ridge.coef_),
    }).sort_values("abs_coef", ascending=False)

    return {
        "model":        ridge,
        "scaler":       scaler,
        "best_alpha":   round(best_alpha, 4),
        "r2_score":     round(r2, 4),
        "rmse":         round(rmse, 6),
        "mae":          round(mae, 6),
        "coef_df":      coef_df,
        "coef_path":    coef_path,
        "alpha_path":   alphas,
        "y_test":       y_test,
        "y_pred":       y_pred,
        "dates_test":   df.index[split:split + len(y_test)],
        "interpretation": (
            f"Ridge (α={best_alpha:.3f}) explains {r2*100:.1f}% of return variance. "
            f"RMSE = {rmse*100:.3f}%. "
            f"Top feature: {coef_df.iloc[0]['feature']}."
        ),
    }


def train_lasso_model(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Lasso Regression — L1 regularization drives weak feature coefficients to 0.
    Acts as automatic feature selection.

    Business use: "Out of 12 technical indicators, which ones actually
                   have predictive power for tomorrow's return?"
    """
    X = df[FEATURE_COLS].values
    y = df["return"].shift(-1).dropna().values
    X = X[:len(y)]

    split    = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Auto-select best alpha via CV
    lasso_cv  = LassoCV(
        alphas=np.logspace(-6, 2, 100),
        cv=TimeSeriesSplit(n_splits=5),
        max_iter=5000,
    )
    lasso_cv.fit(X_train_s, y_train)

    best_alpha = lasso_cv.alpha_
    lasso      = Lasso(alpha=best_alpha, max_iter=5000)
    lasso.fit(X_train_s, y_train)

    y_pred = lasso.predict(X_test_s)
    r2     = r2_score(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

    # Selected vs zeroed-out features
    coef_df = pd.DataFrame({
        "feature":     FEATURE_COLS,
        "coefficient": lasso.coef_,
        "abs_coef":    np.abs(lasso.coef_),
        "selected":    lasso.coef_ != 0,
    }).sort_values("abs_coef", ascending=False)

    n_selected = (lasso.coef_ != 0).sum()
    n_zeroed   = (lasso.coef_ == 0).sum()

    # Regularization path (alpha vs coefficients)
    alphas_path = np.logspace(-6, 2, 80)
    coef_path   = []
    for a in alphas_path:
        l = Lasso(alpha=a, max_iter=5000)
        l.fit(X_train_s, y_train)
        coef_path.append(l.coef_)
    coef_path = np.array(coef_path)

    return {
        "model":        lasso,
        "scaler":       scaler,
        "best_alpha":   round(best_alpha, 6),
        "r2_score":     round(r2, 4),
        "rmse":         round(rmse, 6),
        "coef_df":      coef_df,
        "coef_path":    coef_path,
        "alpha_path":   alphas_path,
        "n_selected":   int(n_selected),
        "n_zeroed":     int(n_zeroed),
        "selected_features": list(coef_df[coef_df["selected"]]["feature"]),
        "y_test":       y_test,
        "y_pred":       y_pred,
        "dates_test":   df.index[split:split + len(y_test)],
        "interpretation": (
            f"Lasso (α={best_alpha:.5f}) selected {n_selected} out of "
            f"{len(FEATURE_COLS)} features, zeroing out {n_zeroed}. "
            f"R² = {r2*100:.1f}%. "
            f"Key features: {', '.join(list(coef_df[coef_df['selected']]['feature'])[:3])}"
        ),
    }



def compare_ridge_lasso(ridge_result: dict, lasso_result: dict) -> pd.DataFrame:
    """
    Side-by-side comparison DataFrame of Ridge and Lasso performance.
    Used for the dashboard comparison table.
    """
    data = {
        "Metric":  ["R² Score", "RMSE", "Best Alpha", "Features Used"],
        "Ridge":   [
            ridge_result["r2_score"],
            ridge_result["rmse"],
            ridge_result["best_alpha"],
            len(FEATURE_COLS),
        ],
        "Lasso":   [
            lasso_result["r2_score"],
            lasso_result["rmse"],
            lasso_result["best_alpha"],
            lasso_result["n_selected"],
        ],
    }
    return pd.DataFrame(data)


def glm_logistic_summary(df: pd.DataFrame) -> object:
    """
    Fit Logistic Regression using Statsmodels for full academic summary.
    Returns a statsmodels summary object — shows p-values, odds ratios,
    confidence intervals, AIC, BIC etc.

    Business use: Present to stakeholders with full statistical transparency.
    """
    X = df[FEATURE_COLS].values
    y = df["target"].values

    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X)
    X_sm    = sm.add_constant(X_s)

    model   = sm.Logit(y, X_sm)
    result  = model.fit(method="newton", maxiter=200, disp=False)

    return result


def glm_ridge_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    OLS summary using Statsmodels (Ridge equivalent with regularization info).
    Returns coefficient table with p-values for academic reporting.
    """
    X = df[FEATURE_COLS].values
    y = df["return"].shift(-1).dropna().values
    X = X[:len(y)]

    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X)
    X_sm    = sm.add_constant(X_s)

    model   = sm.OLS(y, X_sm)
    result  = model.fit()

    # Build readable summary table
    summary = pd.DataFrame({
        "Feature":    ["const"] + FEATURE_COLS,
        "Coef":       result.params,
        "Std Error":  result.bse,
        "t-stat":     result.tvalues,
        "p-value":    result.pvalues,
        "Significant": result.pvalues < 0.05,
    })
    return summary


def full_ml_report(df: pd.DataFrame) -> dict:
    """
    Run the entire ML pipeline on a stock's feature DataFrame.
    Returns all model results in one dict — used by the dashboard.
    """
    report = {}

    print("  → Training Logistic Regression...")
    report["logistic"]  = train_logistic_model(df)

    print("  → Training Ridge Regression...")
    report["ridge"]     = train_ridge_model(df)

    print("  → Training Lasso Regression...")
    report["lasso"]     = train_lasso_model(df)

    report["comparison"] = compare_ridge_lasso(report["ridge"], report["lasso"])

    print("  → Fitting Statsmodels GLM for summary...")
    report["glm_summary"] = glm_logistic_summary(df)

    # Latest-row prediction
    latest = df.iloc[-1]
    report["tomorrow"]   = predict_tomorrow(report["logistic"], latest)

    return report