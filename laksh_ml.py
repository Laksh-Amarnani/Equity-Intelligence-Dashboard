import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import (
    LogisticRegression,
    Ridge, Lasso,
    RidgeCV, LassoCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
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
    X = df[FEATURE_COLS].values
    y = df["target"].values
    return X, y

# Logistic Regression - Direction Prediction

def train_logistic_regression(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    X, y =_split_features(df)

    split   = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]

    tscv = TimeseriesSplit(n_splits=5)
    cv_acc = cross_val_score(pipe, X, y, cv=tscv, scoring="accuracy")
    cv_auc = cross_val_score(pipe, X, y, cv=tscv, scoring="roc_auc")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    cm = confusion_matrix(y_test, y_pred)

    coefs = pipe.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "coefficient": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

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
    pipe = model_result["model"]
    x = latest_row[FEATURE_COLS].values.reshape(1, -1)
    prob = pipe.predict_proba(x)[0]
    label  = "UP 📈" if prob[1] > 0.5 else "DOWN 📉"

    return {
        "direction":    label,
        "prob_up":      round(prob[1], 4),
        "prob_down":    round(prob[0], 4),
        "confidence":   round(max(prob) * 100, 1),
    }

def train_ridge_model(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    X = df[FEATURE_COLS].values
    y = df["return"].shift(-1).dropna().values
    X = X[:len(y)]

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    alphas    = np.logspace(-3, 4, 100)
    ridge_cv  = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5))
    ridge_cv.fit(X_train_s, y_train)

    best_alpha = ridge_cv.alpha_
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train_s, y_train)

    y_pred = ridge.predict(X_test_s)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

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