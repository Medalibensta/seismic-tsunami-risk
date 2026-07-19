"""
Magnitude regression.

Predicts earthquake magnitude from depth, geographic coordinates and the
available intensity / network-quality measures. Two model families are
compared:

    * Linear regression   — interpretable baseline
    * Gradient boosting    — non-linear reference

Magnitude is only weakly determined by location and depth alone (it is a
property of the rupture, not of where it happens); most of the usable signal
comes from the intensity fields (mmi, cdi, felt), which are sparse in the older
catalogue. Results are therefore modest and reported honestly with RMSE / MAE /
R2 — see the README limits section.

Run:
    python src/regression.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import (PROCESSED_CSV, RANDOM_STATE, build_features, clean,
                           load_raw, regression_matrix)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def _load_matrix() -> tuple[pd.DataFrame, pd.Series]:
    if PROCESSED_CSV.exists():
        df = pd.read_csv(PROCESSED_CSV)
    else:
        df = build_features(clean(load_raw()))
    return regression_matrix(df)


def build_models() -> dict:
    return {
        "LinearRegression": Pipeline([
            ("scale", StandardScaler()),
            ("reg", LinearRegression()),
        ]),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE),
    }


def evaluate(name, model, X_tr, y_tr, X_te, y_te) -> tuple[dict, np.ndarray]:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "model": name,
        "RMSE": np.sqrt(mean_squared_error(y_te, pred)),
        "MAE": mean_absolute_error(y_te, pred),
        "R2": r2_score(y_te, pred),
    }
    return metrics, pred


def plot_pred_vs_actual(y_te, pred, name) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(y_te, pred, s=8, alpha=0.2, color="#8e44ad")
    lims = [y_te.min(), y_te.max()]
    ax1.plot(lims, lims, "k--", alpha=0.6)
    ax1.set(title=f"{name}: predicted vs actual magnitude",
            xlabel="Actual M", ylabel="Predicted M")
    residuals = pred - y_te
    ax2.scatter(pred, residuals, s=8, alpha=0.2, color="#16a085")
    ax2.axhline(0, color="k", ls="--", alpha=0.6)
    ax2.set(title="Residuals", xlabel="Predicted M", ylabel="Residual")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "magnitude_pred_vs_actual.png", dpi=130)
    plt.close(fig)


def plot_importance(model, feature_names) -> None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    order = np.argsort(importances)[::-1][:12]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in order][::-1],
            importances[order][::-1], color="#d35400")
    ax.set(title="Magnitude regressor — top feature importances",
           xlabel="Importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "magnitude_feature_importance.png", dpi=130)
    plt.close(fig)


def main() -> None:
    matplotlib.use("Agg")  # headless-safe when run as a script
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X, y = _load_matrix()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE)

    rows, preds = [], {}
    for name, model in build_models().items():
        metrics, pred = evaluate(name, model, X_tr, y_tr, X_te, y_te)
        rows.append(metrics)
        preds[name] = (model, pred)
        print(f"{name:20s} RMSE={metrics['RMSE']:.3f} "
              f"MAE={metrics['MAE']:.3f} R2={metrics['R2']:.3f}")

    table = pd.DataFrame(rows).sort_values("RMSE")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "regression_metrics.csv", index=False)

    best_name = table.iloc[0]["model"]
    best_model, best_pred = preds[best_name]
    plot_pred_vs_actual(y_te, best_pred, best_name)
    plot_importance(best_model, list(X.columns))
    print(f"\nBest model by RMSE: {best_name}")
    print(f"Naive baseline (predict mean) RMSE: {y_te.std():.3f}")
    print("Saved metrics table + 2 figures.")


if __name__ == "__main__":
    main()
