"""
Tsunami-risk classification.

Predicts the USGS `tsunami` flag (~2.4% positive rate) from physical event
features. Because the classes are heavily imbalanced, the module compares four
strategies for handling that imbalance and evaluates them with imbalance-aware
metrics (PR-AUC / recall) rather than accuracy:

    1. Logistic regression + class_weight="balanced"  (linear baseline)
    2. Random forest      + class_weight="balanced"
    3. XGBoost            + scale_pos_weight
    4. Random forest      + SMOTE oversampling

For an early-warning system, missing a real tsunami (false negative) is far
costlier than a false alarm, so we also pick an operating threshold that
targets high recall and report the resulting confusion matrix.

Run:
    python src/classification.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocessing import (PROCESSED_CSV, RANDOM_STATE, build_features,
                           classification_matrix, clean, load_raw)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def _load_matrix() -> tuple[pd.DataFrame, pd.Series]:
    if PROCESSED_CSV.exists():
        df = pd.read_csv(PROCESSED_CSV)
    else:
        df = build_features(clean(load_raw()))
    return classification_matrix(df)


def build_models(scale_pos_weight: float) -> dict:
    """Return the four candidate estimators keyed by a short name."""
    return {
        "LogReg (balanced)": ImbPipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "RandomForest (balanced)": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost (scale_pos_weight)": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            n_jobs=-1, random_state=RANDOM_STATE),
        "RandomForest + SMOTE": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                           random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(name, model, X_tr, y_tr, X_te, y_te) -> tuple[dict, np.ndarray]:
    """Fit `model`, return a metrics dict and the positive-class probabilities."""
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "ROC_AUC": roc_auc_score(y_te, proba),
        "PR_AUC": average_precision_score(y_te, proba),
        "F1@0.5": f1_score(y_te, pred, zero_division=0),
        "precision@0.5": precision_score(y_te, pred, zero_division=0),
        "recall@0.5": recall_score(y_te, pred, zero_division=0),
    }
    return metrics, proba


def threshold_for_recall(y_te, proba, target_recall=0.90) -> float:
    """Smallest threshold whose recall on the test set is >= target_recall."""
    prec, rec, thr = precision_recall_curve(y_te, proba)
    # precision_recall_curve returns len(thr) = len(rec) - 1
    ok = np.where(rec[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return 0.5
    return float(thr[ok[-1]])


def plot_curves(results: dict, y_te) -> None:
    """ROC and precision-recall curves for all models on one figure."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    baseline = y_te.mean()
    for name, proba in results.items():
        fpr, tpr, _ = roc_curve(y_te, proba)
        ax_roc.plot(fpr, tpr, lw=1.8,
                    label=f"{name} (AUC={roc_auc_score(y_te, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_te, proba)
        ax_pr.plot(rec, prec, lw=1.8,
                   label=f"{name} (AP={average_precision_score(y_te, proba):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set(title="ROC curves", xlabel="False positive rate",
               ylabel="True positive rate")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_pr.axhline(baseline, ls="--", color="k", alpha=0.4,
                  label=f"No-skill ({baseline:.3f})")
    ax_pr.set(title="Precision-Recall curves (imbalance-aware)",
              xlabel="Recall", ylabel="Precision")
    ax_pr.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tsunami_roc_pr.png", dpi=130)
    plt.close(fig)


def plot_feature_importance(model, feature_names) -> None:
    """Bar chart of the winning tree model's feature importances."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    order = np.argsort(importances)[::-1][:12]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in order][::-1],
            importances[order][::-1], color="#2980b9")
    ax.set(title="Tsunami classifier — top feature importances",
           xlabel="Importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tsunami_feature_importance.png", dpi=130)
    plt.close(fig)


def main() -> None:
    matplotlib.use("Agg")  # headless-safe when run as a script
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X, y = _load_matrix()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    models = build_models(spw)

    rows, probas = [], {}
    for name, model in models.items():
        metrics, proba = evaluate(name, model, X_tr, y_tr, X_te, y_te)
        rows.append(metrics)
        probas[name] = proba
        print(f"{name:28s} ROC={metrics['ROC_AUC']:.3f} "
              f"PR-AUC={metrics['PR_AUC']:.3f} "
              f"F1={metrics['F1@0.5']:.3f} recall={metrics['recall@0.5']:.3f}")

    table = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "classification_metrics.csv", index=False)
    plot_curves(probas, y_te)

    # Winner by PR-AUC -> refit for feature importance + recall-oriented threshold.
    best_name = table.iloc[0]["model"]
    best_model = models[best_name]
    best_proba = probas[best_name]

    thr = threshold_for_recall(y_te, best_proba, target_recall=0.90)
    pred = (best_proba >= thr).astype(int)
    cm = confusion_matrix(y_te, pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nBest model by PR-AUC: {best_name}")
    print(f"Early-warning threshold (recall>=0.90): {thr:.3f}")
    print(f"  Confusion @thr -> TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"  recall={recall_score(y_te, pred):.3f} "
          f"precision={precision_score(y_te, pred, zero_division=0):.3f}")

    # Feature importance for the underlying tree model.
    imp_model = (best_model.named_steps["clf"]
                 if hasattr(best_model, "named_steps") else best_model)
    plot_feature_importance(imp_model, list(X.columns))
    print(f"\nSaved metrics table + 2 figures.")


if __name__ == "__main__":
    main()
