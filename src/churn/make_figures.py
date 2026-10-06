"""Generate EDA and model figures into reports/figures.

Run from the repo root:  python -m src.churn.make_figures
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .config import FIG_DIR, METRICS_PATH, MODEL_PATH
from .data import load_clean

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
C_STAY, C_LEAVE = "#2a9d8f", "#e76f51"


def churn_rate_bar(df, col, fname, title, order=None):
    rate = df.groupby(col)["Churn"].mean().mul(100)
    if order:
        rate = rate.loc[order]
    ax = rate.plot.bar(color=C_LEAVE, figsize=(6, 4), rot=0)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Churn rate (%)"); ax.set_xlabel(""); ax.set_title(title)
    plt.tight_layout(); plt.savefig(FIG_DIR / fname); plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()

    # 1. class balance
    counts = df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts()
    plt.figure(figsize=(4.5, 4.5))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=[C_STAY, C_LEAVE], startangle=90,
            wedgeprops={"width": 0.45})
    plt.title("Class balance"); plt.tight_layout(); plt.savefig(FIG_DIR / "01_class_balance.png"); plt.close()

    # 2-4. churn rate by key categoricals
    churn_rate_bar(df, "Contract", "02_churn_by_contract.png", "Churn rate by contract type",
                   ["Month-to-month", "One year", "Two year"])
    churn_rate_bar(df, "InternetService", "03_churn_by_internet.png", "Churn rate by internet service")
    churn_rate_bar(df, "PaymentMethod", "04_churn_by_payment.png", "Churn rate by payment method")

    # 5. tenure & monthly charges distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges"]):
        for val, label, color in [(0, "Stayed", C_STAY), (1, "Churned", C_LEAVE)]:
            ax.hist(df.loc[df.Churn == val, col], bins=30, alpha=0.65, label=label, color=color)
        ax.set_title(f"{col} by churn"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "05_numeric_distributions.png"); plt.close()

    # 6. model figures
    saved = joblib_load()
    metrics = json.loads(METRICS_PATH.read_text())
    y_test = np.load(MODEL_PATH.parent / "test_y.npy")
    proba = np.load(MODEL_PATH.parent / "test_proba.npy")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, color=C_LEAVE,
                                     name=f"Tuned RF (AUC {metrics['tuned_rf']['test_auc']})")
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_title("ROC curve (held-out test set)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "06_roc_curve.png"); plt.close()

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(y_test, (proba >= 0.5).astype(int), ax=ax, cmap="Oranges",
                                            display_labels=["Stayed", "Churned"], colorbar=False)
    ax.set_title("Confusion matrix (threshold 0.5)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "07_confusion_matrix.png"); plt.close()

    imp = pd.Series(metrics["permutation_importance"]).sort_values().tail(10)
    ax = imp.plot.barh(color=C_STAY, figsize=(6.5, 4.5))
    ax.set_xlabel("Drop in test ROC-AUC when shuffled"); ax.set_title("Top 10 features (permutation importance)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "08_feature_importance.png"); plt.close()
    print("Figures saved to", FIG_DIR)


def joblib_load():
    import joblib
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    main()
