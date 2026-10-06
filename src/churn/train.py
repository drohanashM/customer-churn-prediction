"""Train, compare, tune and save the churn model.

Run from the repo root:  python -m src.churn.train
"""
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import METRICS_PATH, MODEL_PATH, RANDOM_STATE
from .data import load_clean, split_xy
from .pipeline import make_pipeline


def main():
    df = load_clean()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
    }

    comparison = {}
    for name, est in candidates.items():
        pipe = make_pipeline(X_train, est)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        comparison[name] = {"cv_auc": round(float(cv_auc.mean()), 4), "test_auc": round(float(test_auc), 4)}
        print(f"{name:22s} CV AUC {cv_auc.mean():.4f} | test AUC {test_auc:.4f}")

    # Tune the Random Forest
    search = RandomizedSearchCV(
        make_pipeline(X_train, RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        param_distributions={
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [8, 12, 16, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
        n_iter=20, scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print("Best params:", search.best_params_)

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tuned = {
        "test_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(y_test, pred, output_dict=True),
    }
    print(f"Tuned RF test AUC: {tuned['test_auc']}")
    print(classification_report(y_test, pred))

    # Permutation importance on the raw columns (maps to business-level features)
    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                                 n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    importance = dict(sorted(zip(X.columns, imp.importances_mean.round(4).tolist()),
                             key=lambda kv: -kv[1]))

    joblib.dump({"pipeline": best, "columns": list(X.columns)}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "n_customers": len(df),
        "churn_rate": round(float(y.mean()), 4),
        "comparison": comparison,
        "best_params": {k: (v if v is None or isinstance(v, str) else int(v))
                        for k, v in search.best_params_.items()},
        "tuned_rf": tuned,
        "permutation_importance": importance,
    }, indent=2))

    np.save(MODEL_PATH.parent / "test_proba.npy", proba)
    np.save(MODEL_PATH.parent / "test_y.npy", y_test.to_numpy())
    print("Saved model and metrics.")


if __name__ == "__main__":
    main()
