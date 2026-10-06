from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import NUMERIC, RANDOM_STATE


def make_preprocessor(X):
    categorical = [c for c in X.columns if c not in NUMERIC]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def make_pipeline(X, estimator):
    """preprocess -> SMOTE -> model. SMOTE only runs while fitting, never at predict time,
    and inside cross-validation it only touches the training folds."""
    return Pipeline(
        [
            ("prep", make_preprocessor(X)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", estimator),
        ]
    )
