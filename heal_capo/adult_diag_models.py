from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = [
    "age", "workclass", "education", "marital.status", "occupation",
    "relationship", "capital.gain", "capital.loss", "hours.per.week", "native.country",
]
NUMERIC = ["age", "capital.gain", "capital.loss", "hours.per.week"]
CATEGORICAL = [name for name in FEATURES if name not in NUMERIC]


def build_models():
    sparse = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    dense = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return {
        "logistic_regression": Pipeline([
            ("prep", sparse),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("prep", dense),
            ("model", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0)),
        ]),
        "random_forest": Pipeline([
            ("prep", dense),
            ("model", RandomForestClassifier(
                n_estimators=600, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=0, n_jobs=-1,
            )),
        ]),
    }
