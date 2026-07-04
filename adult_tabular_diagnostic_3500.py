from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from heal_capo.adult_diag_models import FEATURES, NUMERIC, build_models
from heal_capo.adult_v4_data import examples_from_manifest
from heal_capo.adult_v4_manifest import build_fixed_manifest

DATA = "data/adult_semantic_v3.csv"
MANIFEST = "data/adult_v4_fixed_split_seed0.json"


def number(value):
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group(0)) if match else 0.0


def take(raw, rows):
    indices = [int((row.metadata or {})["source_index"]) for row in rows]
    return raw.iloc[indices].copy().reset_index(drop=True)


if __name__ == "__main__":
    if not Path(MANIFEST).is_file():
        build_fixed_manifest(DATA, MANIFEST)
    raw = pd.read_csv(DATA)
    train = take(raw, examples_from_manifest("dev", DATA, MANIFEST) + examples_from_manifest("shots", DATA, MANIFEST))
    test = take(raw, examples_from_manifest("test", DATA, MANIFEST))
    for name in NUMERIC:
        train[name] = train[name].map(number)
        test[name] = test[name].map(number)
    y_train = train["income"].astype(str).str.replace(".", "", regex=False)
    y_test = test["income"].astype(str).str.replace(".", "", regex=False)
    results = []
    for name, model in build_models().items():
        model.fit(train[FEATURES], y_train)
        results.append({"method": name, "accuracy": accuracy_score(y_test, model.predict(test[FEATURES]))})
    out = Path("outputs/diagnostics/adult_tabular_v4_3500_seed0")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(out / "results.csv", index=False)
    print(frame.to_string(index=False))
