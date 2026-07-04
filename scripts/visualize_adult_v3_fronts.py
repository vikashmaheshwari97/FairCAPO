from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def is_true(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_points(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "is_pareto" in data.columns:
        data = data[data["is_pareto"].map(is_true)]
    for column in ("performance", "cost", "fairness_risk"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    return data.drop_duplicates(
        subset=["performance", "cost", "fairness_risk"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faircapo", required=True)
    parser.add_argument("--mocapo", required=True)
    parser.add_argument("--nsga", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    series = [
        ("FairCAPO", load_points(args.faircapo), "o"),
        ("MO-CAPO", load_points(args.mocapo), "s"),
        ("NSGA-II-PO", load_points(args.nsga), "^"),
    ]
    fairness_max = max(
        0.05,
        max(float(v) for _, frame, _ in series for v in frame["fairness_risk"]),
    )

    figure, axis = plt.subplots(figsize=(9.2, 6.0))
    color_source = None
    for label, frame, marker in series:
        color_source = axis.scatter(
            frame["cost"],
            frame["performance"],
            c=frame["fairness_risk"],
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=fairness_max,
            marker=marker,
            s=170,
            edgecolor="black",
            linewidth=1.1,
            label=f"{label} ({len(frame)} points)",
        )

    colorbar = figure.colorbar(color_source, ax=axis)
    colorbar.set_label("Held-out equalized-odds risk")
    axis.set_xlabel("Held-out inference cost over 2,000 records")
    axis.set_ylabel("Held-out accuracy")
    axis.set_title(args.title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

    for label, frame, _ in series:
        print(f"{label}: {len(frame)} unique Pareto points")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
