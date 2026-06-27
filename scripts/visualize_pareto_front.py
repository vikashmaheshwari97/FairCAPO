"""Render static Pareto-front figures for a FairCAPO run (no Streamlit).

Reads a run's `phase2_all_candidates.csv` or an explicit candidates CSV and writes PNGs:
  - pareto_perf_cost.png        performance vs cost, colored by fairness_risk
  - pareto_pairwise.png         2x3 grid of all objective-pair projections
  - objective_parallel.png      parallel-coordinates over the 4 objectives

Usage:
  python scripts/visualize_pareto_front.py \
      --run outputs/phase2_budgeted_mocapo_subj \
      --title "SUBJ / Mistral-Small-3.2" \
      --out outputs/figures/subj_mistral
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OBJECTIVES = ["performance", "cost", "risk", "fairness_risk"]


def load_candidates(run_dir: str, csv_path_arg: str = "") -> pd.DataFrame:
    csv_path = Path(csv_path_arg) if csv_path_arg else Path(run_dir) / "phase2_all_candidates.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No candidate CSV at {csv_path}")
    df = pd.read_csv(csv_path)
    if "is_pareto" not in df.columns:
        df["is_pareto"] = False
    df["is_pareto"] = df["is_pareto"].astype(str).str.lower().isin({"true", "1"})
    for col in OBJECTIVES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def plot_perf_cost(df: pd.DataFrame, title: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    dom = df[~df.is_pareto]
    par = df[df.is_pareto].sort_values("cost")

    ax.scatter(
        dom.cost, dom.performance, c="lightgray", s=45,
        edgecolor="gray", label="dominated", zorder=2,
    )
    sc = ax.scatter(
        par.cost, par.performance, c=par.fairness_risk, cmap="RdYlGn_r",
        vmin=0.0, vmax=max(0.001, df.fairness_risk.max()), s=140,
        edgecolor="black", linewidth=1.2, label="Pareto", zorder=4,
    )
    ax.plot(par.cost, par.performance, "--", color="black", alpha=0.4, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("fairness_risk (lower = fairer)")

    ax.set_xlabel("cost (lower better)")
    ax.set_ylabel("performance (higher better)")
    ax.set_title(f"Pareto front - {title}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "pareto_perf_cost.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pairwise(df: pd.DataFrame, title: str, out: Path) -> Path:
    pairs = [
        ("cost", "performance"), ("fairness_risk", "performance"),
        ("risk", "performance"), ("cost", "fairness_risk"),
        ("risk", "fairness_risk"), ("cost", "risk"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (x, y) in zip(axes.flat, pairs):
        dom = df[~df.is_pareto]
        par = df[df.is_pareto]
        ax.scatter(dom[x], dom[y], c="lightgray", s=30, edgecolor="gray", zorder=2)
        ax.scatter(par[x], par[y], c="crimson", s=90, edgecolor="black", zorder=3)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Objective-pair projections - {title}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / "pareto_pairwise.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_parallel(df: pd.DataFrame, title: str, out: Path) -> Path:
    norm = df.copy()
    for col in OBJECTIVES:
        lo, hi = df[col].min(), df[col].max()
        norm[col] = 0.5 if hi == lo else (df[col] - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = range(len(OBJECTIVES))
    for _, row in norm[~norm.is_pareto].iterrows():
        ax.plot(xs, [row[c] for c in OBJECTIVES], color="lightgray", alpha=0.6, zorder=2)
    for _, row in norm[norm.is_pareto].iterrows():
        ax.plot(xs, [row[c] for c in OBJECTIVES], color="crimson", alpha=0.85,
                linewidth=1.8, zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(OBJECTIVES)
    ax.set_ylabel("min-max normalized (0=best-in-run extreme varies)")
    ax.set_title(f"Parallel coordinates (gray=dominated, red=Pareto) - {title}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "objective_parallel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="Static Pareto-front visualizer.")
    ap.add_argument("--run", default="outputs/phase2_budgeted_mocapo_subj",
                    help="Run directory containing phase2_all_candidates.csv")
    ap.add_argument("--csv", default="",
                    help="Optional candidates CSV. Use this for held-out eval CSVs.")
    ap.add_argument("--title", default="SUBJ / Mistral-Small-3.2")
    ap.add_argument("--out", default="outputs/figures/subj_mistral")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_candidates(args.run, args.csv)
    n_par = int(df.is_pareto.sum())
    print(f"Loaded {len(df)} candidates ({n_par} Pareto) from {args.csv or args.run}")

    paths = [
        plot_perf_cost(df, args.title, out),
        plot_pairwise(df, args.title, out),
        plot_parallel(df, args.title, out),
    ]
    for p in paths:
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
