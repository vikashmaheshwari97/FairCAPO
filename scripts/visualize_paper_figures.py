"""Paper-style HEAL-CAPO figures matching the MO-CAPO figure aesthetic,
extended with the FAIRNESS dimension (the HEAL-CAPO contribution).

Reference figures (docs/MO-CAPO Figures):
  Fig 3/7  empirical attainment surface: Test Accuracy vs Avg Cost, baselines overlaid
  Fig 2/6  nR2 / metric vs token budget trajectory

This script reproduces that look and adds fairness:
  fig_accuracy_cost_fairness.png   Fig-3 style + fairness color (HEAL-CAPO front)
  fig_method_comparison.png        all baselines on Accuracy-vs-Cost, fairness-colored
  fig_fairness_tradeoffs.png       3-panel: Acc-Cost, Acc-Fairness, Cost-Fairness

Usage:
  python scripts/visualize_paper_figures.py \
      --run outputs/phase2_budgeted_mocapo_subj_ddev30 \
      --table outputs/experiment_table/subj_mistral/experiment_table.csv \
      --title "SUBJ / Mistral-Small-3.2" \
      --out outputs/figures/paper_subj_mistral
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Match MO-CAPO paper marker/colour conventions where possible.
METHOD_STYLE = {
    "HEAL-CAPO": dict(color="#000000", marker="o", label="HEAL-CAPO"),
    "FairCAPO": dict(color="#000000", marker="o", label="FairCAPO"),
    "MO-CAPO (fairness off)": dict(color="#1F77B4", marker="D", label="MO-CAPO (fairness off)"),
    "NSGA-II-PO + fairness": dict(color="#E8A33D", marker="s", label="NSGA-II-PO + fairness"),
    "Post-hoc fair. (held-out)": dict(color="#7B5EA7", marker="^", label="Post-hoc fair. (held-out)"),
    "MO-CAPO-style": dict(color="#000000", marker="o", label="MO-CAPO"),
    "NSGA-II-PO": dict(color="#E8A33D", marker="s", label="NSGA-II-PO"),
    "CAPO": dict(color="#7B5EA7", marker="^", label="CAPO"),
    "GEPA": dict(color="#2E8B57", marker="D", label="GEPA"),
    "EvoPromptGA": dict(color="#C44E52", marker="v", label="EvoPromptGA"),
    "EvoPromptDE": dict(color="#C44E52", marker="<", label="EvoPromptDE"),
    "OPRO": dict(color="#8C8C8C", marker="P", label="OPRO"),
    "Initial": dict(color="#999999", marker="x", label="Initial Instructions"),
}


def _pareto_mask(perf, cost) -> list[bool]:
    """Non-dominated for (max perf, min cost)."""
    pts = list(zip(perf, cost))
    keep = []
    for i, (p, c) in enumerate(pts):
        dominated = any(
            (pp >= p and cc <= c) and (pp > p or cc < c)
            for j, (pp, cc) in enumerate(pts) if j != i
        )
        keep.append(not dominated)
    return keep


def load_run(run_dir: str) -> pd.DataFrame:
    df = pd.read_csv(Path(run_dir) / "phase2_all_candidates.csv")
    if "is_pareto" in df.columns:
        df["is_pareto"] = df["is_pareto"].astype(str).str.lower().isin({"true", "1"})
    else:
        df["is_pareto"] = False
    for c in ["performance", "cost", "risk", "fairness_risk"]:
        if c not in df.columns:
            df[c] = 0.0
    return df


def fig_accuracy_cost_fairness(df: pd.DataFrame, title: str, out: Path) -> Path:
    """Fig-3 style attainment: Accuracy vs Cost, Pareto step line, fairness color."""
    fig, ax = plt.subplots(figsize=(7, 5))
    dom = df[~df.is_pareto]
    par = df[df.is_pareto].sort_values("cost")

    ax.scatter(dom.cost, dom.performance, c="lightgray", s=40,
               edgecolor="gray", zorder=2, label="dominated")
    # attainment step (median-style staircase over the Pareto front).
    if len(par) > 0:
        ax.step(par.cost, par.performance, where="post",
                color="black", alpha=0.5, zorder=3)
    sc = ax.scatter(par.cost, par.performance, c=par.fairness_risk,
                    cmap="RdYlGn_r", vmin=0.0,
                    vmax=max(0.001, df.fairness_risk.max()),
                    s=150, edgecolor="black", linewidth=1.2, zorder=4,
                    label="HEAL-CAPO front")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("fairness_risk (lower = fairer)")
    # NOT "$ per 1M calls": this is the token-weighted eval cost summed over the
    # whole dev set (0.08*Sum(input_tok) + 0.32*Sum(output_tok)), in arbitrary
    # units. It equals N_eval_items * (true $/1M-calls), and on the search basis
    # also folds in the one-time fairness-eval tokens.
    ax.set_xlabel("Eval token-cost (0.08·in + 0.32·out, summed over dev set) [a.u.]")
    ax.set_ylabel("Test Score (Accuracy)")
    ax.set_title(f"Attainment surface + fairness — {title}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    p = out / "fig_accuracy_cost_fairness.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_method_comparison(table_csv: str, title: str, out: Path) -> Path | None:
    """Fig-3 style multi-method scatter (Accuracy vs Cost), fairness-colored."""
    path = Path(table_csv)
    if not path.exists():
        return None
    t = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    fmax = max(0.001, t["fairness_risk"].max())

    # When every method ties on accuracy (common on BBQ where perf saturates at
    # 1.0), the y=accuracy axis is uninformative and markers/labels collapse onto
    # one horizontal line. Fall back to cost-vs-fairness, the two axes that
    # actually vary, so the trade-off (cheap-but-biased vs fair-but-costly) reads.
    perf_spread = float(t["performance"].max() - t["performance"].min())
    by_fairness = perf_spread < 1e-6
    ycol = "fairness_risk" if by_fairness else "performance"

    # Draw higher-fairness (worse) markers LAST so they sit on top and are never
    # hidden behind a fairer marker at the same cost (e.g. ablation 0.12 vs
    # post-hoc 0.0, both at cost 268).
    rows = t.sort_values("fairness_risk").to_dict("records")
    placed: list[tuple[float, float]] = []
    for r in rows:
        style = METHOD_STYLE.get(r["method"], dict(color="gray", marker="o"))
        x, y = r["cost"], r[ycol]
        ax.scatter(x, y, marker=style["marker"], s=190, c=[r["fairness_risk"]],
                   cmap="RdYlGn_r", vmin=0, vmax=fmax, edgecolor=style["color"],
                   linewidth=2.2, zorder=3)
        # Nudge each label vertically away from any already-placed label that
        # shares roughly the same (cost, y) spot, so co-located points stay legible.
        dy = 8
        for px, py in placed:
            if abs(px - x) < (t["cost"].max() - t["cost"].min() + 1e-9) * 0.20 \
               and abs(py - y) < (abs(fmax) + 1e-9) * 0.06:
                dy += 15
        placed.append((x, y))
        ax.annotate(f"{r['method']}\n(fair={r['fairness_risk']:.3f}, acc={r['performance']:.3f})",
                    (x, y), textcoords="offset points", xytext=(8, dy),
                    fontsize=7.5, zorder=4)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                               norm=plt.Normalize(vmin=0, vmax=fmax))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("fairness_risk (fill: lower = fairer)")
    # See note above: token-weighted eval cost summed over the dev set, NOT $/1M calls.
    ax.set_xlabel("Eval token-cost (0.08·in + 0.32·out, summed over dev set) [a.u.]")
    if by_fairness:
        ax.set_ylabel("fairness_risk  (lower = fairer)")
        ax.set_title(f"Method comparison — cost vs fairness (accuracy tied at "
                     f"{t['performance'].max():.3f}) — {title}")
        ax.margins(y=0.25)
    else:
        ax.set_ylabel("Test Score (Accuracy)")
        ax.set_title(f"Method comparison (accuracy / cost / fairness) — {title}")
    ax.margins(x=0.18)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out / "fig_method_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_fairness_tradeoffs(df: pd.DataFrame, title: str, out: Path) -> Path:
    """3-panel: the fairness objective vs the other three."""
    panels = [
        ("cost", "performance", "Accuracy vs Cost"),
        ("fairness_risk", "performance", "Accuracy vs Fairness risk"),
        ("fairness_risk", "cost", "Cost vs Fairness risk"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (x, y, t) in zip(axes, panels):
        dom = df[~df.is_pareto]
        par = df[df.is_pareto]
        ax.scatter(dom[x], dom[y], c="lightgray", s=35, edgecolor="gray", zorder=2)
        ax.scatter(par[x], par[y], c="crimson", s=110, edgecolor="black", zorder=3,
                   label="Pareto")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(t)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle(f"Fairness trade-offs — {title}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out / "fig_fairness_tradeoffs.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description="Paper-style HEAL-CAPO fairness figures.")
    ap.add_argument("--run", default="outputs/phase2_budgeted_mocapo_subj")
    ap.add_argument("--table",
                    default="outputs/experiment_table/subj_mistral/experiment_table.csv")
    ap.add_argument("--title", default="SUBJ / Mistral-Small-3.2")
    ap.add_argument("--out", default="outputs/figures/paper_subj_mistral")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_run(args.run)
    print(f"Loaded {len(df)} candidates ({int(df.is_pareto.sum())} Pareto) from {args.run}")

    made = [
        fig_accuracy_cost_fairness(df, args.title, out),
        fig_method_comparison(args.table, args.title, out),
        fig_fairness_tradeoffs(df, args.title, out),
    ]
    for p in made:
        if p is not None:
            print(f"Saved {p}")


if __name__ == "__main__":
    main()
