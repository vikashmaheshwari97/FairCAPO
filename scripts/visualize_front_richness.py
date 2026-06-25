"""Front-richness comparison (search basis): FairCAPO vs NSGA-II-PO + fairness.

This figure makes FairCAPO's clearest win visible: for the SAME budget and the
SAME objectives, FairCAPO's budgeted/intensified search returns a RICH Pareto
front (a menu of accuracy/cost trade-offs) while the off-the-shelf NSGA-II-PO
returns a single take-it-or-leave-it point.

All series are drawn on the SAME SEARCH basis (each method's own
`*_all_candidates.csv`, Pareto rows only, deduplicated by objective vector), so
accuracy and cost are directly comparable — we do NOT mix in held-out numbers.
Markers are colored by `fairness_risk` (|sAMB|) so fairness is still visible.

Usage:
    PYTHONPATH=. python scripts/visualize_front_richness.py \
        --faircapo outputs/phase2_budgeted_mocapo_bbq_local/phase2_all_candidates.csv \
        --nsga outputs/baselines/nsga2_po_bbq_local/nsga2_po_all_candidates.csv \
        --ablation outputs/mocapo_baseline_bbq_local/phase2_all_candidates.csv \
        --title "BBQ / Mistral-Small-3.2 (search basis)" \
        --out outputs/figures/paper_bbq_local/fig_front_richness_bbq.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _truthy(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def load_front(path: str) -> pd.DataFrame:
    """Pareto rows of a candidates CSV, deduped by (perf, cost, risk, fairness)."""
    df = pd.read_csv(path)
    if "is_pareto" in df.columns:
        df = df[df["is_pareto"].map(_truthy)]
    for c in ["performance", "cost", "risk", "fairness_risk"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].astype(float)
    df = df.drop_duplicates(subset=["performance", "cost", "risk", "fairness_risk"])
    return df


def pareto_front_2d(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated set maximizing performance, minimizing cost (sorted by cost)."""
    pts = df.sort_values(["cost", "performance"], ascending=[True, False])
    front, best_perf = [], -1.0
    for _, r in pts.iterrows():
        if r["performance"] > best_perf:
            front.append(r)
            best_perf = r["performance"]
    return pd.DataFrame(front).sort_values("cost")


def main() -> None:
    ap = argparse.ArgumentParser(description="FairCAPO vs NSGA-II-PO front-richness (search basis).")
    ap.add_argument("--faircapo", default="outputs/phase2_budgeted_mocapo_bbq_local/phase2_all_candidates.csv")
    ap.add_argument("--nsga", default="outputs/baselines/nsga2_po_bbq_local/nsga2_po_all_candidates.csv")
    ap.add_argument("--ablation", default="", help="Optional MO-CAPO fairness-off front overlay.")
    ap.add_argument("--title", default="BBQ / Mistral-Small-3.2 (search basis)")
    ap.add_argument("--out", default="outputs/figures/paper_bbq_local/fig_front_richness_bbq.png")
    args = ap.parse_args()

    # NOTE: these are 4-OBJECTIVE Pareto fronts (perf, cost, risk, fairness). We
    # plot EVERY front member as a marker (so a 5-point front shows 5 dots), and
    # only use the 2D acc/cost non-dominated SUBSET for the connecting staircase
    # line. Projecting onto acc/cost would otherwise hide members that are on the
    # front because of fairness/risk, understating the front's true richness.
    fc_front = load_front(args.faircapo)
    nsga_front = load_front(args.nsga)

    # Shared fairness color scale across all plotted points.
    fvals = list(fc_front["fairness_risk"]) + list(nsga_front["fairness_risk"])
    abl_front = None
    if args.ablation and Path(args.ablation).exists():
        abl_front = load_front(args.ablation)
        fvals += list(abl_front["fairness_risk"])
    fmax = max(0.05, max(fvals) if fvals else 0.05)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    cmap = "RdYlGn_r"

    # Optional ablation front (faint, for context — also a multi-point front).
    if abl_front is not None and len(abl_front):
        abl_line = pareto_front_2d(abl_front)
        ax.step([*abl_line["cost"], abl_line["cost"].iloc[-1] * 1.02],
                [*abl_line["performance"], abl_line["performance"].iloc[-1]],
                where="post", color="0.75", linewidth=1.2, linestyle="--", zorder=1)
        ax.scatter(abl_front["cost"], abl_front["performance"], c=abl_front["fairness_risk"],
                   cmap=cmap, vmin=0.0, vmax=fmax, s=90, marker="s",
                   edgecolor="0.5", linewidth=1.0, zorder=2,
                   label=f"MO-CAPO off ({len(abl_front)}-pt front)")

    # FairCAPO front — every member plotted; staircase line through the acc/cost subset.
    fc_line = pareto_front_2d(fc_front)
    ax.step([*fc_line["cost"], fc_line["cost"].iloc[-1] * 1.02],
            [*fc_line["performance"], fc_line["performance"].iloc[-1]],
            where="post", color="0.4", linewidth=1.8, zorder=3)
    ax.scatter(fc_front["cost"], fc_front["performance"], c=fc_front["fairness_risk"],
               cmap=cmap, vmin=0.0, vmax=fmax,
               s=230, marker="o", edgecolor="black", linewidth=1.4, zorder=4,
               label=f"FairCAPO ({len(fc_front)}-pt front)")

    # NSGA-II-PO — the single point.
    sc = ax.scatter(nsga_front["cost"], nsga_front["performance"], c=nsga_front["fairness_risk"],
                    cmap=cmap, vmin=0.0, vmax=fmax, s=320, marker="^",
                    edgecolor="black", linewidth=1.6, zorder=5,
                    label=f"NSGA-II-PO + fairness ({len(nsga_front)}-pt front)")
    for _, r in nsga_front.iterrows():
        ax.annotate("single point\n(no trade-off menu)", (r["cost"], r["performance"]),
                    textcoords="offset points", xytext=(10, -28), fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("fairness_risk = |sAMB|  (lower = fairer)")

    ax.set_xlabel("Avg. Cost [$] per 1M Calls  (search basis)")
    ax.set_ylabel("Dev Accuracy")
    ax.set_title(f"Pareto-front richness — same budget, same objectives\n{args.title}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(0.01, 0.01,
             "Same budget & objectives; only the search algorithm differs. FairCAPO's "
             "budgeting+intensification yields a rich menu; NSGA-II-PO yields one point.",
             fontsize=6.8, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"FairCAPO front: {len(fc_front)} pts | NSGA front: {len(nsga_front)} pts"
          + (f" | ablation front: {len(abl_front)} pts" if abl_front is not None else ""))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
