"""MO-CAPO-Fig-1-style held-out accuracy/cost staircase for FairCAPO.

Plots the FairCAPO held-out Pareto front (Test Accuracy x Cost) as a connected
staircase, with rungs labeled by few-shot count, and overlays the (weaker,
zero-shot) MO-CAPO-style baseline points evaluated on the SAME Dtest for a
comparable axis. Matches the look of docs/MO-CAPO Figures/MO-CAPO 1.png.

Both inputs are test_eval_candidates.csv files produced by
scripts/evaluate_pareto_on_test.py (same 50-example held-out SUBJ split), so
accuracy and cost are directly comparable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def pareto_front_2d(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated set maximizing performance, minimizing cost."""
    pts = df.sort_values(["cost", "performance"], ascending=[True, False])
    front, best_perf = [], -1.0
    for _, r in pts.iterrows():
        if r["performance"] > best_perf:        # cheaper-or-equal AND strictly better acc
            front.append(r)
            best_perf = r["performance"]
    return pd.DataFrame(front)


def few_shot_map(portfolio_csv: str) -> dict[str, int]:
    p = Path(portfolio_csv)
    if not p.exists():
        return {}
    pf = pd.read_csv(p)
    if "num_few_shot" not in pf.columns:
        return {}
    return {str(r["candidate_id"]): int(r.get("num_few_shot", 0) or 0)
            for _, r in pf.iterrows()}


def rung_label(n: int) -> str:
    if n <= 0:
        return "zero-shot"
    if n == 1:
        return "+1 few-shot"
    return f"+{n} few-shot"


def main() -> None:
    ap = argparse.ArgumentParser(description="FairCAPO held-out accuracy/cost staircase.")
    ap.add_argument("--fair", default="outputs/evaluation/subj_mistral_ddev30/test_eval_candidates.csv",
                    help="FairCAPO Dtest candidates CSV.")
    ap.add_argument("--portfolio",
                    default="outputs/phase2_budgeted_mocapo_subj_ddev30_intensified/phase2_prompt_portfolio.csv",
                    help="FairCAPO portfolio CSV (for few-shot counts).")
    ap.add_argument("--mocapo",
                    default="outputs/evaluation/mocapo_style_subj_ddev30/test_eval_candidates.csv",
                    help="MO-CAPO-style Dtest candidates CSV (optional overlay).")
    ap.add_argument("--title", default="SUBJ / Mistral-Small-3.2 (held-out, 50 ex.)")
    ap.add_argument("--out", default="outputs/figures/paper_subj_ddev30_intensified/fig_pareto_staircase.png")
    ap.add_argument("--color-fairness", action="store_true",
                    help="Color the front markers by fairness_risk with a colorbar, "
                         "and annotate each rung with BBQ details when available. "
                         "Use for the BBQ fairness staircase.")
    args = ap.parse_args()

    fair = pd.read_csv(args.fair)
    front = pareto_front_2d(fair).sort_values("cost")
    fsmap = few_shot_map(args.portfolio)
    front["num_few_shot"] = front["source_candidate_id"].astype(str).map(fsmap).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    # FairCAPO staircase (step line through the front, then markers on top).
    xs, ys = front["cost"].to_numpy(), front["performance"].to_numpy()
    ax.step([*xs, xs[-1] * 1.02], [*ys, ys[-1]], where="post",
            color="0.45", linewidth=1.6, zorder=2)

    if args.color_fairness and "fairness_risk" in front.columns:
        # Markers colored by the held-out fairness objective; a colorbar makes
        # "greener = fairer" explicit. fmax has a floor so a near-constant
        # (all ~0) front still renders with a sensible scale.
        fvals = front["fairness_risk"].astype(float).to_numpy()
        fmax = max(0.05, float(fvals.max()))
        sc = ax.scatter(xs, ys, c=fvals, cmap="RdYlGn_r", vmin=0.0, vmax=fmax,
                        s=220, marker="o", edgecolor="black", linewidth=1.3,
                        zorder=3, label="FairCAPO front")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("fairness_risk (configured BBQ score; lower = fairer)")

        # Annotate each rung with the BBQ bias breakdown when the evaluator wrote it.
        samb_col = "detail_bbq_sAMB" if "detail_bbq_sAMB" in front.columns else None
        sdis_col = "detail_bbq_sDIS" if "detail_bbq_sDIS" in front.columns else None
        for _, r in front.iterrows():
            samb = float(r[samb_col]) if samb_col else float(r["fairness_risk"])
            label = f"|sAMB|={abs(samb):.2f}"
            if sdis_col:
                label += f"\nsDIS={float(r[sdis_col]):.2f}"
            ax.annotate(label, (r["cost"], r["performance"]),
                        textcoords="offset points", xytext=(8, -6), fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))
    else:
        ax.scatter(xs, ys, s=180, c="black", marker="o", zorder=3, label="FairCAPO front")

        # Rung labels by few-shot count.
        for _, r in front.iterrows():
            ax.annotate(rung_label(int(r["num_few_shot"])), (r["cost"], r["performance"]),
                        textcoords="offset points", xytext=(8, -4), fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))

    # MO-CAPO-style baseline overlay (weaker zero-shot reference) — only when an
    # overlay file is actually supplied (e.g. SUBJ); skipped for BBQ.
    mo_path = Path(args.mocapo) if args.mocapo.strip() else None
    has_overlay = mo_path is not None and mo_path.is_file()
    if has_overlay:
        mo = pd.read_csv(mo_path)
        ax.scatter(mo["cost"], mo["performance"], s=200, marker="^",
                   c="#7B6FD0", edgecolor="black", linewidth=1.4, zorder=4,
                   label="MO-CAPO-style (zero-shot baseline)*")

    ax.set_xlabel("Token-weighted held-out cost (0.08*in + 0.32*out) [a.u.]")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"FairCAPO held-out accuracy/cost staircase - {args.title}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    rect = (0, 0, 1, 1)
    if has_overlay:
        fig.text(0.01, 0.01,
                 "*MO-CAPO-style = weak phase-1 zero-shot baseline (no few-shot lever / "
                 "intensification); lower reference, not a budget-matched MO-CAPO.",
                 fontsize=6.5, color="0.4")
        rect = (0, 0.03, 1, 1)
    fig.tight_layout(rect=rect)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"FairCAPO front: {len(front)} points (perf {ys.min():.2f}-{ys.max():.2f}, "
          f"cost {xs.min():.0f}-{xs.max():.0f})")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
