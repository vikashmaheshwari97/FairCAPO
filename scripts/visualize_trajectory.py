"""Trajectory plot: HV and nR2 vs budget spent (paper Fig 2).

Reframes a smaller budget run as "competitive early" — MO-CAPO reaches
near-final performance well before the full budget (§F of gap analysis,
MO-CAPO §5 trajectory discussion). The plot shows HV_opt / HV_pes (filled
band) and nR2 (twin axis, right spine) against cumulative budget spent, one
line per method, so you can compare convergence rates across FairCAPO, the
fairness-off ablation, and NSGA-II-PO + fairness.

Input: one or more ``*_trajectory.json`` files written by
``run_phase2_budgeted_mocapo.py`` (one snapshot = ``{iteration, budget_used,
front: [{performance,cost,risk,fairness_risk}], hypervolume,
optimistic_hypervolume, pessimistic_hypervolume, approximation_gap, nr2}``).

In order of ``--label``, pass one ``--trajectory`` per method so the legend
carries the right names.

Usage:
    PYTHONPATH=. python scripts/visualize_trajectory.py \
        --trajectory outputs/seed_0/phase2_budgeted_mocapo_bbq_local/budgeted_mocapo_trajectory.json  \
        --label     FairCAPO \
        --trajectory outputs/seed_0/mocapo_baseline_bbq_local/budgeted_mocapo_trajectory.json \
        --label     "MO-CAPO (fairness off)" \
        --title     "BBQ / Mistral-Small-3.2 — convergence vs budget" \
        --out       outputs/figures/paper_bbq_local/fig_trajectory_bbq.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_trajectory(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def hv_pair_plot_data(
    snapshots: list[dict],
) -> tuple[list[float], list[float], list[float]]:
    """(budget_used, hv_opt, hv_pes) — monotone-safe after dedupe on budget."""
    seen = set()
    x, y_opt, y_pes = [], [], []
    for snap in sorted(snapshots, key=lambda s: s["budget_used"]):
        h = snap.get("hypervolume")
        if h is None or isinstance(h, str) or not np.isfinite(h):
            continue
        x.append(float(snap["budget_used"]))
        y_opt.append(float(snap.get("optimistic_hypervolume", h)))
        y_pes.append(float(snap.get("pessimistic_hypervolume", h)))
    # Deduplicate on budget (final + iter-end can collide).
    dedup = []
    for xi, yo, yp in zip(x, y_opt, y_pes):
        k = round(xi, 4)
        if k in seen:
            continue
        seen.add(k)
        dedup.append((xi, yo, yp))
    if not dedup:
        return [], [], []
    xs, os, ps = zip(*dedup)
    return list(xs), list(os), list(ps)


def nr2_plot_data(
    snapshots: list[dict],
) -> tuple[list[float], list[float]]:
    """(budget_used, nr2) — only points where nr2 is non-null."""
    seen = set()
    x, y = [], []
    for snap in sorted(snapshots, key=lambda s: s["budget_used"]):
        nr = snap.get("nr2")
        if nr is None or isinstance(nr, str) or not np.isfinite(nr):
            continue
        k = round(snap["budget_used"], 4)
        if k in seen:
            continue
        seen.add(k)
        x.append(float(snap["budget_used"]))
        y.append(float(nr))
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HV/nR2 vs budget trajectory (paper Fig 2 style)."
    )
    ap.add_argument(
        "--trajectory",
        action="append",
        default=[],
        dest="trajectories",
        help="Path to a *_trajectory.json. Repeat for each method.",
    )
    ap.add_argument(
        "--label",
        action="append",
        default=[],
        dest="labels",
        help="Legend label for the preceding --trajectory.",
    )
    ap.add_argument(
        "--metric",
        choices=("hv", "nr2", "both"),
        default="both",
        help="Which metric(s) to plot (default: both = twin-axis).",
    )
    ap.add_argument(
        "--title",
        default="HV & nR2 vs Budget",
        help="Plot title.",
    )
    ap.add_argument(
        "--out",
        default="outputs/figures/trajectory/fig_trajectory.png",
        help="Output PNG path.",
    )
    args = ap.parse_args()

    if len(args.labels) != len(args.trajectories):
        ap.error("Provide one --label for each --trajectory.")

    if not args.trajectories:
        print("No --trajectory args; nothing to plot.")
        return

    # --- colour scheme — 4 distinct, accessible colours ---
    COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    styles = [("o", "-"), ("s", "--"), ("D", "-."), ("^", ":")]

    fig, ax_hv = plt.subplots(figsize=(9.6, 6.0))

    twin_kwargs = {}
    if args.metric in ("nr2", "both"):
        ax_nr2 = ax_hv.twinx()
        twin_kwargs["ax_nr2"] = ax_nr2

    for idx, (path, label) in enumerate(zip(args.trajectories, args.labels)):
        color = COLORS[idx % len(COLORS)]
        marker, linestyle = styles[idx % len(styles)]
        snapshots = load_trajectory(path)

        # --- HV band (left axis) ---
        if args.metric in ("hv", "both"):
            x_hv, y_opt, y_pes = hv_pair_plot_data(snapshots)
            if x_hv:
                ax_hv.fill_between(
                    x_hv, y_pes, y_opt,
                    color=color, alpha=0.15,
                )
                ax_hv.plot(
                    x_hv, y_opt,
                    color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.4, markersize=5,
                    label=f"{label} HV",
                )
                # Pessimistic line – thinner, same colour
                ax_hv.plot(
                    x_hv, y_pes,
                    color=color, linestyle="dotted", linewidth=0.7,
                    alpha=0.7,
                )

        # --- nR2 (right axis) ---
        if "ax_nr2" in twin_kwargs:
            x_nr, y_nr = nr2_plot_data(snapshots)
            if x_nr:
                twin_kwargs["ax_nr2"].plot(
                    x_nr, y_nr,
                    color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.4, markersize=5, alpha=0.55,
                    label=f"{label} nR2",
                )

    ax_hv.set_xlabel("Budget Used  (cumulative)")
    ax_hv.set_ylabel("Hypervolume  (HV)")
    ax_hv.grid(True, alpha=0.3)
    ax_hv.set_title(args.title)

    if "ax_nr2" in twin_kwargs:
        ax_nr2 = twin_kwargs["ax_nr2"]
        ax_nr2.set_ylabel("nR2  (lower = better)")

        # Merge legends from both axes.
        lines_a, labels_a = ax_hv.get_legend_handles_labels()
        lines_b, labels_b = ax_nr2.get_legend_handles_labels()
        ax_hv.legend(
            lines_a + lines_b,
            labels_a + labels_b,
            loc="upper center",
            bbox_to_anchor=(0.50, -0.12),
            ncol=min(4, len(labels_a) + len(labels_b)),
            fontsize=8,
        )
    else:
        ax_hv.legend(loc="best", fontsize=8)

    # Footnote.
    fig.text(
        0.01, 0.01,
        "HV_opt/HV_pes = optimistic/pessimistic hypervolume (band). "
        "nR2 measured against the final front (lower = closer to final).",
        fontsize=6.8, color="0.4",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"Saved trajectory plot: {out}")
    for path, label in zip(args.trajectories, args.labels):
        snaps = load_trajectory(path)
        final = snaps[-1] if snaps else {}
        print(
            f"  {label}: {len(snaps)} snapshots, "
            f"final HV={final.get('hypervolume','?'):.4f}, "
            f"final nR2={final.get('nr2','?'):.4f}"
        )


if __name__ == "__main__":
    main()
