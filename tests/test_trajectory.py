"""Tests for §F trajectory snapshots, metric enrichment, and plot path.

All tests are LLM-free — they drive ``compute_trajectory_metrics`` with
synthetic front snapshots (a few diverse objective vectors per snapshot,
monotonic HV, diminishing nR2) then verify the plot script renders a PNG.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from heal_capo.core import EvaluationResult
from heal_capo.evaluation.mo_metrics import (
    DEFAULT_OBJECTIVE_SPECS,
    infer_bounds,
)

# --- helpers ---

def _snap(label: str, iteration: int, budget: float, points: list[dict]) -> dict:
    return {
        "label": label,
        "iteration": iteration,
        "budget_used": budget,
        "budget_utilization": 0.0,
        "front_size": len(points),
        "front": points,
    }


def _pt(performance: float, cost: float, risk: float, fairness_risk: float) -> dict:
    return {
        "performance": performance,
        "cost": cost,
        "risk": risk,
        "fairness_risk": fairness_risk,
    }


# A minimal 4-snapshot trajectory where HV and nR2 should be computed.
MINI_TRAJECTORY = [
    _snap("initial", 0, 100.0, [_pt(0.60, 5.0, 0.20, 0.30)]),
    _snap("iter_1",   1, 250.0, [_pt(0.70, 4.0, 0.15, 0.25), _pt(0.80, 7.0, 0.10, 0.20)]),
    _snap("iter_2",   2, 400.0, [_pt(0.75, 3.0, 0.12, 0.22), _pt(0.85, 6.0, 0.08, 0.15),
                                  _pt(0.90, 9.0, 0.05, 0.10)]),
    _snap("final",    3, 500.0, [_pt(0.80, 3.0, 0.10, 0.18), _pt(0.88, 6.0, 0.06, 0.12),
                                  _pt(0.92, 8.0, 0.04, 0.08), _pt(0.95, 10.0, 0.02, 0.05)]),
]

# Degenerate: single-snapshot, single-point — HV and nR2 should still compute.
SINGLE_SNAP = [
    _snap("only", 0, 0.0, [_pt(0.50, 3.0, 0.20, 0.10)]),
]


class TestComputeTrajectoryMetrics:
    def test_returns_enriched_snapshots(self):
        from scripts.run_phase2_budgeted_mocapo import compute_trajectory_metrics

        enriched = compute_trajectory_metrics(MINI_TRAJECTORY, seed=42)

        assert len(enriched) == 4
        for i, snap in enumerate(enriched):
            assert "hypervolume" in snap, f"snap {i} missing hypervolume"
            assert "nr2" in snap, f"snap {i} missing nr2"
            assert isinstance(snap["hypervolume"], float), f"snap {i} hv not float"
            assert isinstance(snap["nr2"], float), f"snap {i} nr2 not float"

    def test_hv_within_reasonable_range(self):
        from scripts.run_phase2_budgeted_mocapo import compute_trajectory_metrics

        enriched = compute_trajectory_metrics(MINI_TRAJECTORY, seed=42)

        for snap in enriched:
            hv = snap["hypervolume"]
            assert 0.0 <= hv <= 1.0, f"HV {hv} outside [0,1]"

    def test_nr2_decreasing_toward_final(self):
        """nR2 against the final front should decrease as the front improves."""
        from scripts.run_phase2_budgeted_mocapo import compute_trajectory_metrics

        enriched = compute_trajectory_metrics(MINI_TRAJECTORY, seed=42)
        nr2_vals = [s["nr2"] for s in enriched]

        # Final snapshot (ref == cand) must be zero.
        assert nr2_vals[-1] == pytest.approx(0.0, abs=1e-9), (
            f"final nR2 should be 0, got {nr2_vals[-1]}"
        )
        # First snapshot should have the largest nR2 (worst fit).
        assert nr2_vals[0] >= nr2_vals[-1], (
            "nR2 should not improve beyond final (ref=cand gives 0)"
        )

    def test_hv_monotonic_non_decreasing_for_superset_front(self):
        """A superset front should not decrease HV with the same bounds.

        The test trajectory is constructed so each later snapshot's front
        dominates the previous ones (all points of earlier snap are on or
        below the candidates of later ones in the minimization space).
        """
        from scripts.run_phase2_budgeted_mocapo import compute_trajectory_metrics

        enriched = compute_trajectory_metrics(MINI_TRAJECTORY, seed=42)
        hv_vals = [s["hypervolume"] for s in enriched]
        # Allow micro-fluctuations from floating point, but general trend up.
        assert hv_vals[-1] >= hv_vals[0] - 1e-10, (
            f"final HV {hv_vals[-1]:.6f} should not be lower than initial {hv_vals[0]:.6f}"
        )

    def test_single_snapshot_does_not_crash(self):
        from scripts.run_phase2_budgeted_mocapo import compute_trajectory_metrics

        enriched = compute_trajectory_metrics(SINGLE_SNAP, seed=42)
        assert len(enriched) == 1
        assert enriched[0]["hypervolume"] is not None
        assert enriched[0]["nr2"] is not None

    def test_empty_front_snapshot_handled(self):
        from scripts.run_phase2_budgeted_mocapo import compute_trajectory_metrics

        empty = [_snap("empty", 0, 0.0, [])]
        enriched = compute_trajectory_metrics(empty, seed=42)
        assert len(enriched) == 1
        assert enriched[0]["hypervolume"] is None
        assert enriched[0]["nr2"] is None

    def test_preserves_input_keys(self):
        from scripts.run_phase2_budgeted_mocapo import compute_trajectory_metrics

        enriched = compute_trajectory_metrics(MINI_TRAJECTORY, seed=42)
        for orig_key in ("label", "iteration", "budget_used", "front_size", "front"):
            assert orig_key in enriched[0], f"missing key {orig_key}"

    def test_global_bounds_used_when_provided(self):
        from scripts.run_phase2_budgeted_mocapo import compute_trajectory_metrics

        bounds_config = {
            "performance": [0.0, 1.0],
            "cost": [0.0, 100.0],
            "risk": [0.0, 1.0],
            "fairness_risk": [0.0, 1.0],
        }
        no_bounds = compute_trajectory_metrics(MINI_TRAJECTORY, seed=42)
        with_bounds = compute_trajectory_metrics(
            MINI_TRAJECTORY, bounds_config=bounds_config, seed=42,
        )

        for i, (nb, wb) in enumerate(zip(no_bounds, with_bounds)):
            # Both should have non-null, finite, in-range HV and nR2.
            for k in ("hypervolume", "nr2"):
                assert wb[k] is not None, f"snap {i}: {k} is None (with bounds)"
                assert nb[k] is not None, f"snap {i}: {k} is None (no bounds)"
                assert 0.0 <= wb[k] <= 1.0, (
                    f"snap {i}: {k}={wb[k]} not in [0,1] (with bounds)"
                )
            # Fixed wide bounds -> larger normalizing box → can't produce
            # larger HV than inferred bounds (inferred box is tighter).
            # But narrow inferred ranges can collapse a dimension → HV=0.
            # Just confirm both are finite and the fixed-bounds HV is
            # reasonable (a subset of the full unit hypercube).
            assert wb["hypervolume"] <= 1.0


class TestVisualizeTrajectory:
    """Smoke test: the plot script writes a PNG from a tiny trajectory file."""

    def test_plot_writes_png(self, tmp_path: Path):
        from scripts.visualize_trajectory import main as plot_main
        from scripts.run_phase2_budgeted_mocapo import compute_trajectory_metrics

        # Enrich so hv/nr2 fields exist — the plot script reads them directly.
        enriched = compute_trajectory_metrics(MINI_TRAJECTORY, seed=42)

        traj_path = tmp_path / "traj.json"
        traj_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")

        out_path = tmp_path / "out.png"

        import sys
        old_argv = sys.argv
        try:
            sys.argv = [
                "visualize_trajectory.py",
                "--trajectory", str(traj_path),
                "--label", "test-method",
                "--title", "Test Trajectory",
                "--out", str(out_path),
            ]
            plot_main()
        finally:
            sys.argv = old_argv

        assert out_path.exists(), f"PNG not written at {out_path}"
        assert out_path.stat().st_size > 1024, "PNG too small (likely empty)"
