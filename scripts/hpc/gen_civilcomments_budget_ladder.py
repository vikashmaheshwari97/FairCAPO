#!/usr/bin/env python3
"""Generate the CivilComments-WILDS + Gemma-2-27B budget-ladder (v2) experiment.

Motivation (2026-07-07). The 500k_v1 run produced a held-out ~0.69 across all
three methods because the *few-shot accuracy engine never ran*: the 500k budget
died in the initial population (fairness audit alone was ~38% of it), so the
surviving front was 0-shot terse prompts. MO-CAPO's own paper reaches 0.87-0.93
on binary classification (Subj) precisely via few-shot + evolved instructions at
a 7.5M-token budget. This generator rebuilds the CivilComments experiment with
paper-faithful knobs and a budget ladder (1M -> 5M -> 7.5M, the paper's budget)
so we can see where Gemma actually lands.

Self-hosting: the original ``*_500k_v1_gemma_*`` templates were DELETED once that
run was ruled out, so the generator now templates on the committed ``1000k_v2``
files (see TEMPLATE_RUNG) and produces every rung by:

  1. tag-substitution ``1000k_v2`` -> ``<rung>_v2`` (rewrites every output_dir,
     portfolio path, experiment_name, run tag and SLURM job-name, so rungs never
     collide; a no-op for the 1000k rung => regeneration is idempotent), and
  2. targeted line-edits of the search knobs ONLY (comments/paths preserved).
     The knob values below are ABSOLUTE, so re-applying them to the 1000k_v2
     template reproduces it byte-for-byte and only ``max_budget`` varies by rung:

       block_size ...................... 56 -> 30   (300/30 = 10 blocks; finer
                                                      racing => more candidates
                                                      admitted, matches paper)
       few_shot.max_few_shot_examples ..  3 ->  5   (paper k_max = 5; the primary
                                                      accuracy lever)
       few_shot.few_shot_probability ... 0.3 -> 0.5 (explore the few-shot arm more)
       fairness.eval_pairs ............. 80 -> 40   (Phase 2: halve the in-loop
                                                      audit so it stops starving
                                                      exploration; kept in-loop
                                                      for EVERY candidate, so the
                                                      FairCAPO steering story is
                                                      intact -- the reported
                                                      fairness is still the
                                                      held-out 1000-example gap)
       intensification.max_blocks_per_challenger  5 -> 8  (allow deeper racing
                                                      now that there are 10 blocks)
       budget.max_budget ............... 500000 -> <rung>

The held-out eval stays at test_size 1000 and is untouched except for tags.
BIOS files are NOT modified.

Run from the repo root:

    PYTHONPATH=. python scripts/hpc/gen_civilcomments_budget_ladder.py

Writes configs to configs/HPC_Config/ and pipeline/build scripts to scripts/hpc/.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs" / "HPC_Config"
HPC_DIR = REPO / "scripts" / "hpc"

# rung tag -> max_budget (float). Tags use the same "k" unit as the frozen
# 500k_v1 family (1000k/5000k/7500k) so filenames read consistently. The top
# rung is the paper's own 7.5M-token budget (Table 3 reaches 0.87-0.93 on binary
# classification there); 10M was dropped in favour of matching the paper.
RUNGS: dict[str, float] = {
    "1000k": 1_000_000.0,
    "5000k": 5_000_000.0,
    "7500k": 7_500_000.0,
}

# The original 500k_v1 CivilComments configs were DELETED (that run failed), so
# the generator is self-hosting: it templates on the committed {rung}_v2 configs
# at CONFIG_TEMPLATE_RUNG and substitutes the rung tag. The knob line-edits use
# absolute values, so regenerating the template rung reproduces it byte-for-byte;
# other rungs just swap the rung tag and budget.
TEMPLATE_RUNG = "1000k"

# Config files that carry the evolutionary SEARCH knobs (get the line-edits).
SEARCH_CONFIGS = (
    "civilcomments_faircapo_{rung}_v2_gemma_HPC.yaml",
    "civilcomments_ablation_{rung}_v2_gemma_HPC.yaml",
    "civilcomments_nsga2po_{rung}_v2_gemma_HPC.yaml",
)

# Config files that only need tag-substitution (eval / table / aggregate).
PASSTHROUGH_CONFIGS = (
    "civilcomments_eval_large_{rung}_v2_gemma_HPC.yaml",
    "civilcomments_eval_ablation_large_{rung}_v2_gemma_HPC.yaml",
    "civilcomments_eval_nsga_large_{rung}_v2_gemma_HPC.yaml",
    "civilcomments_experiment_table_{rung}_v2_gemma_large_HPC.yaml",
    "civilcomments_aggregate_{rung}_v2_gemma_large_HPC.yaml",
)

# Shell scripts (same {rung}-parameterized, self-hosting template scheme).
SHELL_SCRIPTS = (
    "submit_civilcomments_{rung}_v2_gemma_pipeline.sh",
    "build_civilcomments_{rung}_v2_gemma_large_outputs.sh",
)


def _apply_search_knobs(text: str, budget: float) -> str:
    """Line-edit the evolutionary search knobs, leaving everything else intact.

    Each substitution is anchored to the key's exact indentation so it matches
    at most one line and is a no-op when the key is absent (e.g. the fairness-off
    ablation has no ``eval_pairs``; NSGA has no ``block_size``).
    """
    edits = (
        (r"(?m)^block_size:[ \t]*.*$", "block_size: 30"),
        (r"(?m)^  max_few_shot_examples:[ \t]*.*$", "  max_few_shot_examples: 5"),
        (r"(?m)^  few_shot_probability:[ \t]*.*$", "  few_shot_probability: 0.5"),
        (r"(?m)^  eval_pairs:[ \t]*.*$", "  eval_pairs: 40"),
        (
            r"(?m)^  max_blocks_per_challenger:[ \t]*.*$",
            "  max_blocks_per_challenger: 8",
        ),
        (r"(?m)^  max_budget:[ \t]*.*$", f"  max_budget: {budget}"),
    )
    for pattern, replacement in edits:
        text = re.sub(pattern, replacement, text)
    return text


def _read_template(directory: Path, tmpl: str) -> str:
    """Read the committed template-rung file for a ``{rung}``-parameterized name."""
    return (directory / tmpl.format(rung=TEMPLATE_RUNG)).read_text(encoding="utf-8")


def _retag_rung(text: str, rung: str) -> str:
    """Swap the template rung tag ``1000k_v2`` -> ``<rung>_v2`` (path + display).

    Covers the path tag (``1000k_v2``), the display form (``1000k v2``) and the
    ``cc1000k-`` SLURM job-name prefix. A no-op when ``rung`` is the template
    rung, so regenerating the template rung is idempotent.
    """
    if rung == TEMPLATE_RUNG:
        return text
    text = text.replace(f"{TEMPLATE_RUNG}_v2", f"{rung}_v2")
    text = text.replace(f"{TEMPLATE_RUNG} v2", f"{rung} v2")
    text = text.replace(f"--job-name=cc{TEMPLATE_RUNG}-", f"--job-name=cc{rung}-")
    return text


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"  wrote {path.relative_to(REPO)}")


def generate() -> None:
    for rung, budget in RUNGS.items():
        print(f"[{rung}_v2] budget={budget:,.0f}")

        for tmpl in SEARCH_CONFIGS:
            # Template already carries the provenance header + v2 knobs; swap the
            # rung tag, then re-apply the knobs so this rung gets its own budget
            # (the other knob values are absolute -> a no-op on the template rung).
            text = _retag_rung(_read_template(CONFIG_DIR, tmpl), rung)
            text = _apply_search_knobs(text, budget)
            _write(CONFIG_DIR / tmpl.format(rung=rung), text)

        for tmpl in PASSTHROUGH_CONFIGS:
            text = _retag_rung(_read_template(CONFIG_DIR, tmpl), rung)
            _write(CONFIG_DIR / tmpl.format(rung=rung), text)

        for tmpl in SHELL_SCRIPTS:
            text = _retag_rung(_read_template(HPC_DIR, tmpl), rung)
            _write(HPC_DIR / tmpl.format(rung=rung), text)


if __name__ == "__main__":
    generate()
