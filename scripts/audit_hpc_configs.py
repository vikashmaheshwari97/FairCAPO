from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BIOS_LABELS = [
    "accountant",
    "architect",
    "attorney",
    "chiropractor",
    "comedian",
    "composer",
    "dentist",
    "dietitian",
    "dj",
    "filmmaker",
    "interior_designer",
    "journalist",
    "model",
    "nurse",
    "painter",
    "paralegal",
    "pastor",
    "personal_trainer",
    "photographer",
    "physician",
    "poet",
    "professor",
    "psychologist",
    "rapper",
    "software_engineer",
    "surgeon",
    "teacher",
    "yoga_teacher",
]


CIVILCOMMENTS_LABELS = ["non-toxic", "toxic"]


@dataclass
class Finding:
    severity: str
    path: str
    message: str


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def add(
    findings: list[Finding],
    severity: str,
    path: Path | str,
    message: str,
) -> None:
    findings.append(Finding(severity=severity, path=str(path), message=message))


def labels_match(config: dict[str, Any]) -> bool:
    return list(config.get("labels") or []) == BIOS_LABELS


def civilcomments_labels_match(config: dict[str, Any]) -> bool:
    return list(config.get("labels") or []) == CIVILCOMMENTS_LABELS


def audit_bios_config(path: Path, config: dict[str, Any], findings: list[Finding]) -> None:
    dataset = str(config.get("dataset", "")).strip().lower()
    if dataset != "bias_in_bios":
        return

    lower_path = str(path).lower()
    legacy_bios_v1 = any(
        marker in lower_path
        for marker in (
            "bios_faircapo_500k_v1_hpc.yaml",
            "bios_ablation_500k_v1_hpc.yaml",
            "bios_nsga2po_500k_v1_hpc.yaml",
            "bios_eval_large_500k_v1_hpc.yaml",
            "bios_eval_ablation_large_500k_v1_hpc.yaml",
            "bios_eval_nsga_large_500k_v1_hpc.yaml",
        )
    )

    if config.get("task_type") != "classification":
        add(findings, "error", path, "Bias-in-Bios must use task_type: classification.")

    if not labels_match(config):
        add(findings, "error", path, "Bias-in-Bios label list differs from canonical 28-label order.")

    dev = config.get("dev") or {}
    if dev and str(dev.get("dataset", "")).strip().lower() != "bias_in_bios":
        add(findings, "error", path, "dev.dataset is not bias_in_bios.")
    if dev:
        dev_size = int(dev.get("dev_size", 0) or 0)
        shots_size = int(dev.get("shots_size", 0) or 0)
        test_size = int(dev.get("test_size", 0) or 0)
        if dev_size and dev_size < 280:
            add(
                findings,
                "warn",
                path,
                "BIOS dev_size is below 280; 28-label accuracy estimates will be noisy.",
            )
        if shots_size and shots_size < 112:
            add(
                findings,
                "warn",
                path,
                "BIOS shots_size is below 112; few-shot pool has less than ~4 examples per profession.",
            )
        if test_size and test_size < 500:
            add(
                findings,
                "warn",
                path,
                "BIOS dev.test_size is below 500; keep this as smoke-only evidence.",
            )
        if (
            not legacy_bios_v1
            and str(dev.get("stratify_group_key", "")).strip().lower() != "gender"
        ):
            add(
                findings,
                "error",
                path,
                "BIOS dev split must use stratify_group_key: gender.",
            )
    elif config.get("dataset_split") == "test":
        test_size = int(config.get("test_size", 0) or 0)
        if test_size and test_size < 1000:
            add(
                findings,
                "warn",
                path,
                "BIOS large-held-out test_size is below 1000; use only as a pilot.",
            )
        if (
            not legacy_bios_v1
            and str(config.get("stratify_group_key", "")).strip().lower() != "gender"
        ):
            add(
                findings,
                "error",
                path,
                "BIOS held-out eval must use stratify_group_key: gender.",
            )

    fairness = config.get("fairness") or {}
    fairness_mode = fairness.get("mode")
    fairness_in_loop = bool(fairness.get("in_loop", False))
    if fairness and fairness_mode not in {
        "group_accuracy_gap",
        "label_conditioned_group_accuracy_gap",
        "label_group_accuracy_gap",
    }:
        add(findings, "warn", path, "BIOS fairness mode is not a supported group-gap mode.")

    if "bios_ablation" in lower_path and "eval" not in lower_path and fairness_in_loop:
        add(
            findings,
            "error",
            path,
            "BIOS MO-CAPO ablation search must keep fairness.in_loop: false.",
        )

    if (
        not legacy_bios_v1
        and (
            ("bios_nsga2po" in lower_path and "labelscore" not in lower_path)
            or (
                "bios_eval" in lower_path
                and "large" in lower_path
                and "qwen" not in lower_path
                and "labelscore" not in lower_path
                and "fairshots" not in lower_path
                and "disambig" not in lower_path
                and "enhancedprompts" not in lower_path
            )
        )
    ):
        if fairness_in_loop and fairness_mode not in {
            "label_conditioned_group_accuracy_gap",
            "label_group_accuracy_gap",
        }:
            add(
                findings,
                "error",
                path,
                "BIOS comparison configs must use label-conditioned fairness for comparable HV/table rows.",
            )

    if (
        not legacy_bios_v1
        and "bios_nsga2po" in lower_path
        and "labelscore" not in lower_path
    ):
        budget = config.get("budget") or {}
        if str(budget.get("unit", "")).strip().lower() != "tokens":
            add(findings, "error", path, "BIOS NSGA-II-PO must use budget.unit: tokens.")
        if float(budget.get("max_budget", 0.0) or 0.0) != 500000.0:
            add(findings, "error", path, "BIOS NSGA-II-PO must use max_budget: 500000.0.")
        if bool(budget.get("allow_overspend", True)):
            add(findings, "error", path, "BIOS NSGA-II-PO must use allow_overspend: false.")

        few_shot = config.get("few_shot") or {}
        if not bool(few_shot.get("enabled", False)):
            add(findings, "error", path, "BIOS NSGA-II-PO must enable the shared few-shot pool.")
        if int(few_shot.get("pool_size", 0) or 0) != 112:
            add(findings, "error", path, "BIOS NSGA-II-PO few-shot pool_size must be 112.")
        if int(few_shot.get("max_few_shot_examples", 0) or 0) != 3:
            add(findings, "error", path, "BIOS NSGA-II-PO max_few_shot_examples must be 3.")

    fairness_data = str(fairness.get("fairness_data") or config.get("fairness_data") or "")
    if fairness_data and config.get("dataset_split") == "test":
        add(
            findings,
            "error",
            path,
            "BIOS held-out eval must not use fairness_data; use the joint held-out test split.",
        )
    if fairness_data and "search" not in Path(fairness_data).name.lower():
        add(
            findings,
            "warn",
            path,
            "BIOS in-loop fairness_data should be named as a search-only probe.",
        )
    if "fairness_bios_probe_search_seed0.jsonl" in fairness_data:
        eval_pairs = int(fairness.get("eval_pairs", 0) or 0)
        if eval_pairs != 80:
            add(
                findings,
                "error",
                path,
                "BIOS search fairness probe should use eval_pairs: 80.",
            )

    if fairness_mode in {
        "label_conditioned_group_accuracy_gap",
        "label_group_accuracy_gap",
    }:
        min_count = int(fairness.get("min_count_per_group", 1) or 1)
        if min_count < 5:
            add(
                findings,
                "error",
                path,
                "Label-conditioned BIOS fairness should use min_count_per_group >= 5.",
            )
        selection = config.get("selection") or {}
        min_perf = float(selection.get("min_performance_for_fairness", 0.0) or 0.0)
        gate_mode = str(selection.get("fairness_gate_mode", "continuous") or "continuous").lower()
        if gate_mode in {"hard", "clamp"} and min_perf > 0.0:
            add(
                findings,
                "error",
                path,
                "Label-conditioned BIOS fairness should not use a hard performance gate; use a continuous shortfall penalty.",
            )
        elif min_perf > 0.60:
            add(
                findings,
                "warn",
                path,
                "Label-conditioned BIOS fairness uses a high performance threshold; early exploratory runs should use a moderate continuous gate.",
            )

    evaluation = config.get("evaluation") or {}
    if evaluation.get("classification_mode") == "two_stage_label_scoring":
        add(
            findings,
            "warn",
            path,
            "two_stage_label_scoring is experimental and failed seed-0; do not use for main BIOS results.",
        )

    model_id = str((config.get("llm") or {}).get("model_id", ""))
    if "qwen" in model_id.lower():
        cost_cfg = config.get("cost") or {}
        if (
            float(cost_cfg.get("input_weight", 0.0) or 0.0) == 0.08
            and float(cost_cfg.get("output_weight", 0.0) or 0.0) == 0.32
        ):
            add(
                findings,
                "warn",
                path,
                "Qwen config reuses the Mistral cost profile; use a measured Qwen profile before main comparisons.",
            )
        add(
            findings,
            "warn",
            path,
            "Qwen BIOS seed-0 improved accuracy slightly but worsened fairness; keep out of main comparison.",
        )


def audit_civilcomments_config(
    path: Path, config: dict[str, Any], findings: list[Finding]
) -> None:
    dataset = str(config.get("dataset", "")).strip().lower()
    if dataset not in {"civil_comments", "civilcomments"}:
        return

    lower_path = str(path).lower()

    # Version-aware expected knobs. The frozen 500k_v1 family keeps the original
    # pins; the paper-faithful budget ladder (*_1m_v2 / *_5m_v2 / *_10m_v2,
    # generated by scripts/hpc/gen_civilcomments_budget_ladder.py) raises k_max to
    # 5, halves the in-loop probe to eval_pairs 40 (Phase 2 audit relief), and
    # sets max_budget per rung.
    is_v2 = "_v2" in lower_path
    expected_eval_pairs = 40 if is_v2 else 80
    expected_max_shots = 5 if is_v2 else 3
    if is_v2:
        rung_budgets = {
            "_1000k_v2": 1_000_000.0,
            "_5000k_v2": 5_000_000.0,
            "_7500k_v2": 7_500_000.0,
        }
        expected_budget = next(
            (value for tag, value in rung_budgets.items() if tag in lower_path),
            None,
        )
    else:
        expected_budget = 500000.0

    if config.get("task_type") != "classification":
        add(findings, "error", path, "CivilComments must use task_type: classification.")

    if not civilcomments_labels_match(config):
        add(
            findings,
            "error",
            path,
            "CivilComments label list must be exactly [non-toxic, toxic].",
        )

    dev = config.get("dev") or {}
    if dev and str(dev.get("dataset", "")).strip().lower() not in {
        "civil_comments",
        "civilcomments",
    }:
        add(findings, "error", path, "dev.dataset is not civil_comments.")
    if dev:
        if str(dev.get("stratify_group_key", "")).strip().lower() != "identity":
            add(
                findings,
                "error",
                path,
                "CivilComments dev split must use stratify_group_key: identity.",
            )
    elif config.get("dataset_split") == "test":
        if str(config.get("stratify_group_key", "")).strip().lower() != "identity":
            add(
                findings,
                "error",
                path,
                "CivilComments held-out eval must use stratify_group_key: identity.",
            )

    # Binary toxicity needs no chain-of-thought; the direct-label evaluator keeps
    # each eval cheap so the token budget explores a real front.
    evaluation = config.get("evaluation") or {}
    if str(evaluation.get("classification_mode", "generate")) != "generate":
        add(
            findings,
            "warn",
            path,
            "CivilComments configs are designed for classification_mode: generate "
            "(direct label); CoT will inflate cost and can starve the token budget.",
        )

    fairness = config.get("fairness") or {}
    fairness_mode = fairness.get("mode")
    fairness_in_loop = bool(fairness.get("in_loop", False))
    if fairness and fairness_mode not in {
        "group_accuracy_gap",
        "label_conditioned_group_accuracy_gap",
        "label_group_accuracy_gap",
    }:
        add(findings, "warn", path, "CivilComments fairness mode is not a supported group-gap mode.")

    if fairness and str(fairness.get("group_key", "")).strip().lower() != "identity":
        add(findings, "error", path, "CivilComments fairness.group_key must be identity.")

    if (
        "civilcomments_ablation" in lower_path
        and "eval" not in lower_path
        and fairness_in_loop
    ):
        add(
            findings,
            "error",
            path,
            "CivilComments MO-CAPO ablation search must keep fairness.in_loop: false.",
        )

    # Comparison configs (NSGA search + large held-out evals) must share the
    # label-conditioned fairness basis so their HV/table rows are comparable.
    if ("civilcomments_nsga2po" in lower_path) or (
        "civilcomments_eval" in lower_path and "large" in lower_path
    ):
        if fairness_in_loop and fairness_mode not in {
            "label_conditioned_group_accuracy_gap",
            "label_group_accuracy_gap",
        }:
            add(
                findings,
                "error",
                path,
                "CivilComments comparison configs must use label-conditioned fairness for comparable HV/table rows.",
            )

    if "civilcomments_nsga2po" in lower_path:
        budget = config.get("budget") or {}
        if str(budget.get("unit", "")).strip().lower() != "tokens":
            add(findings, "error", path, "CivilComments NSGA-II-PO must use budget.unit: tokens.")
        if expected_budget is not None and (
            float(budget.get("max_budget", 0.0) or 0.0) != expected_budget
        ):
            add(
                findings,
                "error",
                path,
                f"CivilComments NSGA-II-PO must use max_budget: {expected_budget}.",
            )
        if bool(budget.get("allow_overspend", True)):
            add(findings, "error", path, "CivilComments NSGA-II-PO must use allow_overspend: false.")

        few_shot = config.get("few_shot") or {}
        if not bool(few_shot.get("enabled", False)):
            add(findings, "error", path, "CivilComments NSGA-II-PO must enable the shared few-shot pool.")
        if int(few_shot.get("pool_size", 0) or 0) != 112:
            add(findings, "error", path, "CivilComments NSGA-II-PO few-shot pool_size must be 112.")
        if int(few_shot.get("max_few_shot_examples", 0) or 0) != expected_max_shots:
            add(
                findings,
                "error",
                path,
                f"CivilComments NSGA-II-PO max_few_shot_examples must be {expected_max_shots}.",
            )

    fairness_data = str(fairness.get("fairness_data") or config.get("fairness_data") or "")
    if fairness_data and config.get("dataset_split") == "test":
        add(
            findings,
            "error",
            path,
            "CivilComments held-out eval must not use fairness_data; use the joint held-out test split.",
        )
    if fairness_data and "search" not in Path(fairness_data).name.lower():
        add(
            findings,
            "warn",
            path,
            "CivilComments in-loop fairness_data should be named as a search-only probe.",
        )
    if "fairness_civilcomments_probe_search_seed0.jsonl" in fairness_data:
        eval_pairs = int(fairness.get("eval_pairs", 0) or 0)
        if eval_pairs != expected_eval_pairs:
            add(
                findings,
                "error",
                path,
                f"CivilComments search fairness probe should use eval_pairs: {expected_eval_pairs}.",
            )

    if fairness_mode in {
        "label_conditioned_group_accuracy_gap",
        "label_group_accuracy_gap",
    }:
        min_count = int(fairness.get("min_count_per_group", 1) or 1)
        if min_count < 5:
            add(
                findings,
                "error",
                path,
                "Label-conditioned CivilComments fairness should use min_count_per_group >= 5.",
            )
        selection = config.get("selection") or {}
        min_perf = float(selection.get("min_performance_for_fairness", 0.0) or 0.0)
        gate_mode = str(selection.get("fairness_gate_mode", "continuous") or "continuous").lower()
        if gate_mode in {"hard", "clamp"} and min_perf > 0.0:
            add(
                findings,
                "error",
                path,
                "Label-conditioned CivilComments fairness should not use a hard performance gate; use a continuous shortfall penalty.",
            )

    # Gemma runs must not silently reuse the measured Mistral cost profile.
    model_id = str((config.get("llm") or {}).get("model_id", ""))
    if "gemma" in model_id.lower():
        cost_cfg = config.get("cost") or {}
        if (
            float(cost_cfg.get("input_weight", 0.0) or 0.0) == 0.08
            and float(cost_cfg.get("output_weight", 0.0) or 0.0) == 0.32
        ):
            add(
                findings,
                "warn",
                path,
                "Gemma CivilComments config reuses the Mistral cost profile; set a measured Gemma profile.",
            )


def audit_table_config(path: Path, config: dict[str, Any], findings: list[Finding]) -> None:
    methods = config.get("methods") or []
    if not methods:
        return

    is_experiment_table = "experiment_table" in path.name
    bounds = config.get("bounds") or {}
    cost_bounds = bounds.get("cost")
    if (
        is_experiment_table
        and cost_bounds != [0.0, 12000.0]
        and "bios" in str(path).lower()
    ):
        add(
            findings,
            "warn",
            path,
            f"BIOS table cost bounds are {cost_bounds}; do not compare HV with 12000-bound tables.",
        )

    model = str(config.get("model", "")).lower()
    method_names = " ".join(str(m.get("name", "")) for m in methods).lower()
    if "qwen" in model and "mistral" in method_names:
        add(
            findings,
            "warn",
            path,
            "Qwen table mixes Qwen and Mistral rows; use only as diagnostic context.",
        )

    if "labelscore" in str(path).lower():
        add(
            findings,
            "warn",
            path,
            "Label-score table is diagnostic only; seed-0 label-score result underperformed BIOS v1.",
        )


def audit_prompt_pool(path: Path, config: dict[str, Any], findings: list[Finding]) -> None:
    is_bios_pool = path.name == "phase2_prompt_pool_bios.yaml"
    is_civilcomments_pool = path.name == "phase2_prompt_pool_civilcomments.yaml"
    if not (is_bios_pool or is_civilcomments_pool):
        return

    label = "BIOS" if is_bios_pool else "CivilComments"
    if is_bios_pool and not labels_match(config):
        add(findings, "error", path, "BIOS prompt-pool labels differ from canonical 28-label order.")
    if is_civilcomments_pool and not civilcomments_labels_match(config):
        add(
            findings,
            "error",
            path,
            "CivilComments prompt-pool labels must be exactly [non-toxic, toxic].",
        )

    prompts = config.get("prompt_pool") or []
    categories = {str(item.get("category", "")) for item in prompts if isinstance(item, dict)}
    required = {"cost_first", "accuracy_first", "fairness_first", "balanced"}
    missing = sorted(required - categories)
    if missing:
        add(findings, "error", path, f"{label} prompt pool missing categories: {missing}")

    cost_levels = {
        str(item.get("cost_level", "")).strip().lower()
        for item in prompts
        if isinstance(item, dict)
    }
    missing_cost_levels = sorted({"cheap", "medium", "expensive"} - cost_levels)
    if missing_cost_levels:
        add(
            findings,
            "error",
            path,
            f"{label} prompt pool missing explicit cost levels: {missing_cost_levels}",
        )

    shot_counts = set()
    for item in prompts:
        if not isinstance(item, dict):
            continue
        try:
            shot_counts.add(int(item.get("few_shot_count", 0) or 0))
        except (TypeError, ValueError):
            add(
                findings,
                "error",
                path,
                f"Prompt {item.get('id')} has non-integer few_shot_count.",
            )
    missing_shot_counts = sorted({0, 1, 3} - shot_counts)
    if missing_shot_counts:
        add(
            findings,
            "error",
            path,
            f"{label} prompt pool missing initial few-shot tiers: {missing_shot_counts}",
        )

    identity_terms = ("gender", "religion", "race", "sexual orientation", "identity")
    for item in prompts:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt", "")).lower()
        if item.get("category") != "fairness_first":
            continue
        if is_bios_pool and "gender" not in prompt:
            add(findings, "warn", path, f"Fairness prompt {item.get('id')} does not mention gender.")
        if is_civilcomments_pool and not any(term in prompt for term in identity_terms):
            add(
                findings,
                "warn",
                path,
                f"Fairness prompt {item.get('id')} does not mention an identity attribute.",
            )


def audit_slurm(path: Path, text: str, findings: list[Finding]) -> None:
    lowered = text.lower()
    if "run_bios" not in path.name and "bios" not in path.name:
        return

    if "#SBATCH --nodelist=firefly1".lower() not in lowered:
        add(findings, "warn", path, "BIOS SLURM script is not pinned to firefly1.")

    if "#SBATCH --gpus-per-task=1".lower() not in lowered:
        add(findings, "error", path, "BIOS SLURM script should request exactly one GPU per task.")

    if "#SBATCH --cpus-per-task=32".lower() not in lowered:
        add(findings, "warn", path, "BIOS SLURM script does not request 32 CPU cores.")


def audit_repo(root: Path) -> list[Finding]:
    findings: list[Finding] = []

    for path in sorted(root.joinpath("configs").rglob("*.yaml")):
        try:
            config = load_yaml(path)
        except Exception as exc:
            add(findings, "error", path, f"YAML parse failed: {exc}")
            continue

        audit_bios_config(path, config, findings)
        audit_civilcomments_config(path, config, findings)
        audit_table_config(path, config, findings)
        audit_prompt_pool(path, config, findings)

    for path in sorted(root.joinpath("scripts", "hpc").glob("*.slurm")):
        audit_slurm(path, path.read_text(encoding="utf-8"), findings)

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit HPC experiment configs before spending GPU time.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on warnings as well as errors.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = audit_repo(root)

    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warn"]

    if not findings:
        print("HPC config audit passed: no findings.")
        return 0

    for finding in findings:
        print(f"[{finding.severity}] {finding.path}: {finding.message}")

    print(f"Summary: {len(errors)} error(s), {len(warnings)} warning(s).")
    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
