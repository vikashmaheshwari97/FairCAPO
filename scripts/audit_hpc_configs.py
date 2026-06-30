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


def audit_bios_config(path: Path, config: dict[str, Any], findings: list[Finding]) -> None:
    dataset = str(config.get("dataset", "")).strip().lower()
    if dataset != "bias_in_bios":
        return

    if config.get("task_type") != "classification":
        add(findings, "error", path, "Bias-in-Bios must use task_type: classification.")

    if not labels_match(config):
        add(findings, "error", path, "Bias-in-Bios label list differs from canonical 28-label order.")

    dev = config.get("dev") or {}
    if dev and str(dev.get("dataset", "")).strip().lower() != "bias_in_bios":
        add(findings, "error", path, "dev.dataset is not bias_in_bios.")

    fairness = config.get("fairness") or {}
    if fairness and fairness.get("mode") != "group_accuracy_gap":
        add(findings, "warn", path, "BIOS fairness mode is not group_accuracy_gap.")

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
        add(
            findings,
            "warn",
            path,
            "Qwen BIOS seed-0 improved accuracy slightly but worsened fairness; keep out of main comparison.",
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
    if path.name != "phase2_prompt_pool_bios.yaml":
        return

    if not labels_match(config):
        add(findings, "error", path, "BIOS prompt-pool labels differ from canonical 28-label order.")

    prompts = config.get("prompt_pool") or []
    categories = {str(item.get("category", "")) for item in prompts if isinstance(item, dict)}
    required = {"cost_first", "accuracy_first", "fairness_first", "balanced"}
    missing = sorted(required - categories)
    if missing:
        add(findings, "error", path, f"BIOS prompt pool missing categories: {missing}")

    for item in prompts:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt", "")).lower()
        if item.get("category") == "fairness_first" and "gender" not in prompt:
            add(findings, "warn", path, f"Fairness prompt {item.get('id')} does not mention gender.")


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
