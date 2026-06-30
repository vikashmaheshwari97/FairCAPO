from __future__ import annotations

from pathlib import Path

import yaml

from scripts.audit_hpc_configs import audit_repo


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_audit_flags_labelscore_and_inconsistent_bios_bounds(tmp_path):
    labels = [
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
    write_yaml(
        tmp_path / "configs" / "HPC_Config" / "bios_faircapo_labelscore.yaml",
        {
            "dataset": "bias_in_bios",
            "task_type": "classification",
            "labels": labels,
            "evaluation": {"classification_mode": "two_stage_label_scoring"},
            "fairness": {"mode": "group_accuracy_gap"},
            "llm": {"model_id": "mistralai/mistral-small-3.2"},
        },
    )
    write_yaml(
        tmp_path / "configs" / "HPC_Config" / "bios_experiment_table_labelscore.yaml",
        {
            "bounds": {"performance": [0.0, 1.0], "cost": [0.0, 24000.0]},
            "methods": [{"name": "FairCAPO label-score"}],
        },
    )
    hpc_dir = tmp_path / "scripts" / "hpc"
    hpc_dir.mkdir(parents=True)
    (hpc_dir / "run_bios_hpc.slurm").write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH --nodelist=firefly1",
                "#SBATCH --gpus-per-task=1",
                "#SBATCH --cpus-per-task=32",
            ]
        ),
        encoding="utf-8",
    )

    findings = audit_repo(tmp_path)

    messages = [finding.message for finding in findings]
    assert any("two_stage_label_scoring" in message for message in messages)
    assert any("cost bounds" in message for message in messages)


def test_audit_errors_on_bad_bios_labels(tmp_path):
    write_yaml(
        tmp_path / "configs" / "HPC_Config" / "bios_bad.yaml",
        {
            "dataset": "bias_in_bios",
            "task_type": "classification",
            "labels": ["teacher"],
        },
    )
    (tmp_path / "scripts" / "hpc").mkdir(parents=True)

    findings = audit_repo(tmp_path)

    assert any(
        finding.severity == "error" and "label list" in finding.message
        for finding in findings
    )
