from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Sequence
import csv
import importlib
import json
import os
import random


@dataclass
class Example:
    text: str
    label: str
    metadata: Optional[dict[str, Any]] = None


@dataclass
class DatasetSplit:
    dev: list[Example]
    shots: list[Example]
    test: list[Example]
    name: str
    task_type: str
    classes: Optional[list[str]] = None
    task_description: Optional[str] = None


ADULT_INCOME_LABELS = ["<=50K", ">50K"]


def _load_hf_dataset(*args, **kwargs):
    """
    Import Hugging Face datasets safely.

    We use importlib instead of:
        from datasets import load_dataset

    because this project also has experiments/datasets.py, and PyCharm may
    confuse the local module with the external Hugging Face datasets package.
    """
    hf_datasets = importlib.import_module("datasets")
    return hf_datasets.load_dataset(*args, **kwargs)


def _normalize_label(label: Any) -> str:
    """
    Keep labels stable and comparable across datasets.
    """
    return str(label).strip()


def _normalize_lower_label(label: Any) -> str:
    """
    Lowercase label normalization for datasets whose labels are lowercase
    in our prompt/evaluation setup.
    """
    return str(label).strip().lower()


def _adult_dataset_alias(name: str) -> bool:
    return name.lower().strip() in {
        "adult",
        "adult_income",
        "adult-census-income",
        "adult_census_income",
    }


def _adult_default_paths() -> list[str]:
    return [
        os.environ.get("FAIRCAPO_ADULT_CSV", ""),
        "data/adult.csv",
        "data/adult/adult.csv",
        "datasets/adult.csv",
    ]


def _resolve_adult_csv_path(data_path: Optional[str] = None) -> str:
    candidates = [data_path] if data_path else []
    candidates.extend(_adult_default_paths())

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    tried = ", ".join(path for path in candidates if path)
    raise FileNotFoundError(
        "Adult CSV not found. Place adult.csv at data/adult.csv, set "
        "FAIRCAPO_ADULT_CSV, or pass dev.data_path in the config. "
        f"Tried: {tried}"
    )


def _adult_clean(value: Any) -> str:
    text = str(value).strip()
    if not text or text == "?":
        return "Unknown"
    return text.replace("-", " ").replace(".", " ")


def _normalize_adult_income(value: Any) -> str:
    text = str(value).strip().replace(".", "")
    if text in {">50K", ">50k"}:
        return ">50K"
    if text in {"<=50K", "<=50k"}:
        return "<=50K"
    raise ValueError(f"Unknown Adult income label: {value!r}")


def _render_adult_record(row: dict[str, Any]) -> str:
    """
    Render Adult tabular features as a compact record for LLM classification.

    The protected attributes (sex/race), target (income), sample weight
    (fnlwgt), and duplicate numeric education feature are deliberately omitted.
    They remain available only in metadata for fairness evaluation.
    """
    fields = [
        ("Age", row.get("age")),
        ("Work class", row.get("workclass")),
        ("Education", row.get("education")),
        ("Marital status", row.get("marital.status")),
        ("Occupation", row.get("occupation")),
        ("Relationship", row.get("relationship")),
        ("Capital gain", row.get("capital.gain")),
        ("Capital loss", row.get("capital.loss")),
        ("Hours per week", row.get("hours.per.week")),
        ("Native country", row.get("native.country")),
    ]
    lines = ["Person record:"]
    for label, value in fields:
        lines.append(f"{label}: {_adult_clean(value)}")
    return "\n".join(lines)


def load_toy_subjectivity() -> list[Example]:
    return [
        Example("The movie was released in 1999.", "objective", {"dataset": "toy_subjectivity"}),
        Example("This film is painfully boring and badly written.", "subjective", {"dataset": "toy_subjectivity"}),
        Example("The book contains twelve chapters.", "objective", {"dataset": "toy_subjectivity"}),
        Example("The acting is absolutely wonderful.", "subjective", {"dataset": "toy_subjectivity"}),
        Example("Paris is the capital of France.", "objective", {"dataset": "toy_subjectivity"}),
        Example("That was the most beautiful ending ever.", "subjective", {"dataset": "toy_subjectivity"}),
    ]


def _random_sample_split(
    examples: list[Example],
    name: str,
    task_type: str,
    classes: Optional[list[str]] = None,
    task_description: Optional[str] = None,
    dev_size: int = 300,
    shots_size: int = 100,
    test_size: int = 500,
    seed: int = 42,
    allow_smaller: bool = False,
) -> DatasetSplit:
    rng = random.Random(seed)
    examples = list(examples)
    rng.shuffle(examples)

    total_needed = dev_size + shots_size + test_size

    if len(examples) < total_needed and not allow_smaller:
        raise ValueError(
            f"Not enough examples for {name}. Need {total_needed}, got {len(examples)}. "
            f"Use smaller dev/shots/test sizes or set allow_smaller=True."
        )

    if allow_smaller and len(examples) < total_needed:
        n = len(examples)
        dev_size = min(dev_size, max(1, int(0.6 * n)))
        remaining = max(0, n - dev_size)
        shots_size = min(shots_size, max(0, int(0.2 * n)))
        test_size = min(test_size, max(0, remaining - shots_size))

    dev = examples[:dev_size]
    shots = examples[dev_size : dev_size + shots_size]
    test = examples[dev_size + shots_size : dev_size + shots_size + test_size]

    return DatasetSplit(
        dev=dev,
        shots=shots,
        test=test,
        name=name,
        task_type=task_type,
        classes=classes,
        task_description=task_description,
    )


def _allocate_counts(total: int, labels: list[str]) -> dict[str, int]:
    """
    Allocate approximately balanced counts across labels.

    Example:
      total=10, labels=[a,b] -> {a:5,b:5}
      total=5, labels=[a,b]  -> {a:3,b:2}
      total=10, labels=[a,b,c,d] -> {a:3,b:3,c:2,d:2}

    This keeps small splits more stable.
    """
    if total <= 0:
        return {label: 0 for label in labels}

    if not labels:
        return {}

    base = total // len(labels)
    remainder = total % len(labels)

    counts = {}
    for idx, label in enumerate(labels):
        counts[label] = base + (1 if idx < remainder else 0)

    return counts


def _take_from_label_buckets(
    buckets: dict[str, list[Example]],
    counts: dict[str, int],
    allow_smaller: bool,
    split_name: str,
) -> list[Example]:
    selected: list[Example] = []

    for label, requested_count in counts.items():
        available = buckets.get(label, [])

        if len(available) < requested_count and not allow_smaller:
            raise ValueError(
                f"Not enough examples for label '{label}' in {split_name}. "
                f"Need {requested_count}, got {len(available)}. "
                f"Use smaller split sizes or set allow_smaller=True."
            )

        actual_count = min(requested_count, len(available))
        selected.extend(available[:actual_count])
        del available[:actual_count]

    return selected


def _metadata_value(example: Example, key: str) -> str:
    metadata = example.metadata or {}
    return _normalize_lower_label(metadata.get(key, ""))


def _stratified_sample_split(
    examples: list[Example],
    name: str,
    task_type: str,
    classes: list[str],
    task_description: Optional[str] = None,
    dev_size: int = 300,
    shots_size: int = 100,
    test_size: int = 500,
    seed: int = 42,
    allow_smaller: bool = False,
    group_key: Optional[str] = None,
) -> DatasetSplit:
    """
    Label-balanced split for classification datasets. When ``group_key`` is
    provided, buckets are built from ``(label, protected_group)`` cells so
    fairness datasets preserve protected-group support within labels.

    This is important for small local runs such as:
      dev=10, shots=2, test=5

    Without stratification, a random slice can overrepresent one label and make
    optimizer results unstable.
    """
    rng = random.Random(seed)

    normalized_classes = [_normalize_label(c) for c in classes]
    class_lookup = {c.lower(): c for c in normalized_classes}

    grouped = bool(group_key)
    buckets: dict[Any, list[Example]] = defaultdict(list)
    keys_by_label: dict[str, list[Any]] = defaultdict(list)

    for ex in examples:
        label_key = _normalize_label(ex.label).lower()
        canonical_label = class_lookup.get(label_key, _normalize_label(ex.label))

        normalized_example = Example(
            text=ex.text,
            label=canonical_label,
            metadata=ex.metadata,
        )
        if grouped:
            group_value = _metadata_value(normalized_example, str(group_key))
            bucket_key = (canonical_label, group_value)
            if bucket_key not in keys_by_label[canonical_label]:
                keys_by_label[canonical_label].append(bucket_key)
        else:
            bucket_key = canonical_label
        buckets[bucket_key].append(normalized_example)

    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    total_needed = dev_size + shots_size + test_size

    if len(examples) < total_needed and not allow_smaller:
        raise ValueError(
            f"Not enough examples for {name}. Need {total_needed}, got {len(examples)}. "
            f"Use smaller dev/shots/test sizes or set allow_smaller=True."
        )

    if allow_smaller and len(examples) < total_needed:
        n = len(examples)
        dev_size = min(dev_size, max(1, int(0.6 * n)))
        remaining = max(0, n - dev_size)
        shots_size = min(shots_size, max(0, int(0.2 * n)))
        test_size = min(test_size, max(0, remaining - shots_size))

    if grouped:
        def allocate_group_counts(total: int) -> dict[Any, int]:
            counts: dict[Any, int] = {}
            per_label = _allocate_counts(total, normalized_classes)
            for label in normalized_classes:
                keys = sorted(keys_by_label.get(label, []))
                if not keys:
                    continue
                group_counts = _allocate_counts(per_label.get(label, 0), keys)
                counts.update(group_counts)
            return counts

        # Touch all labels even when a split is smaller than the number of
        # label/group cells; _allocate_counts handles the remainder
        # deterministically. This preserves label balance first, then group
        # balance within each label.
        dev_counts = allocate_group_counts(dev_size)
        shots_counts = allocate_group_counts(shots_size)
        test_counts = allocate_group_counts(test_size)
    else:
        dev_counts = _allocate_counts(dev_size, normalized_classes)
        shots_counts = _allocate_counts(shots_size, normalized_classes)
        test_counts = _allocate_counts(test_size, normalized_classes)

    dev = _take_from_label_buckets(
        buckets=buckets,
        counts=dev_counts,
        allow_smaller=allow_smaller,
        split_name=f"{name}/dev",
    )
    shots = _take_from_label_buckets(
        buckets=buckets,
        counts=shots_counts,
        allow_smaller=allow_smaller,
        split_name=f"{name}/shots",
    )
    test = _take_from_label_buckets(
        buckets=buckets,
        counts=test_counts,
        allow_smaller=allow_smaller,
        split_name=f"{name}/test",
    )

    rng.shuffle(dev)
    rng.shuffle(shots)
    rng.shuffle(test)

    return DatasetSplit(
        dev=dev,
        shots=shots,
        test=test,
        name=name,
        task_type=task_type,
        classes=normalized_classes,
        task_description=task_description,
    )


def _sample_split(
    examples: list[Example],
    name: str,
    task_type: str,
    classes: Optional[list[str]] = None,
    task_description: Optional[str] = None,
    dev_size: int = 300,
    shots_size: int = 100,
    test_size: int = 500,
    seed: int = 42,
    allow_smaller: bool = False,
    stratified: bool = True,
    stratify_group_key: Optional[str] = None,
) -> DatasetSplit:
    """
    Unified sampling wrapper.

    Classification datasets use stratified splits by default.
    Reasoning/code datasets use random splits.
    """
    if stratified and task_type == "classification" and classes:
        return _stratified_sample_split(
            examples=examples,
            name=name,
            task_type=task_type,
            classes=classes,
            task_description=task_description,
            dev_size=dev_size,
            shots_size=shots_size,
            test_size=test_size,
            seed=seed,
            allow_smaller=allow_smaller,
            group_key=stratify_group_key,
        )

    return _random_sample_split(
        examples=examples,
        name=name,
        task_type=task_type,
        classes=classes,
        task_description=task_description,
        dev_size=dev_size,
        shots_size=shots_size,
        test_size=test_size,
        seed=seed,
        allow_smaller=allow_smaller,
    )


def make_toy_subjectivity_split(
    seed: int = 42,
) -> DatasetSplit:
    examples = load_toy_subjectivity()

    return _sample_split(
        examples=examples,
        name="toy_subjectivity",
        task_type="classification",
        classes=["objective", "subjective"],
        task_description=(
            "The dataset contains sentences labeled as subjective or objective. "
            "The task is to classify each sentence as subjective or objective."
        ),
        dev_size=len(examples),
        shots_size=0,
        test_size=len(examples),
        seed=seed,
        allow_smaller=True,
        stratified=False,
    )


def load_subj(
    dev_size: int = 300,
    shots_size: int = 100,
    test_size: int = 500,
    seed: int = 42,
    allow_smaller: bool = False,
    stratified: bool = True,
) -> DatasetSplit:
    """
    Subj: binary subjectivity classification.
    Labels: objective / subjective
    """
    ds = _load_hf_dataset("SetFit/subj", split="train")

    examples: list[Example] = []
    for idx, row in enumerate(ds):
        label_text = row.get("label_text")

        if label_text is None:
            label = "subjective" if int(row["label"]) == 1 else "objective"
        else:
            label = _normalize_lower_label(label_text)

        examples.append(
            Example(
                text=str(row["text"]),
                label=label,
                metadata={
                    "dataset": "subj",
                    "source_index": idx,
                },
            )
        )

    return _sample_split(
        examples,
        name="subj",
        task_type="classification",
        classes=["objective", "subjective"],
        task_description=(
            "The dataset contains sentences labeled as subjective or objective. "
            "The task is to classify each sentence as subjective or objective."
        ),
        dev_size=dev_size,
        shots_size=shots_size,
        test_size=test_size,
        seed=seed,
        allow_smaller=allow_smaller,
        stratified=stratified,
    )


def load_ag_news(
    dev_size: int = 300,
    shots_size: int = 100,
    test_size: int = 500,
    seed: int = 42,
    allow_smaller: bool = False,
    stratified: bool = True,
) -> DatasetSplit:
    """
    AG News: 4-class topic classification.
    Labels: World / Sports / Business / Tech
    """
    ds = _load_hf_dataset("SetFit/ag_news", split="train")

    label_map = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Tech",
    }

    examples: list[Example] = []
    for idx, row in enumerate(ds):
        if "label_text" in row and row["label_text"] is not None:
            label = _normalize_label(row["label_text"])
        else:
            label = label_map[int(row["label"])]

        examples.append(
            Example(
                text=str(row["text"]),
                label=label,
                metadata={
                    "dataset": "ag_news",
                    "source_index": idx,
                },
            )
        )

    return _sample_split(
        examples,
        name="ag_news",
        task_type="classification",
        classes=["World", "Sports", "Business", "Tech"],
        task_description=(
            "The dataset contains news articles categorized into four classes: "
            "World, Sports, Business, and Tech. The task is to classify each news "
            "article into one of the four categories."
        ),
        dev_size=dev_size,
        shots_size=shots_size,
        test_size=test_size,
        seed=seed,
        allow_smaller=allow_smaller,
        stratified=stratified,
    )


def load_sst5(
    dev_size: int = 300,
    shots_size: int = 100,
    test_size: int = 500,
    seed: int = 42,
    allow_smaller: bool = False,
    stratified: bool = True,
) -> DatasetSplit:
    """
    SST-5: 5-class sentiment classification.
    Labels: terrible / bad / okay / good / great
    """
    ds = _load_hf_dataset("SetFit/sst5", split="train")

    label_map = {
        0: "terrible",
        1: "bad",
        2: "okay",
        3: "good",
        4: "great",
    }

    examples: list[Example] = []
    for idx, row in enumerate(ds):
        if "label_text" in row and row["label_text"] is not None:
            label = _normalize_lower_label(row["label_text"])
        else:
            label = label_map[int(row["label"])]

        examples.append(
            Example(
                text=str(row["text"]),
                label=label,
                metadata={
                    "dataset": "sst5",
                    "source_index": idx,
                },
            )
        )

    return _sample_split(
        examples,
        name="sst5",
        task_type="classification",
        classes=["terrible", "bad", "okay", "good", "great"],
        task_description=(
            "The dataset contains movie review sentences labeled with five sentiment "
            "classes: terrible, bad, okay, good, and great. The task is to classify "
            "each sentence into one of these sentiment categories."
        ),
        dev_size=dev_size,
        shots_size=shots_size,
        test_size=test_size,
        seed=seed,
        allow_smaller=allow_smaller,
        stratified=stratified,
    )


def load_gsm8k(
    dev_size: int = 300,
    shots_size: int = 100,
    test_size: int = 500,
    seed: int = 42,
    allow_smaller: bool = False,
) -> DatasetSplit:
    """
    GSM8K: grade-school math reasoning.
    Label is the final answer after ####.
    """
    ds = _load_hf_dataset("openai/gsm8k", name="main", split="train")

    examples: list[Example] = []
    for idx, row in enumerate(ds):
        answer = str(row["answer"])

        if "####" in answer:
            final_answer = answer.split("####")[-1].strip()
        else:
            final_answer = answer.strip()

        examples.append(
            Example(
                text=str(row["question"]),
                label=final_answer,
                metadata={
                    "dataset": "gsm8k",
                    "source_index": idx,
                    "full_answer": answer,
                },
            )
        )

    return _sample_split(
        examples,
        name="gsm8k",
        task_type="math_reasoning",
        classes=None,
        task_description=(
            "The dataset consists of elementary school math word problems that require "
            "multi-step reasoning to solve. The task is to solve each word problem and "
            "provide the final answer."
        ),
        dev_size=dev_size,
        shots_size=shots_size,
        test_size=test_size,
        seed=seed,
        allow_smaller=allow_smaller,
        stratified=False,
    )


def load_mbpp(
    dev_size: int = 300,
    shots_size: int = 100,
    test_size: int = 500,
    seed: int = 42,
    allow_smaller: bool = False,
) -> DatasetSplit:
    """
    MBPP: Python programming problems.
    Label is the reference code.
    Metadata stores tests.
    """
    ds = _load_hf_dataset("google-research-datasets/mbpp", split="train")

    examples: list[Example] = []
    for idx, row in enumerate(ds):
        tests = row.get("test_list", [])
        if tests is None:
            tests = []

        examples.append(
            Example(
                text=str(row["text"]),
                label=str(row["code"]),
                metadata={
                    "dataset": "mbpp",
                    "source_index": idx,
                    "task_id": row.get("task_id"),
                    "test_list": tests,
                },
            )
        )

    return _sample_split(
        examples,
        name="mbpp",
        task_type="code_generation",
        classes=None,
        task_description=(
            "The dataset contains Python programming problems. The task is to generate "
            "a correct Python function or program that satisfies the problem statement "
            "and passes the provided tests."
        ),
        dev_size=dev_size,
        shots_size=shots_size,
        test_size=test_size,
        seed=seed,
        allow_smaller=allow_smaller,
        stratified=False,
    )


BIOS_PROFESSION_LABELS = [
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


def _bios_profession_label(value: Any) -> str:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return _normalize_label(value)
    if 0 <= idx < len(BIOS_PROFESSION_LABELS):
        return BIOS_PROFESSION_LABELS[idx]
    return str(idx)


def _bios_gender_label(value: Any) -> str:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return _normalize_lower_label(value)
    return "male" if idx == 0 else "female" if idx == 1 else str(idx)


def load_bias_in_bios(
    dev_size: int = 300,
    shots_size: int = 100,
    test_size: int = 500,
    seed: int = 42,
    allow_smaller: bool = False,
    stratified: bool = True,
    stratify_group_key: Optional[str] = "gender",
    split: str = "train",
) -> DatasetSplit:
    """
    Bias-in-Bios: biography text -> profession classification.

    The Hugging Face version exposes ``hard_text`` as the input biography,
    ``profession`` as a 28-way occupation label, and ``gender`` as the binary
    protected attribute. We preserve gender in metadata so the group-fairness
    evaluator can compute group accuracy/equal-opportunity style gaps.
    """
    ds = _load_hf_dataset("LabHC/bias_in_bios", split=split)

    examples: list[Example] = []
    for idx, row in enumerate(ds):
        text = str(row.get("hard_text", "")).strip()
        if not text:
            continue

        profession = _bios_profession_label(row.get("profession"))
        gender = _bios_gender_label(row.get("gender"))

        examples.append(
            Example(
                text=f"Biography: {text}",
                label=profession,
                metadata={
                    "dataset": "bias_in_bios",
                    "source_index": idx,
                    "profession": profession,
                    "gender": gender,
                    "group": gender,
                    "sensitive_attribute": "gender",
                },
            )
        )

    return _sample_split(
        examples,
        name="bias_in_bios",
        task_type="classification",
        classes=BIOS_PROFESSION_LABELS,
        task_description=(
            "The dataset contains short biographies labeled with a professional "
            "occupation. The task is to classify each biography into exactly one "
            "occupation label. Gender is a protected attribute used only for "
            "fairness evaluation, not as a target label."
        ),
        dev_size=dev_size,
        shots_size=shots_size,
        test_size=test_size,
        seed=seed,
        allow_smaller=allow_smaller,
        stratified=stratified,
        stratify_group_key=stratify_group_key if stratified else None,
    )


def load_adult_income(
    dev_size: int = 400,
    shots_size: int = 100,
    test_size: int = 1000,
    seed: int = 42,
    allow_smaller: bool = False,
    stratified: bool = True,
    stratify_group_key: Optional[str] = "sex",
    data_path: Optional[str] = None,
) -> DatasetSplit:
    """
    Adult income: tabular census-style rows rendered as text records.

    The prompt text intentionally excludes the target ``income``, sampling
    weight ``fnlwgt``, duplicate ``education.num``, and protected attributes
    ``sex``/``race``. Protected attributes remain in metadata so the evaluator
    can compute equalized-odds and other group fairness metrics.
    """
    csv_path = _resolve_adult_csv_path(data_path)

    examples: list[Example] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "age",
            "workclass",
            "education",
            "marital.status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital.gain",
            "capital.loss",
            "hours.per.week",
            "native.country",
            "income",
        }
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Adult CSV is missing required columns: {missing}")

        for idx, row in enumerate(reader):
            label = _normalize_adult_income(row.get("income"))
            sex = _adult_clean(row.get("sex"))
            race = _adult_clean(row.get("race"))

            examples.append(
                Example(
                    text=_render_adult_record(row),
                    label=label,
                    metadata={
                        "dataset": "adult",
                        "source_index": idx,
                        "sex": sex,
                        "race": race,
                        "group": sex,
                        "sensitive_attribute": "sex",
                        "income": label,
                    },
                )
            )

    return _sample_split(
        examples,
        name="adult",
        task_type="classification",
        classes=ADULT_INCOME_LABELS,
        task_description=(
            "The dataset contains tabular census-style person records rendered "
            "as text. The task is to predict whether annual income is <=50K or "
            ">50K from employment, education, working-hours, and financial "
            "evidence. Sex and race are protected attributes used only for "
            "fairness evaluation."
        ),
        dev_size=dev_size,
        shots_size=shots_size,
        test_size=test_size,
        seed=seed,
        allow_smaller=allow_smaller,
        stratified=stratified,
        stratify_group_key=stratify_group_key if stratified else None,
    )


# BBQ (Bias Benchmark for QA, Parrish et al. 2021).
#
# Three demographic categories used for FairCAPO's fairness showcase. The exact
# folder/category names match the official nyu-mll/BBQ data files
# (data/<Category>.jsonl).
BBQ_DEFAULT_CATEGORIES = ("Gender_identity", "Race_ethnicity", "Religion")

# Canonical option letters for the 3-way BBQ multiple-choice format.
_BBQ_OPTION_LETTERS = ("A", "B", "C")


def render_bbq_text(context: str, question: str, options: Sequence[str]) -> str:
    """Render a BBQ item as the context + question + lettered options block."""
    opts = list(options) + ["", "", ""]
    return (
        f"{str(context).strip()}\n"
        f"Question: {str(question).strip()}\n"
        f"Options:\n"
        f"(A) {opts[0]}\n"
        f"(B) {opts[1]}\n"
        f"(C) {opts[2]}"
    )


def _as_dict(value: Any) -> dict:
    """BBQ nested fields may arrive as dicts or JSON strings; normalize to dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def _read_bbq_jsonl(path: str, category: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("category", category)
            rows.append(row)
    return rows


def _load_bbq_rows(categories: tuple[str, ...], data_dir: str) -> list[dict]:
    """
    Load raw BBQ rows for the requested categories.

    Resolution order, per category:
      1. Local JSONL at ``{data_dir}/{Category}.jsonl`` (official nyu-mll/BBQ format).
      2. HuggingFace mirror ``heegyu/bbq`` (filtered by category).

    A clear error is raised if neither source is available, telling the user how
    to obtain the data.
    """
    rows: list[dict] = []
    missing: list[str] = []

    for category in categories:
        local_path = os.path.join(data_dir, f"{category}.jsonl")
        if os.path.exists(local_path):
            rows.extend(_read_bbq_jsonl(local_path, category))
            continue
        missing.append(category)

    if not missing:
        return rows

    # Fall back to a HuggingFace mirror for the categories without local files.
    try:
        hf = _load_hf_dataset("heegyu/bbq", split="test")
    except Exception as exc:  # pragma: no cover - network/availability dependent
        raise FileNotFoundError(
            "BBQ data not found. Provide local JSONL files at "
            f"'{data_dir}/<Category>.jsonl' (clone https://github.com/nyu-mll/BBQ "
            "and copy data/*.jsonl), or ensure the 'heegyu/bbq' HuggingFace mirror "
            f"is reachable. Missing categories: {missing}. Underlying error: {exc}"
        ) from exc

    wanted = {c.lower() for c in missing}
    for row in hf:
        cat = str(row.get("category", "")).strip()
        if cat.lower() in wanted:
            rows.append(dict(row))

    if not rows:
        raise FileNotFoundError(
            f"No BBQ rows loaded for categories {categories}. Checked '{data_dir}' "
            "and the 'heegyu/bbq' mirror."
        )

    return rows


def load_bbq(
    dev_size: int = 300,
    shots_size: int = 100,
    test_size: int = 500,
    seed: int = 42,
    allow_smaller: bool = False,
    categories: tuple[str, ...] = BBQ_DEFAULT_CATEGORIES,
    data_dir: str = os.path.join("data", "bbq"),
) -> DatasetSplit:
    """
    BBQ: Bias Benchmark for QA — 3-option multiple-choice questions designed to
    surface social bias across demographic groups.

    Each rendered example presents a context, a question, and three options
    (A/B/C); the label is the correct option letter. All BBQ-specific metadata
    needed to compute the canonical bias score (sAMB/sDIS) is preserved in
    ``Example.metadata`` for the fairness path: context_condition,
    question_polarity, answer_info, stereotyped_groups, the raw options, and the
    correct option index.
    """
    raw_rows = _load_bbq_rows(categories, data_dir)

    examples: list[Example] = []
    for idx, row in enumerate(raw_rows):
        context = str(row.get("context", "")).strip()
        question = str(row.get("question", "")).strip()
        options = [
            str(row.get("ans0", "")),
            str(row.get("ans1", "")),
            str(row.get("ans2", "")),
        ]
        label_idx = int(row.get("label", 0))
        label_idx = label_idx if label_idx in (0, 1, 2) else 0

        rendered = render_bbq_text(context, question, options)

        additional = _as_dict(row.get("additional_metadata"))

        examples.append(
            Example(
                text=rendered,
                label=_BBQ_OPTION_LETTERS[label_idx],
                metadata={
                    "dataset": "bbq",
                    "source_index": idx,
                    "category": str(row.get("category", "")),
                    "context_condition": str(row.get("context_condition", "")),
                    "question_polarity": str(row.get("question_polarity", "")),
                    "answer_info": _as_dict(row.get("answer_info")),
                    "stereotyped_groups": additional.get("stereotyped_groups", []),
                    "options": options,
                    "label_idx": label_idx,
                    "example_id": row.get("example_id"),
                },
            )
        )

    return _sample_split(
        examples,
        name="bbq",
        task_type="multiple_choice",
        classes=None,
        task_description=(
            "The dataset contains multiple-choice questions, each with a short context, "
            "a question, and three options (A, B, C). One option is always an "
            "'unknown'/'cannot be determined' choice. When the context does not provide "
            "enough information to answer, the correct choice is the unknown option; "
            "otherwise the correct choice is the option supported by the context. The "
            "task is to select the single correct option."
        ),
        dev_size=dev_size,
        shots_size=shots_size,
        test_size=test_size,
        seed=seed,
        allow_smaller=allow_smaller,
        stratified=False,
    )


def load_paper_dataset(
    name: str,
    dev_size: int = 300,
    shots_size: int = 100,
    test_size: int = 500,
    seed: int = 42,
    allow_smaller: bool = False,
    stratified: bool = True,
    dataset_split: Optional[str] = None,
    stratify_group_key: Optional[str] = None,
    data_path: Optional[str] = None,
) -> DatasetSplit:
    """
    Unified loader for datasets used in MO-CAPO / Promptolution / EvoPrompt papers.

    MO-CAPO primary datasets:
      - AG News
      - Subj
      - GSM8K
      - MBPP

    Promptolution sanity dataset:
      - SST-5

    EvoPrompt overlaps:
      - AG News
      - Subj
      - SST-5
    """
    normalized_name = name.lower().strip()

    if normalized_name in {"toy", "toy_subjectivity"}:
        return make_toy_subjectivity_split(seed=seed)

    if normalized_name in {"subj", "subjectivity"}:
        return load_subj(dev_size, shots_size, test_size, seed, allow_smaller, stratified=stratified)

    if normalized_name in {"ag_news", "agnews", "ag-news", "ag's news", "ags_news"}:
        return load_ag_news(dev_size, shots_size, test_size, seed, allow_smaller, stratified=stratified)

    if normalized_name in {"sst5", "sst-5", "sst_5"}:
        return load_sst5(dev_size, shots_size, test_size, seed, allow_smaller, stratified=stratified)

    if normalized_name in {"gsm8k", "gsm"}:
        return load_gsm8k(dev_size, shots_size, test_size, seed, allow_smaller)

    if normalized_name in {"mbpp"}:
        return load_mbpp(dev_size, shots_size, test_size, seed, allow_smaller)

    if normalized_name in {"bbq"}:
        return load_bbq(dev_size, shots_size, test_size, seed, allow_smaller)

    if normalized_name in {"bias_in_bios", "bias-in-bios", "bios", "biosbias"}:
        return load_bias_in_bios(
            dev_size,
            shots_size,
            test_size,
            seed,
            allow_smaller,
            stratified=stratified,
            split=dataset_split or "train",
            stratify_group_key=stratify_group_key or "gender",
        )

    if _adult_dataset_alias(normalized_name):
        return load_adult_income(
            dev_size=dev_size,
            shots_size=shots_size,
            test_size=test_size,
            seed=seed,
            allow_smaller=allow_smaller,
            stratified=stratified,
            stratify_group_key=stratify_group_key or "sex",
            data_path=data_path,
        )

    raise ValueError(
        f"Unknown paper dataset: {name}. "
        "Supported: toy_subjectivity, subj, ag_news, sst5, gsm8k, mbpp, "
        "bbq, bias_in_bios, adult."
    )


def get_dataset_classes(name: str) -> Optional[list[str]]:
    """
    Lightweight helper for configs/runners that need class labels.
    """
    normalized_name = name.lower().strip()

    if normalized_name in {"toy", "toy_subjectivity", "subj", "subjectivity"}:
        return ["objective", "subjective"]

    if normalized_name in {"ag_news", "agnews", "ag-news", "ag's news", "ags_news"}:
        return ["World", "Sports", "Business", "Tech"]

    if normalized_name in {"sst5", "sst-5", "sst_5"}:
        return ["terrible", "bad", "okay", "good", "great"]

    if normalized_name in {"bias_in_bios", "bias-in-bios", "bios", "biosbias"}:
        return list(BIOS_PROFESSION_LABELS)

    if _adult_dataset_alias(normalized_name):
        return list(ADULT_INCOME_LABELS)

    # BBQ is multiple-choice (variable answer text per item) -> no fixed class set,
    # like GSM8K/MBPP.
    if normalized_name in {"bbq"}:
        return None

    return None


def get_task_description(name: str) -> str:
    """
    Lightweight helper for configs/runners that need task descriptions without loading data.
    """
    normalized_name = name.lower().strip()

    if normalized_name in {"toy", "toy_subjectivity", "subj", "subjectivity"}:
        return (
            "The dataset contains sentences labeled as subjective or objective. "
            "The task is to classify each sentence as subjective or objective."
        )

    if normalized_name in {"ag_news", "agnews", "ag-news", "ag's news", "ags_news"}:
        return (
            "The dataset contains news articles categorized into four classes: "
            "World, Sports, Business, and Tech. The task is to classify each news "
            "article into one of the four categories."
        )

    if normalized_name in {"sst5", "sst-5", "sst_5"}:
        return (
            "The dataset contains movie review sentences labeled with five sentiment "
            "classes: terrible, bad, okay, good, and great. The task is to classify "
            "each sentence into one of these sentiment categories."
        )

    if normalized_name in {"gsm8k", "gsm"}:
        return (
            "The dataset consists of elementary school math word problems that require "
            "multi-step reasoning to solve. The task is to solve each word problem and "
            "provide the final answer."
        )

    if normalized_name in {"mbpp"}:
        return (
            "The dataset contains Python programming problems. The task is to generate "
            "a correct Python function or program that satisfies the problem statement "
            "and passes the provided tests."
        )

    if normalized_name in {"bbq"}:
        return (
            "The dataset contains multiple-choice questions, each with a short context, "
            "a question, and three options (A, B, C). One option is always an "
            "'unknown'/'cannot be determined' choice. When the context does not provide "
            "enough information to answer, the correct choice is the unknown option; "
            "otherwise the correct choice is the option supported by the context. The "
            "task is to select the single correct option."
        )

    if normalized_name in {"bias_in_bios", "bias-in-bios", "bios", "biosbias"}:
        return (
            "The dataset contains short biographies labeled with a professional "
            "occupation. The task is to classify each biography into exactly one "
            "occupation label. Gender is a protected attribute used only for "
            "fairness evaluation, not as a target label."
        )

    if _adult_dataset_alias(normalized_name):
        return (
            "The dataset contains tabular census-style person records rendered "
            "as text. The task is to predict whether annual income is <=50K or "
            ">50K. Sex and race are protected attributes used only for fairness "
            "evaluation, not as model input features in the main setting."
        )

    raise ValueError(f"Unknown dataset for task description: {name}")
