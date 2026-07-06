import csv

from experiments import datasets

IDENTITY_COLUMNS = datasets.CIVILCOMMENTS_IDENTITY_COLUMNS


def _write_csv(path, rows, *, include_split=True):
    """Write a minimal WILDS-style all_data_with_identities.csv."""
    fieldnames = ["comment_text", "toxicity", *IDENTITY_COLUMNS]
    if include_split:
        fieldnames.insert(1, "split")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full = {name: row.get(name, 0.0) for name in fieldnames}
            writer.writerow(full)


def _row(text, toxicity, split="train", **identities):
    row = {"comment_text": text, "toxicity": toxicity, "split": split}
    row.update(identities)
    return row


def test_load_civil_comments_binarizes_and_tags_identity(tmp_path):
    csv_path = tmp_path / "cc.csv"
    _write_csv(
        csv_path,
        [
            _row("Toxic about a group", 0.9, muslim=0.8),
            _row("Neutral mention", 0.1, female=0.7),
            _row("Clean no identity", 0.0),
            _row("Borderline nontoxic", 0.49, black=0.6),
        ],
    )

    split = datasets.load_civil_comments(
        dev_size=2,
        shots_size=1,
        test_size=1,
        seed=0,
        allow_smaller=True,
        stratified=False,
        data_path=str(csv_path),
    )

    all_examples = split.dev + split.shots + split.test
    assert split.name == "civil_comments"
    assert split.task_type == "classification"
    assert split.classes == ["non-toxic", "toxic"]
    assert all(ex.text.startswith("Comment: ") for ex in all_examples)
    # group is always a copy of identity, sensitive_attribute is identity.
    assert all(ex.metadata["group"] == ex.metadata["identity"] for ex in all_examples)
    assert all(ex.metadata["sensitive_attribute"] == "identity" for ex in all_examples)
    # Labels are the binarized toxicity.
    labels = {ex.label for ex in all_examples}
    assert labels <= {"toxic", "non-toxic"}


def test_civil_comments_label_threshold_and_identity_argmax(tmp_path):
    csv_path = tmp_path / "cc.csv"
    _write_csv(
        csv_path,
        [
            _row("exactly at threshold is toxic", 0.5, white=0.9),
            _row("below threshold is nontoxic", 0.499, christian=0.51, muslim=0.99),
            _row("no identity above 0.5", 0.2, male=0.49, black=0.3),
        ],
    )

    split = datasets.load_civil_comments(
        dev_size=3,
        shots_size=0,
        test_size=0,
        seed=0,
        allow_smaller=True,
        stratified=False,
        data_path=str(csv_path),
    )
    by_text = {ex.text: ex for ex in split.dev + split.shots + split.test}

    at = by_text["Comment: exactly at threshold is toxic"]
    assert at.label == "toxic"  # 0.5 >= 0.5
    assert at.metadata["identity"] == "white"

    below = by_text["Comment: below threshold is nontoxic"]
    assert below.label == "non-toxic"  # 0.499 < 0.5
    # muslim (0.99) beats christian (0.51) as the argmax present identity.
    assert below.metadata["identity"] == "muslim"

    none = by_text["Comment: no identity above 0.5"]
    assert none.metadata["identity"] == "none"


def test_civil_comments_caps_long_text(tmp_path):
    csv_path = tmp_path / "cc.csv"
    long_text = "x" * 5000
    _write_csv(csv_path, [_row(long_text, 0.1, female=0.9)])

    split = datasets.load_civil_comments(
        dev_size=1,
        shots_size=0,
        test_size=0,
        seed=0,
        allow_smaller=True,
        stratified=False,
        data_path=str(csv_path),
    )
    example = (split.dev + split.shots + split.test)[0]
    # "Comment: " prefix + capped body + " ..." suffix.
    assert len(example.text) < 5000
    assert example.text.endswith("...")


def test_civil_comments_filters_by_split_column(tmp_path):
    csv_path = tmp_path / "cc.csv"
    _write_csv(
        csv_path,
        [
            _row("train row", 0.1, male=0.9, split="train"),
            _row("test row", 0.1, male=0.9, split="test"),
        ],
    )

    test_split = datasets.load_civil_comments(
        dev_size=1,
        shots_size=0,
        test_size=0,
        seed=0,
        allow_smaller=True,
        stratified=False,
        split="test",
        data_path=str(csv_path),
    )
    texts = {ex.text for ex in test_split.dev + test_split.shots + test_split.test}
    assert texts == {"Comment: test row"}


def test_civil_comments_stratifies_by_label_and_identity(tmp_path):
    csv_path = tmp_path / "cc.csv"
    rows = []
    for identity in ("male", "female"):
        for toxicity in (0.9, 0.0):
            for idx in range(4):
                rows.append(
                    _row(f"{identity} {toxicity} {idx}", toxicity, **{identity: 0.9})
                )
    _write_csv(csv_path, rows)

    split = datasets.load_civil_comments(
        dev_size=4,
        shots_size=4,
        test_size=4,
        seed=0,
        allow_smaller=False,
        stratified=True,
        stratify_group_key="identity",
        data_path=str(csv_path),
    )

    for part in (split.dev, split.shots, split.test):
        cells = {(ex.label, ex.metadata["identity"]) for ex in part}
        assert cells == {
            ("toxic", "male"),
            ("toxic", "female"),
            ("non-toxic", "male"),
            ("non-toxic", "female"),
        }


def test_load_paper_dataset_forwards_civil_comments_args(monkeypatch):
    seen = {}

    def fake_loader(*args, **kwargs):
        seen.update(kwargs)
        return datasets.DatasetSplit(
            dev=[],
            shots=[],
            test=[],
            name="civil_comments",
            task_type="classification",
            classes=datasets.CIVILCOMMENTS_LABELS,
        )

    monkeypatch.setattr(datasets, "load_civil_comments", fake_loader)

    for alias in ("civil_comments", "civilcomments", "wilds_civilcomments"):
        seen.clear()
        datasets.load_paper_dataset(
            alias,
            dev_size=1,
            shots_size=1,
            test_size=1,
            dataset_split="test",
            data_path="/tmp/x.csv",
        )
        assert seen["split"] == "test"
        assert seen["stratify_group_key"] == "identity"
        assert seen["data_path"] == "/tmp/x.csv"


def test_civil_comments_helpers_expose_classes_and_description():
    assert datasets.get_dataset_classes("civil_comments") == ["non-toxic", "toxic"]
    assert datasets.get_dataset_classes("civilcomments") == ["non-toxic", "toxic"]
    description = datasets.get_task_description("wilds_civilcomments")
    assert "toxic" in description.lower()
