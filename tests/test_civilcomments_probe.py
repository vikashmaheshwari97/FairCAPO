import csv

import scripts.build_civilcomments_fairness_probe as probe_builder
from experiments import datasets

IDENTITY_COLUMNS = datasets.CIVILCOMMENTS_IDENTITY_COLUMNS


def _write_wilds_csv(path, rows):
    fieldnames = ["comment_text", "split", "toxicity", *IDENTITY_COLUMNS]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, 0.0) for name in fieldnames})


def _make_balanced_rows(per_cell=6):
    rows = []
    for identity in IDENTITY_COLUMNS:
        for toxicity in (0.9, 0.0):
            for idx in range(per_cell):
                rows.append(
                    {
                        "comment_text": f"{identity} {toxicity} {idx}",
                        "split": "train",
                        "toxicity": toxicity,
                        identity: 0.9,
                    }
                )
    return rows


def test_probe_is_balanced_across_identity_and_toxicity(tmp_path):
    csv_path = tmp_path / "cc.csv"
    rows = _make_balanced_rows(per_cell=6)
    # Add comments with no named identity -> must be dropped from the probe.
    rows.append({"comment_text": "no identity toxic", "split": "train", "toxicity": 0.9})
    rows.append({"comment_text": "no identity clean", "split": "train", "toxicity": 0.0})
    _write_wilds_csv(csv_path, rows)

    items = probe_builder.build_probe(
        csv_path=str(csv_path),
        split="train",
        seed=0,
        examples_per_group=5,
    )

    # 8 identities x 2 toxicity labels x 5 = 80 (the probe invariant).
    assert len(items) == 80

    counts = {}
    for item in items:
        meta = item["meta"]
        assert meta["identity"] != "none"
        assert meta["group"] == meta["identity"]
        counts[(meta["toxicity_label"], meta["identity"])] = (
            counts.get((meta["toxicity_label"], meta["identity"]), 0) + 1
        )
    assert len(counts) == 16
    assert all(count == 5 for count in counts.values())


def test_probe_excludes_search_source_indices(tmp_path):
    csv_path = tmp_path / "cc.csv"
    rows = _make_balanced_rows(per_cell=6)
    _write_wilds_csv(csv_path, rows)

    # Exclude every row index; the probe must then come back empty.
    all_indices = set(range(len(rows)))
    items = probe_builder.build_probe(
        csv_path=str(csv_path),
        split="train",
        seed=0,
        examples_per_group=5,
        exclude_source_indices=all_indices,
    )
    assert items == []


def test_probe_row_to_item_drops_no_identity_and_caps_text():
    long_text = "y" * 5000
    row = {"comment_text": long_text, "toxicity": 0.9, "male": 0.9}
    item = probe_builder._row_to_item(row, source_index=3, text_column="comment_text")
    assert item is not None
    assert item["meta"]["identity"] == "male"
    assert item["label"] == "toxic"
    assert item["text"].endswith("...")

    no_identity = {"comment_text": "hello", "toxicity": 0.9}
    assert (
        probe_builder._row_to_item(no_identity, source_index=0, text_column="comment_text")
        is None
    )
