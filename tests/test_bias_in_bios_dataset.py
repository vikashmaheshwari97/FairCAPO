from experiments import datasets


def test_load_bias_in_bios_preserves_profession_and_gender(monkeypatch):
    rows = [
        {"hard_text": "She teaches algebra.", "profession": 26, "gender": 1},
        {"hard_text": "He builds software.", "profession": 24, "gender": 0},
        {"hard_text": "She treats patients.", "profession": 19, "gender": 1},
        {"hard_text": "He argues cases.", "profession": 2, "gender": 0},
    ]

    monkeypatch.setattr(datasets, "_load_hf_dataset", lambda *args, **kwargs: rows)

    split = datasets.load_bias_in_bios(
        dev_size=2,
        shots_size=1,
        test_size=1,
        seed=0,
        allow_smaller=False,
        stratified=False,
    )

    all_examples = split.dev + split.shots + split.test
    assert split.name == "bias_in_bios"
    assert split.task_type == "classification"
    assert "software_engineer" in split.classes
    assert {ex.metadata["gender"] for ex in all_examples} == {"female", "male"}
    assert all(ex.metadata["group"] == ex.metadata["gender"] for ex in all_examples)
    assert all(ex.text.startswith("Biography: ") for ex in all_examples)


def test_bias_in_bios_helpers_expose_classes_and_description():
    assert "physician" in datasets.get_dataset_classes("bias_in_bios")
    description = datasets.get_task_description("bios")
    assert "occupation" in description


def test_load_paper_dataset_passes_bias_in_bios_split(monkeypatch):
    seen = {}

    def fake_load_bias_in_bios(*args, **kwargs):
        seen["split"] = kwargs.get("split")
        return datasets.DatasetSplit(
            dev=[],
            shots=[],
            test=[],
            name="bias_in_bios",
            task_type="classification",
            classes=datasets.BIOS_PROFESSION_LABELS,
        )

    monkeypatch.setattr(datasets, "load_bias_in_bios", fake_load_bias_in_bios)

    datasets.load_paper_dataset(
        "bias_in_bios",
        dev_size=1,
        shots_size=1,
        test_size=1,
        dataset_split="test",
    )

    assert seen["split"] == "test"
