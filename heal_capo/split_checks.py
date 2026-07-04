from __future__ import annotations


def checked_sets(manifest: dict, names=("dev", "shots", "test")):
    sizes = manifest.get("sizes")
    if not isinstance(sizes, dict):
        raise ValueError("Missing split sizes.")
    sets = {}
    for name in names:
        values = manifest.get(f"{name}_source_indices")
        if not isinstance(values, list):
            raise ValueError(f"Missing split: {name}")
        indices = [int(value) for value in values]
        if len(indices) != len(set(indices)) or any(value < 0 for value in indices):
            raise ValueError(f"Invalid split: {name}")
        if len(indices) != int(sizes.get(name, -1)):
            raise ValueError(f"Wrong split size: {name}")
        sets[name] = set(indices)
    if sets["dev"] & sets["shots"] or sets["dev"] & sets["test"] or sets["shots"] & sets["test"]:
        raise ValueError("Split overlap detected.")
    return sets
