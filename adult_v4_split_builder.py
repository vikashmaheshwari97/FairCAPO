from heal_capo.adult_v4_manifest import build_fixed_manifest


if __name__ == "__main__":
    print(build_fixed_manifest(
        "data/adult_semantic_v3.csv",
        "data/adult_v4_fixed_split_seed0.json",
    ))
