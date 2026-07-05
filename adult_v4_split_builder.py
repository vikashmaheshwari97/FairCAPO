from __future__ import annotations

import argparse
import json

from heal_capo.adult_v4_manifest import build_fixed_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical fingerprinted Adult v4 split manifest.")
    parser.add_argument("--data", default="data/adult_semantic_v3.csv")
    parser.add_argument("--output", default="data/adult_v4_fixed_split_seed0.json")
    args = parser.parse_args()
    payload = build_fixed_manifest(args.data, args.output)
    print(json.dumps({
        "output": args.output,
        "version": payload["version"],
        "data_sha256": payload["data_sha256"],
        "row_count": payload["row_count"],
        "sizes": payload["sizes"],
    }, indent=2))


if __name__ == "__main__":
    main()
