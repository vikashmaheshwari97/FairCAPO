#!/usr/bin/env bash
set -euo pipefail

# Download the CivilComments-WILDS data and place it where the FairCAPO loader
# resolves it by default: data/civilcomments/all_data_with_identities.csv
#
# Run this ONCE on the Rocket login node (has internet; no GPU needed). The CSV
# is ~1 GB and is NOT committed to git -- it is fetched here instead.
#
# Plain HF google/civil_comments is toxicity-only (no identity columns) and
# cannot drive the fairness axis; the WILDS/Jigsaw file with the 8 identity
# columns is required.

cd "$(dirname "$0")/../.."

DEST="data/civilcomments/all_data_with_identities.csv"

if [ -s "${DEST}" ]; then
  echo "Already present: ${DEST}"
  exit 0
fi

mkdir -p data/civilcomments

# The wilds package downloads civilcomments_v1.0/ into root_dir.
echo "Installing wilds (if needed) and downloading CivilComments-WILDS..."
python -m pip install --quiet wilds
python - <<'PY'
from wilds import get_dataset
# Downloads to data/civilcomments_v1.0/ (includes all_data_with_identities.csv).
get_dataset(dataset="civilcomments", download=True, root_dir="data")
PY

SRC="data/civilcomments_v1.0/all_data_with_identities.csv"
if [ ! -s "${SRC}" ]; then
  echo "WILDS download did not produce ${SRC}." >&2
  echo "If wilds is unavailable, place any CSV with columns comment_text, toxicity," >&2
  echo "split, and the 8 identity columns (male, female, LGBTQ, christian, muslim," >&2
  echo "other_religions, black, white) at ${DEST}." >&2
  exit 1
fi

cp "${SRC}" "${DEST}"
echo "Placed CivilComments CSV at ${DEST}"

# Sanity-check the columns the loader/probe builder require.
python - <<'PY'
import csv
required = {
    "comment_text", "toxicity", "split",
    "male", "female", "LGBTQ", "christian", "muslim",
    "other_religions", "black", "white",
}
with open("data/civilcomments/all_data_with_identities.csv", newline="", encoding="utf-8") as handle:
    header = next(csv.reader(handle))
missing = sorted(required - set(header))
if missing:
    raise SystemExit(f"CSV is missing required columns: {missing}")
print("Column check passed:", sorted(required))
PY

echo "Done. Next: build the fairness probe, then submit the pipeline."
