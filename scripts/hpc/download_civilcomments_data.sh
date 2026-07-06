#!/usr/bin/env bash
set -euo pipefail

# Download the CivilComments-WILDS data and place it where the FairCAPO loader
# resolves it by default: data/civilcomments/all_data_with_identities.csv
#
# Run this ONCE on the Rocket login node (has internet; no GPU needed). The CSV
# is ~1 GB and is NOT committed to git -- it is fetched here instead.
#
# NOTE: the WILDS default host (worksheets.codalab.org) currently serves an
# EXPIRED TLS certificate, so the `wilds` package download fails with
# CERTIFICATE_VERIFY_FAILED. This script fetches the archive directly with
# certificate verification disabled (-k). That is safe here: the payload is
# public, read-only benchmark data (and is column-checked below).
#
# Plain HF google/civil_comments is toxicity-only (no identity columns) and
# cannot drive the fairness axis; the WILDS/Jigsaw file with the 8 identity
# columns is required.

cd "$(dirname "$0")/../.."

DEST="data/civilcomments/all_data_with_identities.csv"
VDIR="data/civilcomments_v1.0"
ARCHIVE="${VDIR}/archive.tar.gz"
URL="https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/"

if [ -s "${DEST}" ]; then
  echo "Already present: ${DEST}"
  exit 0
fi

mkdir -p data/civilcomments "${VDIR}"

# Reuse an already-extracted CSV if a prior (partial) run left one behind.
FOUND="$(find data -maxdepth 3 -name all_data_with_identities.csv 2>/dev/null | head -1 || true)"

if [ -z "${FOUND}" ]; then
  echo "Downloading CivilComments-WILDS archive (codalab cert expired -> --insecure)..."
  rm -f "${ARCHIVE}"
  if command -v curl >/dev/null 2>&1; then
    curl -kL --fail --retry 3 -o "${ARCHIVE}" "${URL}"
  else
    wget --no-check-certificate -O "${ARCHIVE}" "${URL}"
  fi
  echo "Extracting..."
  tar xzf "${ARCHIVE}" -C "${VDIR}" || tar xf "${ARCHIVE}" -C "${VDIR}"
  FOUND="$(find "${VDIR}" -maxdepth 2 -name all_data_with_identities.csv 2>/dev/null | head -1 || true)"
fi

if [ -z "${FOUND}" ]; then
  echo "Could not locate all_data_with_identities.csv after download/extract." >&2
  echo "Manual fallback: download the archive in a browser from" >&2
  echo "  ${URL}" >&2
  echo "extract it, and place all_data_with_identities.csv at ${DEST}" >&2
  echo "(any CSV with columns comment_text, toxicity, split, and the 8 identity" >&2
  echo " columns male/female/LGBTQ/christian/muslim/other_religions/black/white works)." >&2
  exit 1
fi

cp "${FOUND}" "${DEST}"
echo "Placed CivilComments CSV at ${DEST} (from ${FOUND})"

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
