#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec bash scripts/hpc/build_adult_7p5m_v3_large_outputs.sh "$@"
