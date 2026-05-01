#!/usr/bin/env bash
set -euo pipefail

# Run from repository root; adjust this if you move the script.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESS_DIR="${REPO_ROOT}/wavesfm/preprocessing"

# Input dataset paths (defaults from datasets file). Override via env if needed.
ICARUS_ROOT="${ICARUS_ROOT:-/ssd1/aa99435/data/wireless-tsfm/raw/ICARUS/POWDER_Dataset}"
POWDER_DIR="${POWDER_DIR:-/ssd1/aa99435/data/wireless-tsfm/raw/POWDER/GlobecomPOWDER}"
RADCOM_RAW_H5="${RADCOM_RAW_H5:-/ssd1/aa99435/data/wireless-tsfm/raw/RADCOM/RadComOta2.45GHz.hdf5}"
RML_ROOT="${RML_ROOT:-/ssd1/aa99435/data/wireless-tsfm/raw/RML2022/RML22.01A}"

# Output base directory. Override via env if needed.
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/data/preprocessed}"

mkdir -p "${OUTPUT_BASE}"

echo "Using outputs under: ${OUTPUT_BASE}"

python "${PREPROCESS_DIR}/preprocess_icarus.py" \
  --data-path "${ICARUS_ROOT}" \
  --output "${OUTPUT_BASE}/icarus.h5" \
  --max-len 4096

python "${PREPROCESS_DIR}/preprocess_rfp.py" \
  --data-path "${POWDER_DIR}" \
  --output "${OUTPUT_BASE}/rfp.h5" \
  --chunk-len 512

python "${PREPROCESS_DIR}/preprocess_radcom.py" \
  --input "${RADCOM_RAW_H5}" \
  --output "${OUTPUT_BASE}/radcom.h5"

python "${PREPROCESS_DIR}/preprocess_rml.py" \
  --data-path "${RML_ROOT}" \
  --version 2022 \
  --output "${OUTPUT_BASE}/rml22.h5"

echo "All preprocessing jobs finished."
