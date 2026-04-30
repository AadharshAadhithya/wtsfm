#!/usr/bin/env bash
# ============================================================================
# run_tsfm_benchmark.sh — TSFM downstream sweep over IQ wireless tasks.
#
# Tasks: radcom  rml  rfp  interf
# Modes: lp (linear probe)  ft2 (partial fine-tune)  lora (LoRA R=32 α=64)
#
# Task-specific epochs, batch sizes, label smoothing, gradient accumulation,
# and stratified splits are handled automatically by run_tsfm_probe.py using
# the same defaults as wavesfm/run_finetune_all.py.
#
# Override via env, e.g.:
#   DATA_ROOT=/mnt/data/preprocessed MODELS="chronos2 timesfm" \
#   MODES="lp lora" SEEDS="0 1 2" bash run_tsfm_benchmark.sh
# ============================================================================
set -euo pipefail

# ── configure ────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-$(dirname "$0")/data/preprocessed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-tsfm_runs}"

MODELS="${MODELS:-chronos2 timesfm patchtst sundial timer_s1 moirai2 tirex toto}"
SEEDS="${SEEDS:-0 1 2}"
MODES="${MODES:-lp ft2 lora}"
TASKS="${TASKS:-radcom rml rfp interf}"

DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

SCRIPT="$(cd "$(dirname "$0")" && pwd)/run_tsfm_probe.py"

# ── per-task h5 paths (matches wavesfm/run_finetune_all.py naming) ────────────
declare -A TASK_H5=(
    [radcom]="${DATA_ROOT}/radcom.h5"
    [rml]="${DATA_ROOT}/rml22.h5"
    [rfp]="${DATA_ROOT}/rfp.h5"
    [interf]="${DATA_ROOT}/icarus.h5"
)

# ── sweep ────────────────────────────────────────────────────────────────────
TOTAL=0; DONE=0; SKIPPED=0

for MODEL in $MODELS; do
    for MODE in $MODES; do
        for SEED in $SEEDS; do
            for TASK in $TASKS; do
                TOTAL=$((TOTAL + 1))

                H5="${TASK_H5[$TASK]}"
                if [[ ! -f "$H5" ]]; then
                    echo "[skip] task=$TASK — h5 not found: $H5"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi

                # run_tsfm_probe.py writes to: OUTPUT_ROOT/{task}_{model}_{mode}_s{seed}/
                SLUG="${TASK}_${MODEL}_${MODE}_s${SEED}"
                OUT_DIR="${OUTPUT_ROOT}/${SLUG}"
                RESULT="${OUT_DIR}/result.json"

                if [[ -f "$RESULT" ]]; then
                    echo "[skip] already done: $RESULT"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi

                CMD=(
                    python "$SCRIPT"
                    --task        "$TASK"
                    --train-data  "$H5"
                    --model       "$MODEL"
                    --mode        "$MODE"
                    --seed        "$SEED"
                    --num-workers "$NUM_WORKERS"
                    --device      "$DEVICE"
                    --output-dir  "$OUTPUT_ROOT"
                    --warmup-epochs 5
                )

                if [[ "$MODE" == "lora" ]]; then
                    CMD+=(--lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA")
                fi

                echo "──────────────────────────────────────────────────────────"
                echo "[run] model=$MODEL  mode=$MODE  task=$TASK  seed=$SEED"
                echo "  ${CMD[*]}"
                mkdir -p "$OUT_DIR"
                "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/train.log"
                DONE=$((DONE + 1))
                echo "[done] model=$MODEL  mode=$MODE  task=$TASK  seed=$SEED"
            done
        done
    done
done

echo "════════════════════════════════════════════════════════════"
echo "[done] total=$TOTAL  ran=$DONE  skipped=$SKIPPED"
echo "Results in: $OUTPUT_ROOT"
