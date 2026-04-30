#!/usr/bin/env bash
# ============================================================================
# run_finetuning_wavesfm.sh — WavesFM downstream fine-tuning, IQ tasks only.
#
# Tasks: radcom  rml  rfp  interf
#
# Modes
# -----
#   lp     — linear probe: freeze full encoder, train tokenizer + head
#   ft2    — partial fine-tune: freeze first 6 transformer blocks
#   lora   — LoRA adapters (rank 32, alpha 64) on Q/V projections
#   strict — strict probe: only classification head + CLS token trainable
#   sl     — supervised baseline: train full model end-to-end
#
# Override via env, e.g.:
#   DATA_ROOT=/mnt/data/preprocessed CKPT_PATH=/runs/wavesfm.pth \
#   MODES="lp lora" SEEDS="0 1 2" bash run_finetuning_wavesfm.sh
# ============================================================================
set -euo pipefail

# ── configure ────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-$(dirname "$0")/data/preprocessed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-wavesfm_runs}"
CKPT_PATH="${CKPT_PATH:-}"          # pretrained WavesFM checkpoint (required for lp/ft2/lora/strict)
CKPT_NAME="${CKPT_NAME:-wavesfm}"

SEEDS="${SEEDS:-0 1 2}"
MODES="${MODES:-lp ft2 lora}"       # space-separated: lp ft2 lora strict sl
TASKS="${TASKS:-radcom rml rfp interf}"

DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-2}"

WAVESFM_DIR="$(cd "$(dirname "$0")/wavesfm" && pwd)"
SCRIPT="${WAVESFM_DIR}/main_finetune.py"

# ── task configuration (from wavesfm/run_finetune_all.py) ───────────────────

declare -A TASK_H5=(
    [radcom]="${DATA_ROOT}/radcom.h5"
    [rml]="${DATA_ROOT}/rml22.h5"
    [rfp]="${DATA_ROOT}/rfp.h5"
    [interf]="${DATA_ROOT}/icarus.h5"
)

declare -A TASK_EPOCHS=(
    [rfp]=10
    [interf]=35
    [rml]=50
    [radcom]=50
)

declare -A TASK_BATCH=(
    [rml]=2048
    [radcom]=2048
)
DEFAULT_BATCH=256

declare -A TASK_SMOOTH=(
    [rfp]=0.1
    [interf]=0.02
)

# interf uses stratified split + class weights
declare -a STRATIFIED_TASKS=(interf)
_is_stratified() {
    local t="$1"
    for s in "${STRATIFIED_TASKS[@]}"; do [[ "$s" == "$t" ]] && return 0; done
    return 1
}

[[ -z "$CKPT_PATH" ]] && echo "[warn] CKPT_PATH not set — non-sl modes train from scratch"

# ── sweep ────────────────────────────────────────────────────────────────────
TOTAL=0; DONE=0; SKIPPED=0

for SEED in $SEEDS; do
    for MODE in $MODES; do
        for TASK in $TASKS; do
            TOTAL=$((TOTAL + 1))

            H5="${TASK_H5[$TASK]}"
            if [[ ! -f "$H5" ]]; then
                echo "[skip] task=$TASK — h5 not found: $H5"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            EPOCHS="${TASK_EPOCHS[$TASK]}"
            BS="${TASK_BATCH[$TASK]:-$DEFAULT_BATCH}"
            OUT_DIR="${OUTPUT_ROOT}/${CKPT_NAME}_${MODE}/${TASK}/s${SEED}"

            BEST_CKPT="${OUT_DIR}/best.pth"
            LAST_CKPT="${OUT_DIR}/$(printf 'checkpoint_%03d.pth' $((EPOCHS - 1)))"
            if [[ -f "$BEST_CKPT" || -f "$LAST_CKPT" ]]; then
                echo "[skip] task=$TASK mode=$MODE seed=$SEED (checkpoint exists)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            mkdir -p "$OUT_DIR"

            CMD=(
                python "$SCRIPT"
                --task        "$TASK"
                --train-data  "$H5"
                --output-dir  "$OUT_DIR"
                --model       vit_multi_small
                --batch-size  "$BS"
                --num-workers "$NUM_WORKERS"
                --epochs      "$EPOCHS"
                --seed        "$SEED"
                --device      "$DEVICE"
                --warmup-epochs 5
            )

            # pretrained checkpoint (all modes except sl)
            if [[ "$MODE" != "sl" && -n "$CKPT_PATH" ]]; then
                CMD+=(--finetune "$CKPT_PATH")
            fi

            # mode flags
            case "$MODE" in
                lp)     ;; # freeze_encoder is True by default (not sl_baseline)
                ft2)    CMD+=(--frozen-blocks 6) ;;
                lora)   CMD+=(--lora --lora-rank 32 --lora-alpha 64) ;;
                strict) CMD+=(--strict-probe) ;;
                sl)     CMD+=(--sl-baseline) ;;
                *)      echo "[error] unknown mode: $MODE"; exit 1 ;;
            esac

            # task-specific flags
            if _is_stratified "$TASK"; then
                CMD+=(--stratified-split --class-weights)
            fi
            if [[ -v TASK_SMOOTH[$TASK] ]]; then
                CMD+=(--smoothing "${TASK_SMOOTH[$TASK]}")
            fi
            if [[ "$TASK" == "interf" ]]; then
                CMD+=(--accum-steps 2)
            fi

            echo "──────────────────────────────────────────────────────────"
            echo "[run] mode=$MODE  task=$TASK  seed=$SEED  epochs=$EPOCHS"
            echo "  ${CMD[*]}"
            "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/train.log"
            DONE=$((DONE + 1))
            echo "[done] mode=$MODE  task=$TASK  seed=$SEED"
        done
    done
done

echo "════════════════════════════════════════════════════════════"
echo "[done] total=$TOTAL  ran=$DONE  skipped=$SKIPPED"
echo "Results in: $OUTPUT_ROOT"
