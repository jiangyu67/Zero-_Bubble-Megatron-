#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE_DIR}"

mkdir -p logs
mkdir -p reports

STEPS=100
WARMUP=5
DEVICE="cuda"
BATCH_SIZE=8
SEQ_LENGTH=2048
HIDDEN_SIZE=2048

LOG_A="logs/exp_a_bf16_baseline.log"
LOG_B="logs/exp_b_mxfp8_vanilla.log"
LOG_C="logs/exp_c_mxfp8_zb.log"

printf "Starting Exp A: BF16 Baseline (standard 1F1B)\n"
"${PYTHON:-python}" benchmark_mxfp8_zb.py \
  --mode bf16_baseline \
  --steps "${STEPS}" \
  --warmup "${WARMUP}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-length "${SEQ_LENGTH}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_A}"

printf "\nStarting Exp B: MXFP8-Sim Vanilla (standard 1F1B, no async optimization)\n"
./run_nsys_profile.sh mxfp8_sim report_vanilla \
  --steps "${STEPS}" \
  --warmup "${WARMUP}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-length "${SEQ_LENGTH}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_B}"

printf "\nStarting Exp C: MXFP8-Sim + Zero-Bubble (async comm queue)\n"
./run_nsys_profile.sh mxfp8_zb report_zb \
  --steps "${STEPS}" \
  --warmup "${WARMUP}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-length "${SEQ_LENGTH}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_C}"

printf "\nAll experiments completed.\n"
printf "Logs: %s\n" "${LOG_A}"
printf "Logs: %s\n" "${LOG_B}"
printf "Logs: %s\n" "${LOG_C}"
printf "NSYS reports saved under: reports/\n"
