#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_distributed_nsys_profile.sh [options] <output-base> -- <distributed launch command ...>

Example:
  ./run_distributed_nsys_profile.sh qwen2p5_3b_profile \
    --profile-step-start 10 \
    --profile-step-end 12 \
    --profile-ranks 0 \
    -- \
    torchrun --nproc-per-node=8 train_rl.py --mock-data --train-iters 20

Options:
  --profile-step-start N   Inject Megatron profiling start step (default: 10)
  --profile-step-end N     Inject Megatron profiling end step (default: 12)
  --profile-ranks CSV      Inject Megatron profile ranks, e.g. 0 or 0,1 (default: 0)
  --no-inject-profile-args Do not append --profile / step / rank arguments
  --trace VALUE            Override NSYS trace domains (default: cuda,nvtx)
  --capture-range VALUE    Override NSYS capture range (default: cudaProfilerApi)
  --help                   Show this help

Environment overrides:
  NSYS_TRACE, NSYS_CAPTURE_RANGE, NSYS_CAPTURE_RANGE_END, NSYS_STATS, NSYS_EXPORT
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

PROFILE_STEP_START=10
PROFILE_STEP_END=12
PROFILE_RANKS="0"
INJECT_PROFILE_ARGS=true
TRACE_OVERRIDE=""
CAPTURE_RANGE_OVERRIDE=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile-step-start)
      PROFILE_STEP_START="$2"
      shift 2
      ;;
    --profile-step-end)
      PROFILE_STEP_END="$2"
      shift 2
      ;;
    --profile-ranks)
      PROFILE_RANKS="$2"
      shift 2
      ;;
    --no-inject-profile-args)
      INJECT_PROFILE_ARGS=false
      shift
      ;;
    --trace)
      TRACE_OVERRIDE="$2"
      shift 2
      ;;
    --capture-range)
      CAPTURE_RANGE_OVERRIDE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

OUTPUT_BASE="$1"
shift

if [ "$1" != "--" ]; then
  echo "Missing '--' before distributed launch command." >&2
  usage
  exit 1
fi
shift

if [ "$#" -eq 0 ]; then
  echo "Missing distributed launch command." >&2
  usage
  exit 1
fi

if [ "${PROFILE_STEP_END}" -le "${PROFILE_STEP_START}" ]; then
  echo "--profile-step-end must be greater than --profile-step-start." >&2
  exit 1
fi

mkdir -p reports
OUTPUT="reports/${OUTPUT_BASE}_$(date +%Y%m%d_%H%M%S)"

NSYS_TRACE="${TRACE_OVERRIDE:-${NSYS_TRACE:-cuda,nvtx}}"
NSYS_CAPTURE_RANGE="${CAPTURE_RANGE_OVERRIDE:-${NSYS_CAPTURE_RANGE:-cudaProfilerApi}}"
NSYS_CAPTURE_RANGE_END="${NSYS_CAPTURE_RANGE_END:-stop}"
NSYS_STATS="${NSYS_STATS:-false}"
NSYS_EXPORT="${NSYS_EXPORT:-none}"

NSYS_ARGS=(
  --trace="${NSYS_TRACE}"
  --capture-range="${NSYS_CAPTURE_RANGE}"
  --force-overwrite=true
  --stats="${NSYS_STATS}"
  --export="${NSYS_EXPORT}"
  --output "${OUTPUT}"
)

if [ "${NSYS_CAPTURE_RANGE}" != "none" ]; then
  NSYS_ARGS+=( --capture-range-end="${NSYS_CAPTURE_RANGE_END}" )
fi

COMMAND=( "$@" )
if [ "${INJECT_PROFILE_ARGS}" = true ]; then
  COMMAND+=(
    --profile
    --profile-step-start "${PROFILE_STEP_START}"
    --profile-step-end "${PROFILE_STEP_END}"
  )

  IFS=',' read -r -a PROFILE_RANK_ARRAY <<< "${PROFILE_RANKS}"
  for rank in "${PROFILE_RANK_ARRAY[@]}"; do
    if [ -n "${rank}" ]; then
      COMMAND+=( --profile-ranks "${rank}" )
    fi
  done
fi

echo "NSYS output: ${OUTPUT}.nsys-rep"
echo "Launch command: ${COMMAND[*]}"

set +e
nsys profile "${NSYS_ARGS[@]}" "${COMMAND[@]}"
COMMAND_STATUS=$?
set -e

if compgen -G "${OUTPUT}.nsys-rep" > /dev/null; then
  echo "NSYS report saved to ${OUTPUT}.nsys-rep"
else
  echo "Expected NSYS report was not generated for ${OUTPUT}" >&2
  exit 1
fi

if [ "${COMMAND_STATUS}" -ne 0 ]; then
  echo "Wrapped command exited with status ${COMMAND_STATUS}" >&2
  exit "${COMMAND_STATUS}"
fi
