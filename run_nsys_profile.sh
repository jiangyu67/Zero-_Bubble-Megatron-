#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <benchmark-mode> [output-base] [benchmark args]"
  echo "Example: $0 mxfp8_zb report_zb --steps 10 --warmup 2"
  echo "Note: this wrapper is benchmark-focused. Use run_distributed_nsys_profile.sh for torchrun."
  exit 1
fi

MODE="$1"
shift
OUTPUT_BASE="report_${MODE}"
if [ "$#" -ge 1 ] && [[ "$1" != --* ]]; then
  OUTPUT_BASE="$1"
  shift
fi
mkdir -p reports
OUTPUT="reports/${OUTPUT_BASE}_$(date +%Y%m%d_%H%M%S)"

# Prefer a CUDA+NVTX trace with cudaProfilerApi capture for reliability.
# benchmark_mxfp8_zb.py calls torch.cuda.profiler.start/stop on step_index==9.
# Users can override via env vars if their CUDA runtime/driver combo is problematic.
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx}"
NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-cudaProfilerApi}"
NSYS_CAPTURE_RANGE_END="${NSYS_CAPTURE_RANGE_END:-stop}"

# Optional NVTX capture support if explicitly requested.
if [[ "${NSYS_CAPTURE_RANGE}" == "nvtx" ]]; then
  # Range name used by benchmark_mxfp8_zb.py when step_index==9.
  NSYS_NVTX_CAPTURE="${NSYS_NVTX_CAPTURE:-PROFILE_STEP_10}"
fi

NSYS_ARGS=(
  --trace="${NSYS_TRACE}"
  --capture-range="${NSYS_CAPTURE_RANGE}"
  --capture-range-end="${NSYS_CAPTURE_RANGE_END}"
  --force-overwrite=true
  --stats=false
  --export=none
  --output "${OUTPUT}"
)

if [[ "${NSYS_CAPTURE_RANGE}" == "nvtx" ]]; then
  NSYS_ARGS+=( --nvtx-capture "${NSYS_NVTX_CAPTURE}" )
fi

set +e
nsys profile \
  "${NSYS_ARGS[@]}" \
  "${PYTHON:-python}" benchmark_mxfp8_zb.py --mode "${MODE}" "$@"
COMMAND_STATUS=$?
set -e

if compgen -G "${OUTPUT}.nsys-rep" > /dev/null; then
  echo "NSYS report saved to ${OUTPUT}.nsys-rep"
elif compgen -G "${OUTPUT}.qdstrm" > /dev/null; then
  echo "NSYS trace saved to ${OUTPUT}.qdstrm (report conversion may have failed)"
else
  echo "Expected NSYS output was not generated for ${OUTPUT}" >&2
  echo "Tip: if you are using nvtx capture, ensure torch.cuda.nvtx is available and PROFILE_STEP_10 is emitted." >&2
  exit 1
fi

if [ "${COMMAND_STATUS}" -ne 0 ]; then
  echo "Benchmark command exited with status ${COMMAND_STATUS}" >&2
  exit "${COMMAND_STATUS}"
fi
