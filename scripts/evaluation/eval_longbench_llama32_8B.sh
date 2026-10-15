#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
# Legacy wrapper: keep old entry name alive, forward to the corrected Llama-3.1-8B script.
set -euo pipefail

echo "[warn] eval_longbench_llama32_8B.sh is deprecated; forwarding to eval_longbench_llama31_8B.sh" >&2
exec "$(dirname "$0")/eval_longbench_llama31_8B.sh" "$@"
