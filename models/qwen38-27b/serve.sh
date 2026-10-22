#!/usr/bin/env bash
# Merlin 3.0.0: shim for the one launcher, tools/merlin/launch.py --profile 27b-tp2-3090.
# TP=2 on two cards. Knobs: MODEL PORT HOST SERVED VENV GPUS KV_BYTES LPT SPEC_K MAX_PIXELS CHAT_TEMPLATE THINKING P2P RECOVERSSM CHUNK BLOCK_SIZE CUDA_HOME (README.md).
#   MODEL=/path/to/checkpoint PORT=8100 bash models/qwen38-27b/serve.sh
set -euo pipefail
REPO="${MERLIN_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec "${MERLIN_PYTHON:-python3}" "$REPO/tools/merlin/launch.py" --profile 27b-tp2-3090 -- "$@"
