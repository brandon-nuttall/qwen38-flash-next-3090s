#!/usr/bin/env bash
# Merlin 3.0.0: shim for the one launcher, tools/merlin/launch.py --profile fn-quad-3090.
# Knobs: PORT HOST MODEL_NAME PLE_HOME SCALES GUARD CACHE_ROOT TREE VENV WARM COUNTERS IMG_CAP_PX DRAFT_HEAD DRAFT_VOCAB REDUCED_DRAFT_VOCAB KV_BYTES CHUNK BLOCK_SIZE LPT AUTO_TOPO CHAT_TEMPLATE CUDA_HOME (docs/reference.md).
#   bash models/flash-next/serve.sh /path/to/Qwen3.8-Flash-Next-W4A16-Merlin [extra vllm args]
set -euo pipefail
REPO="${MERLIN_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec "${MERLIN_PYTHON:-python3}" "$REPO/tools/merlin/launch.py" --profile fn-quad-3090 -- "$@"
