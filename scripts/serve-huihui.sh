#!/usr/bin/env bash
# Swift1.5-Huihui (halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin), the abliterated Swift 1.5
# checkpoint, on this engine: scripts/serve-v2.5.sh with three settings for this checkpoint: its KV-scale sidecar
# (SCALES), its own served model names (MODEL_NAME), so a client cannot mistake it for the stock or Swift models, and
# `xhigh` as the default reasoning effort. Everything else is serve-v2.5.sh's and passes through unchanged; a SCALES or
# MODEL_NAME you set yourself wins, and a later --default-chat-template-kwargs replaces the xhigh default.
#
#   scripts/serve-huihui.sh /path/to/huihui-checkpoint [extra vllm args]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${1:?usage: scripts/serve-huihui.sh <checkpoint-dir> [extra vllm args]}"; shift
[ -f "$CKPT/EDITED_TENSORS.txt" ] || { echo "$CKPT is not the Swift1.5-Huihui checkpoint (no EDITED_TENSORS.txt); for the stock checkpoint use scripts/serve-v2.5.sh"; exit 1; }
export SCALES="${SCALES:-$HERE/scales/qsa_kv_scales_swift_huihui_dense.json}"
export MODEL_NAME="${MODEL_NAME:-flash-next-swift-huihui flash-next-abliterated}"
echo "serve-huihui: SCALES=$SCALES MODEL_NAME=\"$MODEL_NAME\" default reasoning_effort=xhigh"
exec bash "$HERE/scripts/serve-v2.5.sh" "$CKPT" \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}' "$@"
