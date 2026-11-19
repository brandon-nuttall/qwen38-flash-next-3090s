#!/usr/bin/env bash
# First-boot serving wrapper for Hy4-flash W4A16 on the merlin port
# ("neuron", 8 x RTX 3090). Direct vllm serve, conservative first-boot
# defaults; the launcher registry gets its profile row after this boot
# measures real memory numbers (port-log follow-up #1).
#
# Env knobs (house conventions): GPUS HOST PORT MAXLEN SEQS MNBT GMU SPEC KV
#   SPEC=on  -> DFly2 drafter (5 speculative tokens, method dfly)
#   SPEC=off -> target only (the port-log's reference setting)
#   KV=auto (default: engine default) | bfloat16 | fp8_e4m3
set -euo pipefail
REPO=/home/brandon/merlin-engine
CHECKPOINT="${1:-/home/brandon/models/hy4-flash-W4A16-Merlin}"
GPUS="${GPUS:-0,1,3,4,5,6,7,8}"          # card 2 is the GTX 1650 display
PORT="${PORT:-8010}"
HOST="${HOST:-0.0.0.0}"
MAXLEN="${MAXLEN:-32768}"                 # first-boot conservative; native 1M
SEQS="${SEQS:-4}"                        # upstream recipe starts at 4
MNBT="${MNBT:-4096}"
GMU="${GMU:-0.90}"                       # GNOME keeps GL context on card 0
SPEC="${SPEC:-on}"
KV="${KV:-auto}"

[ -s "$CHECKPOINT/config.json" ] || { echo "no checkpoint at $CHECKPOINT (quant done?)"; exit 1; }
[ -s "$CHECKPOINT/model.safetensors.index.json" ] || { echo "no index in $CHECKPOINT"; exit 1; }

kv_args=()
case "$KV" in
  auto) ;;
  bfloat16|fp8_e4m3) kv_args=(--kv-cache-dtype "$KV") ;;
  *) echo "KV must be auto|bfloat16|fp8_e4m3" >&2; exit 1 ;;
esac

spec_args=()
case "$SPEC" in
  on)  spec_args=(--speculative-config '{"num_speculative_tokens":5,"method":"dfly"}') ;;
  off) ;;
  *) echo "SPEC must be on|off" >&2; exit 1 ;;
esac

cd "$REPO"
exec env CUDA_VISIBLE_DEVICES="$GPUS" VLLM_USE_RUST_FRONTEND=0 \
  ./venv/bin/vllm serve "$CHECKPOINT" \
  --served-model-name hy4-flash \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$SEQS" \
  --max-num-batched-tokens "$MNBT" \
  --gpu-memory-utilization "$GMU" \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --attention-backend FLASHMLA_SPARSE \
  --tool-call-parser hy_v4 \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  "${kv_args[@]}" "${spec_args[@]}" \
  --host "$HOST" --port "$PORT" "$@"
