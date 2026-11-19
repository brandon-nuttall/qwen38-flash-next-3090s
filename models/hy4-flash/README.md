# Hy4-flash — Merlin recipe (WIP, first-boot stage)

Serving `tencent/Hy4-flash` (220B multimodal MoE, MLA + DSA + engram, DFly2
drafter) as a W4A16-Merlin checkpoint on this host's 8 x RTX 3090 sm_86.

## Status

- Engine port: branch `hy4-flash-port` — upstream vLLM #68296 cherry-pick +
  0.34 helpers vendored; port log: `docs/hy4-flash-port.md`.
  flash-next regression battery on this port: PASS (2026-11-19, see
  `~/hy4-quant/README.md` for the recorded gates).
- Quant: `~/hy4-quant/` (llm-compressor AWQ, experts-only INT4 g128, BF16 source).
- First boot: `serve-hy4.sh` — direct `vllm serve` wrapper while the launcher
  registry (`tools/merlin/profiles/hy4-flash.json`) waits for measured memory
  numbers (port-log follow-up #1; upstream supports **TP/TEP only** — no PP).

## First boot (planned invocation)

```bash
# inside a GPU container mirroring the regression instance (see ~/.harbor
# conventions): mounts ~/merlin-engine, the checkpoint dir and /cache
SPEC=on KV=auto MAXLEN=32768 ./models/hy4-flash/serve-hy4.sh <checkpoint-dir>
```

Upstream reference shape (recipes.vllm.ai/tencent/Hy4-flash): TP8, BF16,
`--attention-backend FLASHMLA_SPARSE`, `--tool-call-parser hy_v4
--reasoning-parser hy_v4 --enable-auto-tool-choice`, DFly2 speculative
`{"num_speculative_tokens":5,"method":"dfly"}`, engram tables host-resident
(per-family default landed in the port: `vllm/config/engram.py`).

Sources: [model card](https://huggingface.co/tencent/Hy4-flash),
[vLLM recipe](https://recipes.vllm.ai/tencent/Hy4-flash),
[PR #68296](https://github.com/vllm-project/vllm/pull/68296).
License note: Tencent Hy License 1.0 — §10 MaaS carve-outs; private use fine.
