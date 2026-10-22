# Ornith 1.5 122B-A10B on 4× RTX 3090

Ornith 1.5 is ornith-ai's fine-tune of Qwen3.5-122B-A10B. Merlin serves its W4A16 build at 262K context on four
RTX 3090s, behind one OpenAI-compatible endpoint:
- TP2×PP2 with expert parallelism;
- Gated DeltaNet RecoverSSM;
- MTP speculative decoding with the model's own draft head;
- prefix caching.

The model's own chat template and sampling are served as shipped: T 0.6, top_p 0.95, top_k 20. The reasoning parser
is `qwen3` and the tool-call parser `qwen3_coder`.

> **Status: tested on the reference host** (registry profile `ornith-quad-3090`, 4 × RTX 3090 with P2P), and on a
> 4 × RTX 3090 host without P2P (`ornith-quad-3090-nop2p`). The full bench card ran on the reference host on 2026-10-21.

```bash
MODEL=/path/to/Ornith-1.5-122B-A10B-W4A16-Merlin PORT=8000 bash models/ornith1.5/serve.sh
# or, from the checkout: merlin serve ornith1.5 /path/to/Ornith-1.5-122B-A10B-W4A16-Merlin
```

Extra arguments go to `vllm serve` after the launcher's own flags; later flags win. Environment knobs:

| Knob | Default |
|---|---|
| `MODEL` | required (the checkpoint directory) |
| `GPUS` | `0,1,2,3` |
| `PORT` | `8000` |
| `HOST` | `127.0.0.1` |
| `SERVED` | `ornith` |
| `VENV` | `<repo>/venv` |
| `TREE` | unset (optional engine tree, put first on `PYTHONPATH`) |
| `CACHE_ROOT` | `<repo>/.vllm-cache` |
| `P2P` | the row's: `on` for `ornith-quad-3090`, `off` for `ornith-quad-3090-nop2p` (a mismatch is refused) |
| `KV_BYTES` | `3556769792` (P2P row) / `3892314112` (no-P2P row) |
| `CHUNK` | `1024` |
| `SEQS` | `8` |
| `MAXLEN` | `262144` |
| `BLOCK_SIZE` | `4608` (at least 4608) |
| `LPT` | unset (no long-prefill threshold) |
| `PP_PARTITION` | `25,23` (two counts summing to 48; empty = the engine's even split) |
| `MAX_PIXELS` | `8388608` (at least 8,388,608) |
| `UV` | `1` (uniform verify on; `0` = off) |

## Merlin 3.0.0 bench card (2026-10-21)

Maintainer measurements on 4× RTX 3090 (220 W, PCIe Gen4 with P2P) with the Merlin 3.0.0 release install plus the Ornith engine changes (uniform verify on) and a prefill experiment that did not ship, one
fresh boot per run (the full card, then a second boot for the identity check and GSM8K).
Decode is one stream with thinking on, in tok/s after the first streamed token, median of 3. Prefill is one cold
request.

| Measurement | Ornith 1.5 |
|---|---|
| Decode, 4K / 131K / 262K prompt (262K = exactly 261,888 prompt tokens + 128 out) | 144 / 106 / 100 tok/s |
| Cold prefill, 10K / 100K prompt | 5,148 / 3,049 tok/s |
| Prompts sent at once (not a residency test) | 3 × ~262K needle prompts (259,261-261,565 tokens) sent at once, every needle found |
| Needles | 10/10 at 131K and 23/23 at 262K, exact |
| Agent-trace replay (recorded tool-use sessions) | 52/52 tool calls valid, decode p50 149 tok/s |
| GSM8K-200 (5-shot, thinking off; second boot) | 190/200 |
| KV pool | 894,618 tokens |
| MTP acceptance (tokens per step, K=3, served sampling, thinking on) | prose 2.39-2.48, code 2.56, GSM8K 2.58 |

## The recipe

| Setting | Value | Why |
|---|---|---|
| Layout | TP2 × PP2 + expert parallel | four 24 GB cards, 122B weights at W4A16 |
| Layer partition | `VLLM_PP_LAYER_PARTITION=25,23` (`PP_PARTITION`) | the MTP layer rides the 23-layer stage |
| State space | RecoverSSM: `--use-replayssm --mamba-backend triton`, Model Runner V2; SSM state fp32 | |
| Speculation | MTP, 3 draft tokens, probabilistic drafting | the model's own MTP head; its routed experts are INT4 (acceptance equal to BF16: 2.943 vs 2.947) |
| Uniform verify | `MERLIN_MOE_UNIFORM_VERIFY=1` (`UV`) | runs the verify batch through one MoE graph; bitwise identical to UV off at 1 session on one compile cache (same answers at 6 sessions and 258K) |
| Prefix cache | `--enable-prefix-caching --mamba-cache-mode align`, `--block-size 4608 --prefix-match-unit 64` | the engine needs ≥ 4,512 at fp32 SSM state |
| Chunk | `--max-num-batched-tokens 1024` | 2048 fits but measured no gain |
| KV | FP8 with the checkpoint's calibrated scales. `--kv-cache-memory-bytes 3556769792` with P2P, `3892314112` without | byte-pinned, never a utilisation fraction; 3 × 262K by engine fit on both rows. Each pin was verified at 8 MP + 262K with uniform verify off; the UV full card then held 262K needles and answered 3 × ~262K needle prompts sent at once |
| Interconnect | with P2P: `NCCL_P2P_LEVEL=SYS NCCL_PROTO=Simple VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`; without: `NCCL_P2P_DISABLE=1`; custom all-reduce off on both | |
| Images | `--mm-processor-kwargs` longest edge 8,388,608 px, 8 images per prompt, `--mm-processor-cache-gb 0` | the 8 MP release floor |
| Tools | `--enable-auto-tool-choice --tool-call-parser qwen3_coder`, reasoning parser `qwen3` | |

## Limits of the evidence

- Each KV pin's 8 MP + 262K headroom was measured with uniform verify off. The UV full card then held 262K needles and
  answered 3 × ~262K needle prompts sent at once, but UV's own image headroom was not measured separately.
- Uniform verify is bitwise identical to UV off at 1 session (0 of 1,456 positions differ), with the same answers at 6 sessions
  and on a 258,000-token row, when both boots share one compile cache. A fresh compile cache drifts at temperature 0, with or without UV.
- The no-P2P host's 262K identity requests failed (prompt plus output over 262,144 tokens) and give no identity evidence. The
  RecoverSSM check on the no-P2P host matched at 6 sessions; at 1 session the outputs differed on 9 of 12 prompts (temperature-0
  drift).

## Launcher environment

Besides the reference argv, the launcher sets `VLLM_CACHE_ROOT` / `TRITON_CACHE_DIR` under `CACHE_ROOT`, per-port
`MASTER_PORT` / `VLLM_PORT`, `VLLM_NO_USAGE_STATS=1` / `DO_NOT_TRACK=1`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` and the
venv-toolkit `CUDA_HOME` default, and it puts `TREE` first on `PYTHONPATH` while keeping an inherited `PYTHONPATH` behind it.

## Checkpoint

`checkpoint.sha256` lists the 130 files of the Hugging Face repository (the KV-calibrated pack plus `NOTICE`, `LICENSE-MIT`, `LICENSE-QWEN` and ornith-ai's card with its two images); the model card `README.md` and its two images are not listed. Run
`sha256sum -c checkpoint.sha256` inside the checkpoint directory.

Build:
- source `ornith-ai/Ornith-1.5-122B` @ `dfa07bab90e90746ffbfc0b0cba3126eb9265de2`;
- routed experts INT4 g128;
- MTP routed experts INT4 g128 RTN; MTP attention, router, norms and shared expert stay BF16;
- GDN projections INT8;
- FP8 KV scales calibrated over the 12 attention layers and the MTP layer.

The model's own chat template rejects a system message that is not the first message. A client that sends per-turn
system messages gets an error, and with in-place system prompts it loses prefix-cache reuse.

## License

Code in this directory (the launcher and checksum list): Apache-2.0 (the repository's [LICENSE](../../LICENSE)).
Model weights are not part of this repository. The checkpoint `halt95/Ornith-1.5-122B-A10B-W4A16-Merlin` is a quantised
derivative of [`ornith-ai/Ornith-1.5-122B`](https://huggingface.co/ornith-ai/Ornith-1.5-122B), whose model card declares the
MIT License ([`licenses/LICENSE-MIT`](licenses/LICENSE-MIT); the source repository ships no licence file). Ornith 1.5 is a
fine-tune of [`Qwen/Qwen3.5-122B-A10B`](https://huggingface.co/Qwen/Qwen3.5-122B-A10B); the Qwen materials within it retain
their Apache-2.0 terms ([`licenses/LICENSE-Apache-2.0-Qwen.txt`](licenses/LICENSE-Apache-2.0-Qwen.txt)). Provenance and the
changes made to the checkpoint: [`licenses/Ornith-NOTICE`](licenses/Ornith-NOTICE). This is a summary, not legal advice; the
licence texts govern.
