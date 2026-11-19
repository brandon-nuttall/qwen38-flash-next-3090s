# Hy4-flash port notes (branch `hy4-flash-port`)

Base: `merlin-v3.0.2.1` (the prod tag). One port commit:
`[Model] Add HY4 Flash and multimodal support — merlin port`
= upstream vllm-project/vllm PR **#68296** (merge commit `b067b6f6`, 20 commits,
merged 2026-11-19) applied as a merge-cherry-pick with the adaptations below.
67 files, +12,096/−161. Authorship upstream (Jee Jee Li); committer + sign-off local.

## What lands

- `vllm/models/hy_v4_flash/` — flash target model: hybrid SWA + compressed
  sparse attention (indexpool cache/ops/prepare), engram table, MoE with the
  routed-hidden-size transform (3072→1536), vision tower + `vl_model`, configs.
- `vllm/models/hy_v4/nvidia/dfly*.py` + `vllm/v1/worker/gpu/spec_decode/dfly/`
  + `speculator_config.py` — the **DFly** drafter (5 draft layers sharing the
  target's sparse KV + indexer selection; zero extra KV). `method: "dfly"`.
- Vision/multimodal: `mm_preprocess.py`, `mrope.py` (`set_token_rope`),
  `vl_cudagraph.py`; Rust chat frontend `multimodal/models/hy_v4_vl*` (+expand).
- Parsers/registries: `renderers/hy_v4.py`, `transformers_utils/processors/hy_v4.py`,
  tool-parser whitespace fix (`_completed_arg_pairs`), tokenizer/renderer rows.
- Config plumbing: `vllm/config/engram.py` (Flash archs + option refusal),
  `vllm/config/speculative.py` (`"dfly"` method, `hy_v4_vl` model_type),
  `vllm.py` (`HYV4DFlyModel` in the breakable-cudagraph set).

## Adaptations vs upstream (why they differ from the PR)

| file | adaptation |
|---|---|
| `speculative.py` | dfly default-model translated to the 0.33-style method gate (`self.model = "dfly"`); upstream's `get_method_config_class` machinery is not vendored. |
| `nvidia/model.py` | iHC sequence parallelism **gated off** (`self.ihc_sequence_parallel = False`): upstream #67659's 3-arg `is_sequence_parallel_active` and `speculative.uses_aux_hidden_states` are absent in this base. `set_token_rope` wiring kept. |
| `nvidia/attention.py` | merlin's `requested_sparse` expression kept (upstream `sparse_layer_ids` helper absent); `external_topk`/`topk_snapshot` adopted (the merged indexer path references them). |
| `config/engram.py` | HYV4Flash archs merged with merlin's `flashnext_family.target_archs()` refactor; Flash option refusal covers `dp_shared_memory`/`embedding_across_dp` only (no `disk_offload`/`peer_group_size` fields in this base). |
| `config/vllm.py` | ROCm breakable-cudagraph arch extension skipped (merlin prunes the AMD hy_v4 tree). |
| `nvidia/moe.py` | upstream `expert_map_manager` EP attrs adopted verbatim — this tree has `fused_moe/routed_experts.py` with the full chain. |
| tests / docs / .buildkite | upstream-only paths from the PR not vendored (merlin prunes them). Rust `models.rs` keeps `hy_v4_vl` only (no `inkling2` module here). |

## Boot-phase follow-ups (first rebuild + first boot will surface these)

1. **Launcher wiring**: `merlin serve` `SPEC_CHOICES = ("mtp", "dflash2")` needs a
   `dfly` choice (MTP is *not* supported for flash per the upstream recipe); new
   profile row(s) in `tools/merlin/profiles/hy4-flash.json` — **TP8 only** (TP/TEP;
   upstream supports no PP/DP/CP/EPLB/LoRA), so no octo-3090 tp4pp2-style shape.
2. **DFly dispatch on the merlin spec flow**: verify `speculator_config.py` +
   `v1/worker/gpu/spec_decode/dfly/config.py` dispatch `method=="dfly"` end-to-end
   against 0.33's scheduler/spec-decode glue (the PR was written against 0.34).
3. **Runtime API drift**: only syntax was checked locally (py_compile). First boot
   with the rebuilt wheel is the real import/attr test. Watch: FusedMoEFactory
   kwargs (`routed_input_transform`), `prefill/decode_context_parallel_size` refs
   (removed by the SP gate — re-grep), engram loader hooks vs the PLE pull transport.
4. **Engram host memory**: 64B-param n-gram table, FP8 in-checkpoint — expect
   ~60+ GiB host RAM like Flash-Next's PLE; wire into `--ple`-style docs once booted.
5. **sm_86 qualification**: upstream tuned for SM90/SM100 (DeepGEMM MegaMoE,
   FlashTopK, small-head FlashMLA). Expect Triton/FlashInfer fallbacks; benchmark
   prefill/decode before trusting the profile (see the 3090 GPU-unlock + P2P notes).
6. **Dependency floor**: `transformers>=5.23` (config class + `HYV4FlashProcessor`),
   check against the release lock (`release/requirements-0.33.lock`) at rebuild.

## Upstream patches tracked but NOT picked

- #68462 (open) HYV4/hunyuan tool-parser multi-tool streaming fix — revisit after boot.
- #68365 (open) fp8_ds_mla KV on Flash SWA — later KV-dtype tuning candidate.
- #68389/#68387/#68364/#68321 (open) SM90/Blackwell perf + NVFP4 — irrelevant on sm_86.
- #68609/#68347 (open) P/D disaggregation — not our topology.
- #50246 (open) generic DFly + D-Cut — superseded by #68296's in-tree DFly.
- #67659/#67490 (merged upstream) — their context arrived inside #68296; the SP
  feature itself stays gated until a vLLM 0.34 sync.

## Reference commands

Upstream run shape (for parity checks once the profile exists):
`vllm serve tencent/Hy4-flash --dtype bfloat16 --max-model-len 8192 --max-num-seqs 4
--max-num-batched-tokens 8192 --limit-mm-per-prompt '{"image":1,"video":0}' -tp 8
--enable-auto-tool-choice --reasoning-parser hy_v4 --tool-call-parser hy_v4
--speculative-config '{"num_speculative_tokens":5,"method":"dfly"}'`
(recipe: [vllm-project/vllm recipes/models/tencent/Hy4-flash.yaml](https://recipes.vllm.ai/tencent/Hy4-flash),
model: [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash)).
