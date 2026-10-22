# Qwen3.8-27B v2.0.0: 0.32 tile reclaim and 8 MP stress results

Measured on 4× RTX 3090 (sm_86), 2026-10-15. The v2.0.0 recipe targets Merlin 3.0.0,
based on **vLLM 0.32.0**. These are Qwen3.8-27B base-checkpoint measurements; they are not
ThinkingCap performance or quality measurements.

## Tile reclaim

FlashInfer **0.7.0.post1**, with the ported sm_86 FP8 small-query verify tile and
`FLASHINFER_SM8X_FP8_SMALLQ=cta32`. ITL below is the harness's decode-step interval, in ms,
median over three repetitions at each prompt target.

| Profile | 4K | 32K | 131K | 131K on 0.32 without tile | 131K on 0.30 |
|---|---:|---:|---:|---:|---:|
| TP2 | **19.99** | **21.41** | **25.27** | 27.27 | 25.3 |
| TP4 | **15.83** | **16.62** | **18.37** | 19.45 | 18.4 |

The tile reclaims the 6–8 % depth regression to 0.30 parity on TP2 and TP4.
These step intervals are not single-stream output tok/s; speculative acceptance also affects output rate.
The tile run used a vLLM 0.32 candidate build in an isolated tile environment, rather than the final release assembly.

**GSM8K-200, as served, TP2, tile on:** 192 / 193 / 191 across three runs, mean **192.0/200**,
against v1.3.0's **196/200**: −2.0 percentage points, 95 % Newcombe CI **[−5.89, +1.61] pp**.
The difference is within the confidence interval (not significantly worse); it is not a claim of identical accuracy.
A same-day 0.32 tile-off GSM8K-200 comparison remains pending.

**Numerics and provenance.** Kernel validation passed; the port was bitwise equal to the old shipped tile on
44/44 cells. Tile OFF vs pristine was identical on 28/28 outputs. Tile ON vs OFF differed (25 logprob
differences and 3 token divergences), as expected from a different reduction order.
The validation harness reported a cache-provenance check failure because its selection-line allow-list did not admit warm torch AOT cache-hit lines;
the result was adjudicated as reclaimed because the OFF outputs were identical. Updating that allow-list remains a follow-up.

These tile measurements used 4× RTX 3090, vLLM 0.32 candidate code and FlashInfer 0.7.0.post1,
2026-10-15; they do not qualify the final release assembly.
The apply patch is `0001-sm8x-fp8-smallq-verify-tile-070.patch` (sha256 prefix `8fd81372`),
with JIT suffix `_sm8x_smallq_070_v1`.
The release installer must package this 0.7.0.post1 patch; setting the knob on unpatched FlashInfer does not activate it.
See the [release tooling status](../../../../release/README.md).

## Image cap and pools

The v2.0.0 per-image cap is **8,388,608 pixels (8 MP)**. Larger images are downscaled before encoding.
It is a per-image cap, not a qualification of 64 simultaneous full-size images.

27B TP2 serves images up to 8 MP. With the vision-peak option on (`VLLM_MERLIN_VISION_PEAK=1`, the TP2 launcher default), the 50-item TP2 stress sweep's worst minimum free was 806 MiB per card (2026-10-15), above the 524 MiB floor. A separate knob OFF/ON identity check matched tokens and logprobs on 6/6 prompts. 27B TP4 (608-690 MiB free across runs: 690 MiB on 2026-10-15, 608 MiB in the 2026-10-16 mini gate) and Flash-Next (902 MiB) meet the target.

| Profile | KV pool, tokens |
|---|---:|
| TP2 | **826,689** |
| TP4 | **1,974,072** |

FGPC is **ON** for TP2 and TP4 via `--enable-mamba-shared-prefix-checkpoint`, including the ThinkingCap wrappers.
Chunked vision encoding is inert at ≤8 MP.
The integrated launchers default to `MAX_PIXELS=8388608` (8 MP); examples select that value explicitly.

The 8 MP stress result was measured on RTX 3090 TP2 with a pre-merge vLLM 0.32 build on 2026-10-15:
98 images, no OOM, and 76/76 correct responses. The headroom target was still missed.
The pool totals are separate from image headroom; the integrated launchers still need assembled-tree GPU qualification.
The older 0.30 TP2 8 MP OOM is historical.

## Other 0.32 measurements

Concurrent decode measured TP4 with 14×127K resident: **439.5 tok/s aggregate**, **31.79 ms/tok per agent**,
**0 preemption**. This is a separate concurrency workload, not a row of the tile ladder.
With MTP off, the TP2 prefix-cache hit ratio was **0.9996**, with TTFT **18.25→0.52 s** at 32K.
These concurrency and cache measurements used RTX 3090 TP4/TP2 on a pre-release vLLM 0.32 build, 2026-10-15.
The release-candidate engine and then the merged release-candidate engine matched that reference's T=0 tokens and logprobs
on four configurations, with 28/28 documents plus four controls per configuration. These identity checks do not
rerun the concurrency, cache, image or tile workloads on the final release assembly.

## Evidence limits

These are recorded hardware results, transcribed for the release; this documentation change did not run a GPU.
No final assembled release installer, container, or ThinkingCap 0.32 performance/quality ladder was tested here.
Same-day tile-off GSM8K-200 remains pending. The
[2026-10-12 base card](../2026-10-12/BENCH-CARD.md) and
[ThinkingCap card](../../../thinkingcap/benchmarks/2026-10-12/ThinkingCap-BENCH-CARD.md)
are historical vLLM 0.30 records.
