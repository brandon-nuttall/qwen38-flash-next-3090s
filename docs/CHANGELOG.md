# Changelog

Every Merlin release in one place, newest first. Until 3.0.0 each model shipped from its own repository with its own
version numbers; here each sits under the engine release it ran on, with its original tag named. Figures under the 2.x
releases were measured on vLLM 0.28-0.30 when they shipped and are kept as published; they are not re-measured on 3.0.0.

| Release | Date | Engine | Models |
|---|---|---|---|
| [3.0.0](#merlin-300) | 2026-10-22 | vLLM 0.32.0 | Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron 3.5 Super VL, Ornith 1.5 |
| [2.5.3](#v253) | 2026-10-12 | Merlin engine v2.5.2 (vLLM 0.30.0) | Flash-Next, Swift1.5-Huihui (new); Qwen3.8-27B v1.3.0 / v1.3.1 and ThinkingCap (new) |
| [2.5.2](#v252) | 2026-10-12 | Merlin engine v2.5.2 (vLLM 0.30.0) | Flash-Next |
| [2.5.1](#v251) | 2026-10-06 | vLLM 0.30.0 + Merlin commits | Flash-Next, Swift1.5 |
| [2.2.0](#v220) | 2026-09-25 | vLLM 0.30.0 + 58 commits | Flash-Next, Swift1.5 (new); Qwen3.8-27B v1.0.0 (new) |
| [2.0.1](#v201) | 2026-09-18 | vLLM pre-0.30 | Flash-Next |
| [2](#v2) | 2026-09-17 | vLLM pre-0.30 | Flash-Next |
| [1](#v1) | 2026-09 | vLLM 0.28.0 + patches | Flash-Next |

## Merlin 3.0.0

2026-10-22. Built on vLLM 0.32.0. One engine, one install and one command for all seven checkpoints, in one repository.

### In brief

- **Seven checkpoints, one engine.** Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap and two new ones,
  Nemotron 3.5 Super VL and Ornith 1.5 (ornith-ai's Qwen3.5-122B-A10B fine-tune), each with a recipe under `models/`.
  Every one runs at its native context (262,144 tokens; Nemotron 3.5 Super VL 1,048,576) with an FP8 KV cache, MTP
  speculative decoding and vision.
- **One command.** `./setup.sh` checks the host, recommends a checkpoint and layout for your GPUs, shows every download
  with its size, installs the prebuilt engine (about 1.5 minutes, no compiling, nothing system-wide) and starts the
  server. Answers can be changed before anything downloads. Walkthrough: [How to serve](how-to-serve.md).
- **One launcher.** `merlin serve` identifies the checkpoint and picks the largest tested layout for the cards it finds
  (`--tp`, `--profile`, `--instances N` for several servers on one host); `merlin doctor` checks the host first.
- **Prebuilt engine for Ampere and Ada.** One wheel for sm_80, sm_86 and sm_89, checked against its published sha256.
- **More hardware support.** Profiles for RTX 3090 without P2P, eight RTX 3090s, RTX 4090 (24 and 48 GB), RTX A6000 / A40,
  A100, A30 and CMP 170HX; see [Which checkpoint runs where](../README.md#which-checkpoint-runs-where).
- **Faster prefill.** Flash-Next cold prefill 5,020 → 5,660 tok/s at 10K and 5,417 → 6,051 tok/s at 100K against
  v2.5.3 (+12–13 %); Qwen3.8-27B on four cards 2,483 → 3,009 tok/s at 10K (+21 %) against v1.3.0.
- **RecoverSSM on every checkpoint**, including 27B on four cards (pool 1,974,072 → 2,012,105 tokens) and both new
  ones, and **27B on one RTX 3090**: a 271,506-token pool, one 262K or two 131K sessions, images up to 4 MP.

### Measured on Merlin 3.0.0

4× RTX 3090 without NVLink, P2P on, 220 W, 2026-10-20 bench cards (27B one- and two-card agents/GSM8K, ThinkingCap two-card and Nemotron agents/GSM8K are from a
2026-10-17 pre-release build). Decode: one stream, thinking on, median of 3, at a
4K / 131K prompt and the full 262,144-token context (261,888 prompt + 128 out).

| Checkpoint | GPUs | KV pool, tokens | Decode, tok/s, 4K / 131K / 262K | 262K prefill | 8 agents | GSM8K-200 |
|---|---|---|---|---|---|---|
| Flash-Next | 4× | 924,993 | 168 / 170 / 184 | 5,838 tok/s | 638 tok/s | 199 |
| Swift1.5 | 4× | 924,993 | 177 / 180 / 177 | 5,834 tok/s | 643 tok/s | 197 |
| Swift1.5-Huihui | 4× | 924,993 | 187 / 188 / 189 | 5,880 tok/s | 660 | 198 |
| Qwen3.8-27B | 1× | 271,506 | 124 / 95 / 73 | 456 tok/s | n/a (two at most) | 195 |
| Qwen3.8-27B | 2× | 826,689 | 207 / 173 / 142 | 858 tok/s | 517 tok/s | 194 |
| Qwen3.8-27B | 4× | 2,012,105 | 250 / 235 / 215 | 1,596 tok/s | 752 tok/s | 197 |
| ThinkingCap | 2× | 826,689 | 198 / 182 / 165 | 860 tok/s | 502 tok/s | 193 |
| ThinkingCap | 4× | 2,012,105 | 253 / 255 / 248 | 1,594 tok/s | 733 tok/s | 195 |
| Nemotron 3.5 Super VL | 4× | 1,392,275 at 262K (1,856,367 at the shipped 1M context) | 97 / 101 / 91 | 2,951 tok/s | 429 tok/s | 195 |
| Ornith 1.5 | 4× | 894,618 | 144 / 106 / 100 | 1,704 tok/s | 468 | 190 |

Ornith 1.5 was measured on 2026-10-21 with its engine changes (uniform MoE verify on) before they merged into 3.0.0;
its bench card is [models/ornith1.5/benchmarks/2026-10-21](../models/ornith1.5/benchmarks/2026-10-21/BENCH-CARD.md).

### New checkpoints

- **Nemotron 3.5 Super VL** (NVIDIA, 120B-A12B MoE): W4A16 routed experts with an INT4 MTP head, tensor parallel 2 ×
  pipeline parallel 2 with expert parallel on four cards, RecoverSSM on its Mamba-2 path. It serves its native
  1,048,576-token context by default (pool 1,856,367 tokens): a 1,048,052-token needle came back exact and, re-sent, was
  served from the resident cache (1,047,936 tokens cached, 3 s to first token). At a 262K context the pool held 5 × 262K
  and 9 × 131K sessions at once with 0 preemptions; needles 23/23 at 262K and 10/10 at 131K (2026-10-17 pre-release build). One image per prompt,
  downscaled to the model's native ~3.4 MP.
- **Ornith 1.5** (ornith-ai's Qwen3.5-122B-A10B fine-tune, 122B MoE): INT4 routed experts, INT4 MTP experts, INT8
  GDN projections and calibrated FP8 KV scales; tensor parallel 2 × pipeline parallel 2 with expert parallel,
  RecoverSSM, MTP K=3 with uniform MoE verify (the verify batch through one MoE graph). The INT4 MTP experts accept as the BF16 ones do (mean acceptance length 2.943 against 2.947). Pool
  894,618 tokens (3 × 262K); three ~262K prompts sent at once all came back with their needles; a recorded tool-use
  session replayed with 52/52 tool calls valid. Tested with P2P on the reference host and without P2P on a second
  4× RTX 3090 host (pool 977,838 tokens, GSM8K-200 193/200). Checkpoint about 66.4 GiB.
- **Qwen3.8-27B on one card.** The single-card layout returns, tested on an RTX 3090 with Qwen3.8-27B: pool 271,506
  tokens, decode 124 / 95 / 73 tok/s at 4K / 131K / 262K, 262K prefill 456 tok/s and, on the 2026-10-17 pre-release build, GSM8K-200 195/200 and needles 5/5 at
  131K and 12/12 at 262K. ThinkingCap serves on the same profile but was not run on one card.

### One install, one command

- **`./setup.sh`** fetches uv and a Python 3.13 into the checkout when the host has none, then checks the host and
  names the exact fix for each failure: driver R580 or later, GPUs and P2P, GCC/G++ 13 or newer, disk against the
  plan, host RAM for the model. It asks one question at a time, with Enter taking the answer it recommends for the
  hardware it found: model, layout, models folder, local or network access (with a generated API key) and port.
- **Every download shown before it starts:** the engine wheel, the 222 locked packages and the checkpoint, each with
  its size, then one confirmation. The checkpoint download resumes, is pinned to one published revision and is checked
  against the release checksums; a marker skips the full re-hash while the files are unchanged (`--rehash` forces it).
  Setup ends with `merlin doctor`, the endpoint, a curl test and the server in the foreground.
- **Re-runs and updates.** `./setup.sh` again starts with the saved answers and asks nothing; `git pull && ./setup.sh`
  reinstalls only when the wheel, lock or commit moved and never downloads the model again; a failed reinstall
  restores the previous environment. `--reconfigure`, `--yes --model M` and `--dry-run` (fetches and writes nothing)
  are there for scripted installs, and `merlin pull <model>` downloads and verifies a checkpoint without a GPU.
- **Careful with your host.** Answers are kept per model, and a folder that holds another checkpoint is refused. The
  API key is stored owner-only, shown once and never printed in commands. uv's archive is checked against its
  published sha256. Disk is budgeted per filesystem.
- **`merlin serve <checkpoint>`** identifies the checkpoint, picks the largest tested layout that fits the idle cards
  and starts the server through one launcher. Every layout is a profile; every knob (`PORT`, `HOST`, `KV_BYTES`,
  `MAX_PIXELS`, `CHUNK`, `BLOCK_SIZE`, `LPT`, ...) is an environment variable. `--dry-run` prints the resolved command.
- **Card selection follows the board.** Cards are ordered by tested model, NUMA node, host bridge and PCIe switch, read
  from `nvidia-smi topo -m`. `--instances N` cuts separate card sets over that order, each with its own ports and
  compile cache, and fills mixed hosts class by class. A layout qualified with P2P is never picked without it; the
  one-card layout and `--tp 1 --instances N` are named instead. On 4× CMP 170HX, eight agents at 16K measured 873 tok/s
  on `--tp 1 --instances 4`, 523 on two TP=2 servers and 371 on one TP=4 server.
- **`merlin doctor`** checks GPUs, driver, host RAM and shared memory, honours `GPUS` / `CUDA_VISIBLE_DEVICES`, and
  reads available memory within a container's limit. `merlin list` shows every profile with its hardware.
- **Hardware profiles** beyond the reference host: RTX 3090 without P2P, eight RTX 3090s, RTX 4090 (24 and 48 GB),
  RTX A6000 / A40, A100, A30 and CMP 170HX. 4× CMP 170HX (64 GB) is tested; the others ship untested and expected to
  work. A card below its VRAM class gets a smaller pool instead of a refusal.
- **Prebuilt wheel.** One wheel for sm_80, sm_86 and sm_89 installs without compiling. The installer accepts only the
  wheel named in the published sha256, built from the checkout's native sources; a source install builds the same three
  architectures (about 1.5 hours on 8 cores). Requirements are locked to 222 exact pins with SHA-256 hashes, with the
  CUDA 13.0 toolkit wheels. A fresh install (2026-10-21) and a from-source build (2026-10-22) passed on the reference
  host.

### Engine

- **Rebased onto vLLM 0.32.0** (an upstream release candidate, the five upstream commits of the next release candidate, then the final release's one commit,
  [a macOS-only socket-family fix](https://github.com/vllm-project/vllm/pull/62177)). Eleven earlier backports now come
  from upstream, including prefix-cache and MTP fixes, GDN metadata reuse and the 1× M-RoPE cache. The rebased engine
  was temperature-0 token- and logprob-identical to the reference build on four configurations.
- **FP8 KV cache on Ampere.** Merlin's sm_86 FP8 attention reader for Flash-Next is kept, and 27B gets the FlashInfer
  0.7.0.post1 sm_86 FP8 small-query verify tile, which restores four-card step-time parity with vLLM 0.30: inter-token
  latency at 4K / 32K / 131K 19.99 / 21.41 / 25.27 ms on two cards and 15.83 / 16.62 / 18.37 ms on four.
- **Tables in host RAM.** Flash-Next's n-gram embedding table (about 47.7 GiB) lives in system memory, one copy shared
  by every GPU. The token embedding tables of the main model and its MTP drafter are read from pinned host memory:
  frees VRAM, no speed loss (about 1.2 GiB per card on 27B at TP=2). It is now on by default for Qwen3.5 MoE too, the
  Ornith 1.5 support (`VLLM_HOST_EMBED_TABLE=0` keeps the table on the GPU).
- **RecoverSSM on every checkpoint.** [RecoverSSM](https://github.com/vllm-project/vllm/pull/58863) recovers the GDN and
  Mamba-2 state for MTP speculative decoding, on by default for all seven checkpoints.
- **Cache retention and thrash protection.** Follow-ups resume from cache on a 64-token grid. Cache-thrash fixes are on
  by default for hybrid models: cost-aware admission and session-coherent yields. Under 8 × 131K and 12 × 80K
  oversubscription, follow-up turns reused 74.7 % and 82.1 % of their prompt from cache. A cold request waits behind
  cached follow-ups only when admitting it would evict blocks they still need.
- **PCIe all-reduce.** Flash-Next on P2P hosts uses FlashInfer's PCIe all-reduce for decode and NCCL `Simple` for
  prefill, in place of vLLM's custom all-reduce with `NCCL_PROTO=LL`.
- **Marlin W8A16** replaces AllSpark for the Flash-Next INT8 DeltaNet projections.
- **Bounds guards with idle P8.** The runtime bounds guards stay on, and idle cards drop to P8: 110 W idle on four RTX
  3090s, against 446 W when the guard reader kept them awake.
- **Leaner memory use.** The prompt-logprob GPU workspace is bounded by a 256-row logits chunk (was 1,024), with the
  Flash-Next pool unchanged at 924,993 tokens. 27B shared-prefix checkpoints are evict-first, one per prefix, with no
  extra blocks held at rest. Nemotron's image preprocessor no longer opens a CUDA context on GPU 0, returning about
  260 MiB to the first card.
- **New fine-tunes run as is.** A Flash-Next-family checkpoint that ships its own FP8 KV scales is served with them
  (`SCALES=` still overrides), with no repository update.
- **Upstream protocol change.** The upstream [/v1/messages protocol change](https://github.com/vllm-project/vllm/pull/61427)
  is merged with Merlin's own edits to that endpoint.
- `use_local_argmax_reduction` works with the Flash-Next MTP drafter, contributed by
  [brandon-nuttall](https://github.com/brandon-nuttall). It is off in every shipped profile, since it needs greedy
  drafting: it gave +10–20 % eight-way aggregate decode on the contributor's eight-card rig and no gain on the
  reference layout (459 tok/s against 483 for the shipped profile).

### Per-model recipes

- **Flash-Next, Swift1.5 and Swift1.5-Huihui.** Every MTP drafter scores the full vocabulary with a BF16 draft head;
  `DRAFT_HEAD=int8` is available (6.5–7.2 % lower ms per output token at the same MTP acceptance) and planned as the
  default in the next release, and no reduced draft-vocabulary table ships (`REDUCED_DRAFT_VOCAB=1` is unavailable). The Swift1.5 and Huihui launchers are
  generated from the checkpoint profile with their own KV scales, both tested on the reference host. The pool holds
  3 × 262K and 6 × 131K sessions at once, 0 preemptions, and an 8 MP image fits at the full pin. Flash-Next decode
  with thinking on is 1.4–5.8 % faster than v2.5.2 at every depth (187.4 against 177.2 tok/s at 259,587 tokens).
- **Qwen3.8-27B and ThinkingCap.** RecoverSSM on every layout, MTP K=4 with an INT4 MTP head, shared-prefix checkpoints
  on, and the 0.7.0.post1 verify tile. Four cards: pool 2,012,105 tokens (7.68 × 262K), cold prefill +15 % at 100K and
  time to first token at 255K 179.3 → 161.6 s against v1.3.0, step time unchanged. Two cards take 8 MP images (806 MiB
  worst-case free per card over a 50-image sweep, above the 524 MiB floor).
- **27B on one card at 4 MP.** The single-card layout caps images at 4 MP (4,194,304 pixels) and keeps the 262K KV
  pin, with three output-identical memory settings on by default for it alone: a 96 MiB FlashInfer workspace, the MTP
  drafter sharing the target's workspace, and the vision rotary applied in place. On an RTX 3090, 4 MP passed at one and
  two sequences and after a 262K needle with 558–563 MiB minimum free (bar 524 MiB); without the settings the card ran
  out of memory at 3 MP.
- **Nemotron 3.5 Super VL at its native 1M.** The 1,048,576-token context is the default, with RecoverSSM and MTP K=3
  at the same KV pin as the 262K pool; one full 1M session fits on four RTX 3090s, or five 262K sessions.
- **Ornith 1.5.** Two profiles, with and without P2P, both tested: `merlin serve ornith1.5 <dir>`. Uniform MoE verify is
  on (`UV=0` turns it off) and was bitwise identical to verify off at one session.

### Defaults

- **Reasoning effort `medium` on every Qwen3.8 checkpoint** (Flash-Next was `low`; Swift1.5, Swift1.5-Huihui and
  ThinkingCap were `xhigh`). A request still picks its own effort through `chat_template_kwargs`. At `medium` the MTP
  drafter lands more tokens per step than at `xhigh` and GSM8K-200 scored the same or better: Flash-Next prose 2.66
  against 2.29 and code 3.06 against 2.35; ThinkingCap prose 3.27 against 2.78 and code 3.55 against 2.65. Nemotron's
  template has no `reasoning_effort` (its own switch is `medium_effort`); Ornith 1.5's has only `enable_thinking`.
- **One chat template.** Every Qwen recipe serves froggeric's fixed Qwen3.8 template (v22.5), which accepts system
  messages after the first turn, so coding agents keep their prefix cache across turns. `CHAT_TEMPLATE=<path>` serves
  another, `CHAT_TEMPLATE=` the checkpoint's own.
- **8 MP image cap** on every checkpoint (was 4 MP), except Nemotron (its native ~3.4 MP) and one-card 27B rows (4 MP).
- **Full-vocabulary MTP drafter** with a BF16 draft head on the Flash-Next family (`DRAFT_HEAD=bf16|int8|int4`).
  Nemotron and the 27B family use INT4 MTP heads, Ornith 1.5 INT4 MTP experts.

### Docs and repository

- **One repository.** The Merlin vLLM fork sits at the root and each checkpoint has a recipe under `models/`, every one
  at version 3.0.0; checkpoints stay on Hugging Face. One installer replaces the separate engine bundles, per-package
  patch series and compiled-ops tarball of earlier releases.
- **[How to serve](how-to-serve.md)** walks through setup end to end: the questions, talking to the server, serving on
  your network, choosing GPUs, switching models and fixing problems.
- **Bench cards** for every checkpoint, dated 2026-10-20 (Ornith 1.5: 2026-10-21): decode ladder with thinking on and
  off, exact 262K, cold prefill, sessions held at once, images, agents and GSM8K-200, with the previous release beside
  it where one exists. Earlier cards keep their original dates and toolchains.
- The README links the upstream vLLM README and documentation, with a NOTICE and a security policy.

### Fixes

- **RecoverSSM in align-mode prefix caching.** With prefix caching, RecoverSSM and MTP, a window whose accepted tokens
  crossed a block boundary could store a wrong block-boundary state (the final state was always correct), and a later
  cache hit on that block could diverge from a cold run. The boundary and final states are now stored in one pass. It
  affects earlier 27B two-card and Flash-Next releases served with prefix caching: at block size 4,096 and K=3,
  roughly 1 window in 1,365 could write such a state.
- **Engine fixes carried into 3.0.0:** the padded guard sentinel, prefix-cache keys for DCP and EAGLE, the RecoverSSM
  pipeline broadcast, the host n-gram table's boot, format validation and event lifetime, prefill logits lifetime and a
  FlashInfer staging fence.
- A prefix-cache reset that cannot complete keeps the retained follow-up prefixes.
- Sequence parallelism warns instead of refusing to start; it stays off by default.

### Upgrading

Make a fresh checkout and run `./setup.sh`: a models folder that already holds a checkpoint is checked against the
release checksums and reused rather than downloaded again. The old serve scripts (`scripts/serve-v2.5.sh` and
others) forward to the new launcher with a notice and are removed in the next release. Merlin 3.0.0 does not carry the
reduced-vocabulary INT4 draft head of the Flash-Next v3.0.0 build (2026-10-10, never published).
Upgrading is recommended for anyone serving with prefix caching, for the RecoverSSM fix above. A clone of
qwen38-flash-next-3090s can point at the new name with
`git remote set-url origin https://github.com/halt95/merlin-engine.git`.

## v2.5.3

2026-10-12. Engine unchanged: Merlin engine v2.5.2.

- **New: Swift1.5-Huihui** (Flash-Next tag `v2.5.3`), Swift1.5 with huihui-ai's abliteration on its 96 dense output
  projections. It does not refuse: 29 of 278 StrongREJECT prompts refused (10.4 %). GSM8K 96.29 % against Swift1.5's
  95.98 % in the same run.
- **New: Qwen3.8-27B v1.3.0** on Merlin engine v2.5.2 plus 27B patches, two and four cards. Single-stream thinking
  decode at 4K / 32K / 128K / 255K: 206 / 198 / 172 / 144 tok/s on two cards, 258 / 249 / 238 / 226 on four.
  RecoverSSM on two cards (pool 826,689, three 262K sessions resident), pool 1,974,072 on four (seven 262K sessions).
  FlashInfer 32-row verify tile, MTP K=4, cached-token counts in every response, a per-image pixel cap.
- **New: ThinkingCap**, BottleCap AI's reasoning fine-tune of Qwen3.8-27B, served as a profile of 27B v1.3.0.
- **Qwen3.8-27B v1.3.1** (2026-10-13), documentation only: the glibc floor is 2.34 (Ubuntu 22.04, Debian 12, RHEL 9).

## v2.5.2

2026-10-12. Merlin engine v2.5.2 = v2.5.1 plus one fix.

- **Idle cards drop to P8 with the bounds guards on.** The guard reader polled every 50 ms forever, holding idle RTX
  3090s in P2: 446 W against 110 W in P8 on four cards. It now parks once the engine drains, after a final read of every
  counter, so no bounds check is lost. `VLLM_E2_GUARD_IDLE=0` restores the old behaviour. The same fix ships in
  Qwen3.8-27B v1.3.0. Thanks to the community member who reported it and proposed the first fix.
- No kernel, scheduler or numerics change; decode and prefill read as v2.5.1's within run-to-run noise.

## v2.5.1

2026-10-06. Flash-Next and Swift1.5 (tag `v2.5.1-swift1.5`). v2.5.0 was not published.

- **Agent follow-up turns resume from cache** on a 64-token grid instead of the 4,096-token block: median time to
  first token on follow-ups 0.96 s (v2.2.0: 1.56 s). With eight deep conversations at once, 92.5 % of follow-up prompt
  tokens came from cache.
- **KV pool 806,792 → 924,993 tokens (+15 %)**: six 131K or three 262K sessions resident with 0 preemptions.
- Cold prefill 5,219-5,533 tok/s from 10K to 261K, 1.0-2.8 % above v2.2.0; single-stream decode about 3 % below it.
- Transformers 5.18.0 (combining marks tokenize as upstream), up to 42 images per prompt at 4 MP each.
- GSM8K-200 198/200, structured output 80/80, the 262K needle found exactly.

## v2.2.0

2026-09-25.

- **Flash-Next** moves onto the public vLLM 0.30.0 tag plus 58 commits: a reliability release at parity with v2
  (greedy repeatability, any-rank fault detection, faster shutdown).
- **New: Swift1.5** (tag `v2.2.0-swift1.5`, 2026-09-28), the Merlin checkpoint with the 300 dense tensors Swift 1.5
  changed, its own FP8 KV scales and `xhigh` effort by default.
- **New: Qwen3.8-27B v1.0.0** (2026-09-27), the first 27B release: stock vLLM 0.30.0 plus five patches, two cards (pool
  794,351) or four (1,949,844), FP8 KV, MTP K=3, froggeric's chat template.

## v2.0.1

2026-09-18. The v2 shape plus one engine fix: structured output under concurrency.

## v2

2026-09-17. Flash-Next moves from four-way tensor parallel to **tensor parallel 2 × pipeline parallel 2** with expert
parallel, the FP8 n-gram table served over a host-mapped pull transport and the token embeddings moved into pinned
host memory. The KV pool grows from 342,912 to **806,792 tokens**: three full 262K sessions resident at once.
Single-stream decode 162 / 166 / 169 / 176 tok/s at 4K / 32K / 131K / 261K (parity with v1); prefill 10-26 % faster.

## v1

2026-09. The first Flash-Next release: vLLM 0.28.0 plus community commits and six patches, four-way tensor parallel with
expert parallel, MTP K=3, KV pool 342,912 tokens.
