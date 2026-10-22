# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Merlin (fork) environment registry, moved out of vllm/envs.py.

Declarations (TYPE_CHECKING), the fork parsers, the fork getters
(MERLIN_ENV), the fork's runtime-only names (MERLIN_IGNORED) and the
draft-vocab compile factor. vllm/envs.py merges them with three hook lines:
``environment_variables.update(MERLIN_ENV)``, ``ignored_factors |=
MERLIN_IGNORED`` and ``add_merlin_compile_factors(factors)``. The getters
began as the bodies that stood in vllm/envs.py; the Python 0/1 switches now
share one parser (``_e2_strict_switch``) and the determinism toggles default
to ``SHIPPED_KERNEL_TOGGLE_DEFAULTS``. Imports nothing but the standard
library at module level, so vllm.envs can import it first.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Lower-peak eager Qwen3 vision tower (exactly "1"; default off).
    # AOT: no -- eager tower dispatch; compiled vision blocks keep the plain path.
    # CAPTURE: no -- metadata-only encoder graphs keep the plain path.
    VLLM_MERLIN_VISION_PEAK: bool = False
    # Prompt scoring runs eagerly after forward, outside AOT and CUDA graphs.
    VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE: int = 256
    VLLM_MERLIN_PROMPT_LOGPROBS_TENSOR_BUDGET_MIB: int = 384
    # AOT: no -- eager tower dispatch, outside compiled decoder regions.
    # CAPTURE: no -- metadata-only encoder graphs keep the plain path.
    # Both controls are runtime-only (MERLIN_IGNORED); A/B arms reuse AOT caches.
    VLLM_MERLIN_VISION_CHUNK_ROWS: int = 8192
    VLLM_MERLIN_VISION_CHUNK_MIN_PIXELS: int = 8_388_608
    VLLM_E2_QSA_LOGITS_GRANULE_MB: int = 0
    # Prefix-cache thrash fixes for hybrid (Mamba/GDN) models, default ON; see
    # vllm/v1/core/cache_thrash.py. Cost-aware admission: seconds a young cold
    # miss may be parked behind waiting resident hits (0 = strict arrival
    # order). Coherent yields: need-sized, session-coherent yields (strict
    # "0"/"1"; "0" restores the old one-pin-at-a-time yields). AOT: no.
    # CAPTURE: no -- scheduler-side admission order and yield policy only.
    VLLM_E2_SCHED_MISS_DEFER_S: int = 120
    VLLM_E2_SCHED_COHERENT_YIELD: bool = True
    # Drafter MoE profile forwards at startup (exactly "0" skips them; any
    # other value runs them). Host-side dummy forwards in the memory-profiling
    # window only: AOT: no. CAPTURE: no. Runtime-only (MERLIN_IGNORED).
    VLLM_SPEC_PROFILE_SHAPES: bool = True
    # Profiling-teardown KV retention check (vllm/v1/worker/gpu/cudagraph_utils.py):
    # "1" turns the boot abort into a warning; default "0" aborts. AOT: no.
    # CAPTURE: no. Runtime-only (MERLIN_IGNORED).
    MERLIN_ALLOW_PROFILING_KV_RETENTION: bool = False
    # --- 27B series knobs -------------------
    # Moved verbatim from the 27B series' vllm/envs.py registration (same
    # defaults). Hashed unless listed in MERLIN_IGNORED below.
    VLLM_FLASHINFER_PLAN_CACHE: bool = True
    VLLM_GDN_BA_SPLITK: bool = True
    VLLM_DRAFT_SEQ_LENS_NOSYNC: bool = True
    VLLM_DRAFT_SEQ_LENS_VERIFY: bool | None = None
    VLLM_DRAFT_BOUNDED_PLAN: bool = False
    VLLM_DRAFT_BOUNDED_PLAN_VERIFY: bool | None = None
    VLLM_TARGET_SEQ_LENS_ASYNC: bool = False
    VLLM_TARGET_SEQ_LENS_ASYNC_VERIFY: bool | None = None
    # GDN (RecoverSSM) host metadata trims, decode steps only. Both are
    # host-side only: the staged buffers keep their addresses and receive the
    # same values, so neither AOT nor capture identity depends on them (ignored
    # factors). Default off.
    VLLM_GDN_SHARED_ALIGN_INDEX: bool = False
    VLLM_GDN_DECODE_FASTPATH: bool = False
    # RecoverSSM: enqueue the post-sampling GDN commit (and the mamba align
    # postprocess that consumes it) after the MTP proposal instead of before it,
    # on the same stream. AOT: no -- Python call order after sampling, outside
    # the compiled forward. CAPTURE: no -- not inside any captured graph.
    # Ignored factor, so an A/B flip reuses one AOT cache.
    VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE: bool = False
    # RecoverSSM post-sampling commit execution mode
    # (vllm/v1/worker/gpu/model_states/recoverssm_commit_graph.py): "eager"/"0"/unset
    # = the unchanged launch chain, "staged" = staged inputs + eager chain (control
    # arm), "graph"/"1" = staged inputs + a replayed standalone CUDA graph of the same
    # chain. Default only: the worker method set_recoverssm_commit_mode flips it in a
    # running process; the boot value also decides whether the CUDA-graph capture
    # phase captures the commit graphs (not for eager). AOT: no -- read by Python
    # after sampling, outside the compiled forward. CAPTURE: no for the forward
    # graphs; the commit graphs are separate in-process graphs, keyed on the step's
    # tensors, never cached across processes, and bit-identical to the eager chain.
    # Ignored factor, so an A/B flip reuses one AOT cache. Unknown values raise at
    # boot.
    VLLM_RECOVERSSM_COMMIT_GRAPH: str = ""
    # --- Opt-in Qwen3.5 MoE / single-card memory knobs (off by default) ----
    # FULL-graph MTP verify on Qwen3.5 MoE / sm_86
    # (v1/qwen35_series.uniform_verify_active). Runtime-only (MERLIN_IGNORED).
    MERLIN_MOE_UNIFORM_VERIFY: bool = False
    # MTP drafter reuses the target's FlashInfer float workspace
    # (merlin/draft_workspace.py). Runtime-only (MERLIN_IGNORED).
    MERLIN_SHARE_DRAFT_WORKSPACE: bool = False
    # VISION_PEAK path: rotate Q/K inside the QKV buffer (no QK copy). Runtime-only.
    # Inert unless VLLM_MERLIN_VISION_PEAK=1 runs the eager tower on the flash-attn
    # ViT backend.
    VLLM_MERLIN_VISION_ROTARY_INPLACE: bool = False
    # -----------------------------------------------------------------------
    # None selects the family default; explicit values retain exact "1" parsing.
    # None and False must be distinct compile factors.
    VLLM_HOST_EMBED_TABLE: bool | None = None
    VLLM_PLE_OFFLOAD_READY_TIMEOUT: float = 600.0
    # "1" samples the MTP draft token from the draft's own
    # softmax truncated to its top-k/top-p set under the request's sampling
    # params, and caches that truncated distribution for the verifier. Exactly
    # "0"/"1"; default OFF. Read once at speculator init (static).
    # AOT: no -- sample_draft is outside the compiled model regions.
    # CAPTURE: yes -- draft sampling is CAPTURED under FULL graphs (prefill,
    # decode and fused decode), so the truncation kernel is baked into them;
    # the draft top-k/top-p buffers it reads are persistent and refreshed by
    # the runner before every propose. Graphs are per-boot (never persisted),
    # so hashing would only cost an AOT miss on every configuration and could not
    # protect in-process replay; the flag is frozen at init instead.
    VLLM_SPEC_DRAFT_UNDER_TARGET_TOPK: bool = False
    # "1" keys the block-verification acceptance draws and
    # the draft Gumbel noise by round (anchor position + in-round offset)
    # instead of absolute position, which makes rejection_sample_method="block"
    # distribution-exact. Effective only with block verification (a logged
    # no-op otherwise); standard/synthetic keying is never changed. Exactly
    # "0"/"1"; default OFF (byte-identical draws). Read once at rejection
    # sampler and speculator init (static).
    # AOT: no -- the rejection kernels and sample_draft run outside the
    # compiled model regions; Triton keys its own cache on the new constexprs.
    # CAPTURE: yes -- the draft sampler is CAPTURED under FULL draft graphs
    # with the STEP_KEYED constexpr; the keys it reads (positions, the
    # persistent current_draft_step, seeds) are device buffers, so replay needs
    # no host RNG. Same shape as the draft top-k/top-p switch: frozen at init, graphs never persisted.
    VLLM_SPEC_STEP_KEYED_RNG: bool = False
    # --- Flash-Next knobs ---------------------------------------------
    # Registered so compile_factors() hashes them; the read sites still parse
    # and fail closed themselves. Each entry states two identities: AOT (does it
    # change the compiled artifact; the only one compile_factors() keys) and
    # CAPTURE (does it change what a cudagraph bakes in; protected only by not
    # changing it within one process). Where either is not established from the
    # code, the knob is hashed anyway: a spurious miss costs a recompile.
    #
    # COMPILE/CAPTURE FACTORS (hashed):
    # CAPTURE: yes, established -- clip counting allocates persistent per-layer
    # int64 counters and its increment ops are CAPTURED into the cudagraph (see
    # the graph-safety note in models/qwen4_exp/nvidia/qsa.py); unset, those ops
    # are absent. AOT: not established (the QSA forward is a custom op); hashed
    # conservatively.
    VLLM_QSA_KV_CLIP_COUNT: str = ""
    # CAPTURE: yes -- the resolved value is a launch constant baked into the
    # graph at capture (csrc/merlin/topk/persistent_topk.cuh,
    # PersistentTopKParams::exact_medium).
    # AOT: not established. Hashed conservatively.
    # NOT A PYTHON READ SITE: exact_medium_enabled() in
    # csrc/merlin/topk/topk.cu is folded on and reads no variable. This entry
    # is a raw passthrough kept for the hash; do not use
    # envs.MERLIN_TOPK_EXACT_MEDIUM to decide behaviour. Registered with the
    # medium-path exact restart so the cache-factor contract is complete.
    MERLIN_TOPK_EXACT_MEDIUM: str | None = None
    #
    # RUNTIME-ONLY (registered for validate_environ + a single cached read):
    # Static K/V scale sidecar: fill_()ed into existing fp32 device buffers
    # after load and before capture, and the kernel does tl.load(k_scale_ptr),
    # so the value is never a constexpr and the graph is identical.
    VLLM_QSA_KV_SCALES: str = ""
    # Guard telemetry: mode raise|warn and the reader poll interval in
    # seconds. counter() returns a real tensor in every mode, so the Triton
    # arguments and the captured graph do not depend on either.
    VLLM_E2_GUARD_MODE: str = "warn"
    VLLM_E2_GUARD_INTERVAL: str = "0.05"
    # Guard counter transport: "mapped" (default; counters in mapped page-locked
    # host memory, read without a D2H copy, fatal counts checked at every step
    # boundary) or "copy" (device counters, side-stream D2H reader).
    VLLM_E2_GUARD_TRANSPORT: str = "mapped"
    # Debug allocator telemetry (poll seconds + output directory).
    VLLM_E2_MEMSTATS: str = "0"
    VLLM_E2_MEMSTATS_DIR: str = ""
    # PLE offload child placement: rewrites CUDA_VISIBLE_DEVICES for the child
    # only (CUDA_VISIBLE_DEVICES is itself an ignored factor).
    VLLM_PLE_OFFLOAD_HOME_DEVICE: str = ""
    # PLE timeouts / deadlines, in seconds.
    VLLM_PLE_PROGRESS_DEADLINE_S: float = 300.0
    # PLE offload process only: compute the n-gram row IDs with NumPy instead
    # of ~50 tiny torch CPU ops (stage-0 graph stall). Same int64
    # arithmetic, same IDs, same rows; read once at import by
    # vllm/models/qwen4_exp/nvidia/ple_pull.py. AOT: no. CAPTURE: no -- the
    # offload process has no CUDA context. Strict "0"/"1", default off.
    VLLM_E2_PLE_FAST_IDS: bool = False
    # Stage-0 GPU worker: issue the PLE request (D2H of the step's PLE inputs +
    # the offload notification) right after prepare_inputs instead of just
    # before the forward, on real batches only. Same bytes, earlier on the
    # stream; dummy/profile/capture forwards keep the late issue, so no graph
    # changes. Read once at import by vllm/v1/worker/gpu/model_runner.py.
    # AOT: no. CAPTURE: no. Strict "0"/"1", default off.
    VLLM_E2_PLE_EARLY_REQUEST: bool = False
    #
    # The tie-determinism toggles are registered and
    # classified in the "determinism toggles" block below, and the
    # one knob still unclassified sits in that block's own fence, which
    # tests/test_envs.py reads as a set of identifiers.
    MERLIN_TOPK_EXACT_COUNTERS_LOG: int = 0
    # Pre-tie-determinism fallback for the counter reader's slot count; runtime-only
    # (ignored factor), never consulted on this build. See the dict entry.
    MERLIN_TOPK_COUNTER_SLOTS: str | None = None
    # Flash-Next family-name aliases, "alias=canonical[,alias=canonical...]"
    # (vllm/flashnext_family.py). Default unset/"" (the resolved name sets are
    # exactly the historical Qwen4Exp literals). flashnext_family is stdlib-only
    # (imported by the config layer and the model registry), so it parses
    # os.environ itself; this entry is the registration, not a second parser.
    # AOT: no. CAPTURE: no -- it only changes which architecture / model_type
    # names resolve to the family at config and registry time; no kernel, op or
    # captured graph depends on it. Runtime-only (ignored_factors).
    VLLM_FLASHNEXT_FAMILY_ALIASES: str = ""
    #
    # --- determinism toggles -----------------------------------------
    # FULL_K, QSA_SORT and TIE_DET/TIE_RECENT default to the shipped value 1
    # (SHIPPED_KERNEL_TOGGLE_DEFAULTS is the single source; vllm/env_override.py
    # puts it into the environment at import). All are hashed (none is in
    # ignored_factors) because each reaches a capture. Only MERLIN_QSA_SORT is
    # still read; FULL_K / TIE_DET / TIE_RECENT are folded on in C++ and these
    # entries are raw passthroughs for the hash, never a behaviour switch.
    #
    # Marlin MoE split-K reduction order; the CUDA half of unmerged upstream
    # PR #46639 (`use_full_k`).
    MERLIN_FULL_K: str | None = "1"
    # persistent_topk tie-break at the k-th cutoff, and its direction.
    MERLIN_TIE_DET: str | None = "1"
    MERLIN_TIE_RECENT: str | None = "1"
    #
    # A REAL Python read site, unlike the three above: parsed here and read
    # once at import by models/qwen4_exp/nvidia/ops/qsa_indexer.py, which adds
    # a torch.sort + two torch.where ops to the QSA selection path.
    MERLIN_QSA_SORT: bool = True
    # Reduced MTP draft vocabulary: path to a 1-D int ids table
    # (models/qwen4_exp/nvidia/draft_head.py). CAPTURE: yes (the FULL draft
    # graphs record an N-row head GEMM plus the full-vocab scatter); AOT: no
    # (compute_logits is outside the compiled submodule). Hashed conservatively:
    # compile_factors() keys the path AND a fingerprint of the table's content
    # (N + sha256 of the ids), so a different table at the same path is a
    # different identity. Default unset (OFF).
    VLLM_DRAFT_VOCAB_FILE: str | None = None
    # Optional sha256 (64 hex) of the VLLM_DRAFT_VOCAB_FILE bytes; when set the
    # load refuses a file that does not match. Integrity check only: not a
    # compile factor (ignored_factors). Default unset (no check).
    VLLM_DRAFT_VOCAB_SHA256: str | None = None
    # GDN in_proj_ba GEMM on the shared aux stream, overlapped with
    # in_proj_qkvz. Read at layer build by
    # model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py; CUDA only, and
    # serial under breakable CUDA graph capture. Same kernels, same inputs,
    # disjoint outputs, so the result is bit-identical. CAPTURE: yes (adds a
    # fork/join to the captured graph); default OFF, hashed by
    # compile_factors (not in ignored_factors).
    VLLM_GDN_BA_AUX_STREAM: bool = False
    #
    # --- persistent_topk kernel switches ------------------------------
    # Same classification as the three raw passthroughs above and registered
    # for the same single reason: compile_factors() is a deny-list, so a name
    # it does not know cannot be hashed, and a Ray deployment that propagates
    # registered variables can otherwise drop an explicitly-disabled switch
    # and leave a worker on the default. Each is folded in
    # csrc/merlin/topk/topk.cu (its *_enabled() getter returns a constant and
    # reads no variable); the value reaches the GPU as a kernel launch
    # argument, so a cudagraph bakes it in.
    # CAPTURE: yes, all three. AOT: not established (these kernels are built
    # into the extension, not by the AOT compiler); hashed conservatively
    # anyway, same rule as above. None is in ignored_factors.
    #
    # Raw passthroughs, so "unset" reads as None here; the kernel ignores the
    # variable either way. Read the getter in topk.cu for the shipped
    # behaviour (EXACT_MEDIUM, EXACT_DECODE and DET_LARGE on).
    #
    # Take the exact restart on an overflowing MEDIUM row:
    # MERLIN_TOPK_EXACT_MEDIUM. Declared ONCE, in the cross-change knob
    # block above (same raw passthrough, same hashing); the 0.30 tree already
    # carried that entry, so it is not repeated here.
    # The same restart on an overflowing DECODE row.
    MERLIN_TOPK_EXACT_DECODE: str | None = None
    # Deterministic placement and tie membership on the LARGE path.
    # Always on: det_large_enabled() in
    # csrc/merlin/topk/topk.cu is folded on (no variable read), and the value
    # reaches the large-path radix_topk as a launch argument, so a cudagraph
    # bakes it in. CAPTURE: yes. AOT: not established (built into the
    # extension, not by the AOT compiler); hashed conservatively like the
    # switches above.
    MERLIN_TOPK_DET_LARGE: str | None = None
    #
    # The large-path FAULT INJECTORS. TEST
    # ONLY, and registered for exactly the same reason as the switches above,
    # not because anything should set them in production:
    #
    #   * the shipped topk.cu folds both off (params.fault_inject_cta = -1,
    #     fault_inject_emit = false: no variable is read); a test build that
    #     wires them reaches the GPU through kernel launch arguments, so a
    #     cudagraph BAKES THEM IN. An injected build and a clean build differ
    #     in the captured graph while sharing a cache key -- which is the
    #     failure mode: a graph captured under injection returns an all -1 row
    #     for every deterministic large row, and on replay nothing re-reads the
    #     environment to notice the variable has since been unset. FAULT_INJECT
    #     was once "unregistered and capture-baked"; hashing
    #     them is what stops a cached artifact crossing between the arms.
    #   * compile_factors() is a deny-list, so a name it does not know cannot
    #     be hashed at all. Neither is in ignored_factors.
    #
    # Raw passthroughs, like the switches. Do not parse them here.
    #
    # Which CTA of the group has its fault predicate forced; unset / negative
    # is off, and setting it also suppresses __trap().
    MERLIN_TOPK_LARGE_FAULT_INJECT: str | None = None
    # Which fault: unset / "pre" forces the pre-emission predicate caught at
    # X3, "emit" forces the emission-bound slot check caught at X4. Ignored
    # unless MERLIN_TOPK_LARGE_FAULT_INJECT names a CTA.
    MERLIN_TOPK_LARGE_FAULT_KIND: str | None = None
    #
    # MERLIN_TOPK_SORTED_EMIT -- persistent_topk
    # emits every row in the canonical order MERLIN_QSA_SORT imposes
    # (ascending index, padding last as -1), and qsa_indexer.py then skips its
    # Python sort. Same classification as the kernel switches above: folded on
    # in csrc/merlin/topk/topk.cu (sorted_emit_enabled reads no variable), and
    # the value reaches the GPU as a kernel launch argument AND removes three torch
    # ops from the captured sequence, so a cudagraph bakes it in. CAPTURE: yes.
    # AOT: not established for the kernel half; hashed conservatively (a fork
    # env knob must be a compile factor: an os.environ knob that is not
    # registered is not in the cache key). Raw passthrough: Python never parses
    # it to decide anything -- qsa_indexer.py reads the C++ cache through
    # persistent_topk_toggle_state() field 6. Not in ignored_factors.
    MERLIN_TOPK_SORTED_EMIT: str | None = "1"
    #
    # MERLIN_MARLIN_MOE_CFG: per-shape Marlin MoE block_size_m and
    # thread config, chosen among kernels that are already compiled. Unset (the
    # default) = the engine's own heuristic, byte-identical. The raw string is
    # parsed in Python by vllm/merlin/marlin_moe_cfg.py (the single read site and
    # parser; this entry is a passthrough). Syntax and the exactness classes are
    # documented there. IGNORED as a compile factor, see MERLIN_IGNORED.
    MERLIN_MARLIN_MOE_CFG: str | None = None
    #
    # NOT REGISTERED: MERLIN_TOPK_EXACT_TRAP is a build-time -D define in
    # persistent_topk.cuh, not an environment variable.
    #
    # --- UNAUDITED: host-side knobs with no read site in this tree ----------
    # Names only, one per line, inside the fence; a knob leaves the fence when
    # its AOT and capture identity are written down against its read site
    # (tests/test_envs.py enforces this). MERLIN_PIN_BSM (Marlin block_size_m
    # pinned to 64) is an unused alternative to the split-K toggle above.
    # --8<-- [start:unaudited-host-side-knobs]
    #     MERLIN_PIN_BSM
    # --8<-- [end:unaudited-host-side-knobs]


# Compat flag for the 0/1 switch family. False (3.0.0): a value outside
# {"0", "1"} logs one warning naming the knob and the value, then keeps the
# interpretation that knob had before the helper existed, so no served
# behaviour changes. A later release flips this to True to make every
# out-of-set value a boot error.
STRICT_SWITCH_ENFORCE = False

_switch_warned: set[tuple[str, str]] = set()


def _e2_strict_switch(
    name: str,
    default: bool = False,
    *,
    legacy: str = "strict",
    none_if_unset: bool = False,
) -> bool | None:
    """Parse a fork on/off switch; the one parser for every Python 0/1 knob.

    Accepted: exactly ``"0"`` and ``"1"``; unset means ``default`` (``None``
    when ``none_if_unset``, for knobs where unset and explicit 0 differ).
    Anything else is handled per ``STRICT_SWITCH_ENFORCE``: raise when True;
    otherwise warn once and apply the knob's pre-helper reading, ``legacy``:
    ``"strict"`` raises (it always did), ``"int"`` is ``bool(int(raw))``
    (``"2"`` is on, ``"true"`` and ``""`` raise), ``"eq1"`` is
    ``raw == "1"`` (everything else off). An empty value is out of set like
    any other: it never silently means the default.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None if none_if_unset else default
    if raw == "0":
        return False
    if raw == "1":
        return True
    msg = f'{name} must be exactly "0" or "1" (or unset), got {raw!r}'
    if STRICT_SWITCH_ENFORCE:
        raise ValueError(msg)
    if (name, raw) not in _switch_warned:
        _switch_warned.add((name, raw))
        import logging

        logging.getLogger(__name__).warning(
            "%s; keeping the legacy reading (%s) for 3.0.0, "
            "this becomes a boot error in a later release",
            msg,
            legacy,
        )
    if legacy == "eq1":
        return raw == "1"
    if legacy == "int":
        try:
            return bool(int(raw))
        except ValueError:
            raise ValueError(msg) from None
    raise ValueError(msg)


def _e2_nonneg_int(name: str, default: int = 0) -> int:
    """Parse a fork knob that accepts only a plain non-negative decimal integer.

    Unset means ``default``. Anything else (``""``, ``" 16"``, ``"-1"``,
    ``"1.5"``) is a boot error.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not (raw.isascii() and raw.isdigit()):
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return int(raw)


def _e2_positive_int(name: str, default: int) -> int:
    value = _e2_nonneg_int(name, default)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


# compile_factors() key holding the draft-vocab table fingerprint (N + content hash).
DRAFT_VOCAB_TABLE_FACTOR = "VLLM_DRAFT_VOCAB_FILE:table"


def host_embed_table_enabled_for_model(architecture: str) -> bool:
    """Resolve the single knob: on for Qwen3.5 (dense and MoE) and Nemotron 3.5 Super VL, off otherwise."""
    # 27B series. Read through vllm.envs so the value comes
    # from the same (possibly cached) module __getattr__ as every other reader;
    # imported here, not at module level, to keep this module stdlib-only.
    import vllm.envs as envs

    setting = envs.VLLM_HOST_EMBED_TABLE
    if setting is not None:
        return setting
    return architecture in (
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "NemotronH_Omni_Reasoning_V3",  # Nemotron 3.5 Super VL
    )


# Shipped value of each determinism toggle. The registry getters default to
# these and vllm/env_override.py setdefaults them into os.environ at import,
# so there is one table, not two.
SHIPPED_KERNEL_TOGGLE_DEFAULTS: dict[str, str] = {
    "MERLIN_FULL_K": "1",
    "MERLIN_QSA_SORT": "1",
    "MERLIN_TIE_DET": "1",
    "MERLIN_TIE_RECENT": "1",
    "MERLIN_TOPK_SORTED_EMIT": "1",
}

MERLIN_ENV: dict[str, Callable[[], Any]] = {
    "VLLM_MERLIN_VISION_PEAK": lambda: _e2_strict_switch(
        "VLLM_MERLIN_VISION_PEAK", default=False, legacy="eq1"
    ),
    "VLLM_MERLIN_PROMPT_LOGPROBS_TENSOR_BUDGET_MIB": lambda: _e2_positive_int(
        "VLLM_MERLIN_PROMPT_LOGPROBS_TENSOR_BUDGET_MIB", 384
    ),
    "VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE": lambda: _e2_positive_int(
        "VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE", 256
    ),
    # 0 disables chunking; negative/malformed values fail closed.
    "VLLM_MERLIN_VISION_CHUNK_ROWS": lambda: _e2_nonneg_int(
        "VLLM_MERLIN_VISION_CHUNK_ROWS", default=8192
    ),
    # The execution gate always floors this at today's 8 MP cap.
    "VLLM_MERLIN_VISION_CHUNK_MIN_PIXELS": lambda: _e2_nonneg_int(
        "VLLM_MERLIN_VISION_CHUNK_MIN_PIXELS", default=8_388_608
    ),
    # QSA prefill indexer logits backing granule in MiB (default 0 = off: the
    # logits are allocated at exactly the size each chunk needs). A positive
    # value rounds the allocation up to a multiple of it (16-64 recommended) and
    # reuses one backing for every chunk of a call. Strict non-negative integer.
    "VLLM_E2_QSA_LOGITS_GRANULE_MB": lambda: _e2_nonneg_int(
        "VLLM_E2_QSA_LOGITS_GRANULE_MB"
    ),
    # --- 27B series knobs; getters moved
    # verbatim from the series' vllm/envs.py. ---
    # --- FlashInfer decode-plan cache (fa2/sm_86) ---
    # Single-entry cache that skips fast_decode_plan when the schedule cannot have
    # changed. On by default; VLLM_FLASHINFER_PLAN_CACHE=0 disables it. Registered here
    # (not read from os.environ) so it is part of the compile cache key.
    "VLLM_FLASHINFER_PLAN_CACHE": lambda: _e2_strict_switch(
        "VLLM_FLASHINFER_PLAN_CACHE", default=True, legacy="int"
    ),
    # --- GDN in_proj_ba split-K (sm_86, small M) ---
    # Triton split-K GEMM for the unquantized GDN in_proj_ba at M <= 16 tokens. On by
    # default; VLLM_GDN_BA_SPLITK=0 keeps cuBLAS. Registered here (not read from
    # os.environ) so it is part of the compile cache key.
    "VLLM_GDN_BA_SPLITK": lambda: _e2_strict_switch(
        "VLLM_GDN_BA_SPLITK", default=True, legacy="int"
    ),
    # --- MTP draft-step seq_lens without a device sync ---
    # Draft step >= 2 takes the previous draft build's CPU seq_lens + 1 instead of a
    # blocking seq_lens.cpu(). On by default; VLLM_DRAFT_SEQ_LENS_NOSYNC=0 disables it.
    # VLLM_DRAFT_SEQ_LENS_VERIFY=1 also syncs, compares, logs mismatches and uses the
    # synced value (diagnostic; off by default). Registered here (not read from
    # os.environ) so both are part of the compile cache key.
    "VLLM_DRAFT_SEQ_LENS_NOSYNC": lambda: _e2_strict_switch(
        "VLLM_DRAFT_SEQ_LENS_NOSYNC", default=True, legacy="int"
    ),
    # None when unset so unset and explicit 0 are distinct compile factors.
    "VLLM_DRAFT_SEQ_LENS_VERIFY": lambda: _e2_strict_switch(
        "VLLM_DRAFT_SEQ_LENS_VERIFY", legacy="int", none_if_unset=True
    ),
    # --- MTP draft decode planned from CPU bounds, no seq_lens sync (draft step 1+) ---
    # Plans the FlashInfer fa2 draft decode from the CPU upper bound on seq_lens when the
    # bounds provably give the same schedule as the true lengths, and writes the true page
    # table on the GPU; otherwise syncs exactly as before. Off by default;
    # VLLM_DRAFT_BOUNDED_PLAN=1 enables it. VLLM_DRAFT_BOUNDED_PLAN_VERIFY=1 also syncs,
    # checks bounds / schedule / page table against the synced lengths, logs mismatches and
    # uses the exact path on a mismatch (diagnostic). Both are host-side builder knobs and
    # are listed in compile_factors' ignored set, so a flip reuses the AOT cache.
    "VLLM_DRAFT_BOUNDED_PLAN": lambda: _e2_strict_switch(
        "VLLM_DRAFT_BOUNDED_PLAN", default=False, legacy="int"
    ),
    # None when unset (diagnostic knob; not a compile factor, see compile_factors).
    "VLLM_DRAFT_BOUNDED_PLAN_VERIFY": lambda: _e2_strict_switch(
        "VLLM_DRAFT_BOUNDED_PLAN_VERIFY", legacy="int", none_if_unset=True
    ),
    # --- Target (verify) FlashInfer build without the seq_lens sync ---
    # Right after post_update the runner enqueues a pinned non-blocking D2H copy of
    # num_computed_tokens + an event; the next verify build waits on that event only (not on
    # the draft steps queued behind it) and computes the EXACT seq_lens on the CPU (same
    # values seq_lens.cpu() returns, so the page table and FlashInfer plan are identical).
    # Off by default; VLLM_TARGET_SEQ_LENS_ASYNC=1 enables it. _VERIFY=1 also syncs, compares
    # against seq_lens.cpu(), logs and uses the synced lengths on a mismatch (diagnostic).
    # Registered here (not read from os.environ) so both are part of the compile cache key.
    "VLLM_TARGET_SEQ_LENS_ASYNC": lambda: _e2_strict_switch(
        "VLLM_TARGET_SEQ_LENS_ASYNC", default=False, legacy="int"
    ),
    # None when unset so unset and explicit 0 are distinct compile factors.
    "VLLM_TARGET_SEQ_LENS_ASYNC_VERIFY": lambda: _e2_strict_switch(
        "VLLM_TARGET_SEQ_LENS_ASYNC_VERIFY", legacy="int", none_if_unset=True
    ),
    # See its declaration. Default off.
    "VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE": lambda: _e2_strict_switch(
        "VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE", default=False, legacy="int"
    ),
    # --- GDN metadata host trims (see their declaration). Default off. ---
    # VLLM_GDN_SHARED_ALIGN_INDEX=1: the align-mode state-index gather index is
    # computed once per batch and reused by every GDN group (update_block_table)
    # instead of being re-derived per group.
    "VLLM_GDN_SHARED_ALIGN_INDEX": lambda: _e2_strict_switch(
        "VLLM_GDN_SHARED_ALIGN_INDEX", default=False, legacy="int"
    ),
    # VLLM_GDN_DECODE_FASTPATH=1: pure spec-decode FULL-graph batches whose spec
    # rows are a prefix of the batch stage the GDN metadata with slices and
    # fills instead of CPU-mask advanced indexing and H2D copies, and skip the
    # accepted-count work RecoverSSM overrides with ones.
    "VLLM_GDN_DECODE_FASTPATH": lambda: _e2_strict_switch(
        "VLLM_GDN_DECODE_FASTPATH", default=False, legacy="int"
    ),
    # RecoverSSM commit execution mode (see its declaration); runtime only.
    "VLLM_RECOVERSSM_COMMIT_GRAPH": lambda: os.getenv(
        "VLLM_RECOVERSSM_COMMIT_GRAPH", ""
    ).strip(),
    # --- end of the 27B series knobs ---
    # --- Opt-in Qwen3.5 MoE / single-card memory knobs (off by default) ---
    "MERLIN_MOE_UNIFORM_VERIFY": lambda: _e2_strict_switch("MERLIN_MOE_UNIFORM_VERIFY"),
    "MERLIN_SHARE_DRAFT_WORKSPACE": lambda: _e2_strict_switch(
        "MERLIN_SHARE_DRAFT_WORKSPACE"
    ),
    "VLLM_MERLIN_VISION_ROTARY_INPLACE": lambda: _e2_strict_switch(
        "VLLM_MERLIN_VISION_ROTARY_INPLACE"
    ),
    "VLLM_HOST_EMBED_TABLE": lambda: _e2_strict_switch(
        "VLLM_HOST_EMBED_TABLE", legacy="eq1", none_if_unset=True
    ),
    # Timeout for PLE weight loading and TP worker registration.
    "VLLM_PLE_OFFLOAD_READY_TIMEOUT": lambda: float(
        os.getenv("VLLM_PLE_OFFLOAD_READY_TIMEOUT", "600")
    ),
    # Draft top-k/top-p truncation switch; see the declaration.
    "VLLM_SPEC_DRAFT_UNDER_TARGET_TOPK": lambda: _e2_strict_switch(
        "VLLM_SPEC_DRAFT_UNDER_TARGET_TOPK", default=False
    ),
    # Step-keyed block-verification draws; see the declaration.
    "VLLM_SPEC_STEP_KEYED_RNG": lambda: _e2_strict_switch(
        "VLLM_SPEC_STEP_KEYED_RNG", default=False
    ),
    # --- Flash-Next fork knobs; see the TYPE_CHECKING block for
    # which of these are compile/capture factors and which are ignored.
    # Compile/capture factors.
    "VLLM_QSA_KV_CLIP_COUNT": lambda: os.getenv("VLLM_QSA_KV_CLIP_COUNT", "").strip(),
    # Capture-configuration factor, folded in C++; raw passthrough for the hash.
    "MERLIN_TOPK_EXACT_MEDIUM": lambda: os.getenv("MERLIN_TOPK_EXACT_MEDIUM", None),
    # Runtime only; all of the following are in compile_factors.ignored_factors.
    "VLLM_QSA_KV_SCALES": lambda: os.getenv("VLLM_QSA_KV_SCALES", "").strip(),
    "VLLM_E2_GUARD_MODE": lambda: (
        os.getenv("VLLM_E2_GUARD_MODE", "warn").strip().lower()
    ),
    "VLLM_E2_GUARD_INTERVAL": lambda: os.getenv("VLLM_E2_GUARD_INTERVAL", "0.05"),
    "VLLM_E2_GUARD_TRANSPORT": lambda: (
        os.getenv("VLLM_E2_GUARD_TRANSPORT", "mapped").strip().lower()
    ),
    "VLLM_E2_MEMSTATS": lambda: os.getenv("VLLM_E2_MEMSTATS", "0"),
    "VLLM_E2_MEMSTATS_DIR": lambda: os.getenv("VLLM_E2_MEMSTATS_DIR", ""),
    "VLLM_PLE_OFFLOAD_HOME_DEVICE": lambda: os.getenv(
        "VLLM_PLE_OFFLOAD_HOME_DEVICE", ""
    ).strip(),
    "VLLM_PLE_PROGRESS_DEADLINE_S": lambda: float(
        os.getenv("VLLM_PLE_PROGRESS_DEADLINE_S", "").strip() or "300"
    ),
    # PLE offload process: NumPy n-gram ID path (default off). Strict "0"/"1".
    "VLLM_E2_PLE_FAST_IDS": lambda: _e2_strict_switch(
        "VLLM_E2_PLE_FAST_IDS", default=False
    ),
    # Stage-0 runner: early PLE request after prepare_inputs (default off).
    "VLLM_E2_PLE_EARLY_REQUEST": lambda: _e2_strict_switch(
        "VLLM_E2_PLE_EARLY_REQUEST", default=False
    ),
    # Seconds between `topk-exact-counters` lines from each GPU worker
    # (0, the default, disables the reader entirely). The line reports the
    # medium-path exact-restart counters; see
    # vllm/models/qwen4_exp/nvidia/ops/topk_counters.py.
    "MERLIN_TOPK_EXACT_COUNTERS_LOG": lambda: int(
        os.getenv("MERLIN_TOPK_EXACT_COUNTERS_LOG", "0")
    ),
    # Fallback slot count for the counter reader on a build WITHOUT the
    # persistent_topk_exact_counters_slots op (pre-tie-determinism). This tree registers
    # that op, so topk_counters.slot_count() never consults the variable here.
    # Raw passthrough: the reader's int(os.getenv(..., "5")) is the parser.
    "MERLIN_TOPK_COUNTER_SLOTS": lambda: os.getenv("MERLIN_TOPK_COUNTER_SLOTS", None),
    # Family-name aliases; raw passthrough, flashnext_family.py is the parser.
    "VLLM_FLASHNEXT_FAMILY_ALIASES": lambda: os.getenv(
        "VLLM_FLASHNEXT_FAMILY_ALIASES", ""
    ),
    # Determinism toggles. See the declaration block above for the AOT /
    # capture classification and for why three of these are raw passthroughs.
    # All four are absent from ignored_factors on purpose, so all four are
    # hashed into the compile cache key.
    "MERLIN_FULL_K": lambda: os.getenv(
        "MERLIN_FULL_K", SHIPPED_KERNEL_TOGGLE_DEFAULTS["MERLIN_FULL_K"]
    ),
    "MERLIN_TIE_DET": lambda: os.getenv(
        "MERLIN_TIE_DET", SHIPPED_KERNEL_TOGGLE_DEFAULTS["MERLIN_TIE_DET"]
    ),
    "MERLIN_TIE_RECENT": lambda: os.getenv(
        "MERLIN_TIE_RECENT", SHIPPED_KERNEL_TOGGLE_DEFAULTS["MERLIN_TIE_RECENT"]
    ),
    "MERLIN_QSA_SORT": lambda: _e2_strict_switch(
        "MERLIN_QSA_SORT", default=True, legacy="int"
    ),
    # Reduced draft vocabulary; empty reads as unset, like draft_head.py.
    "VLLM_DRAFT_VOCAB_FILE": lambda: os.getenv("VLLM_DRAFT_VOCAB_FILE") or None,
    "VLLM_DRAFT_VOCAB_SHA256": lambda: os.getenv("VLLM_DRAFT_VOCAB_SHA256") or None,
    # GDN in_proj_ba on the aux stream; see the declaration block above.
    "VLLM_GDN_BA_AUX_STREAM": lambda: _e2_strict_switch(
        "VLLM_GDN_BA_AUX_STREAM", default=False
    ),
    # persistent_topk kernel switches. Raw passthroughs kept for the hash
    # (the C++ is folded and reads none of them); see the declaration block.
    # MERLIN_TOPK_EXACT_MEDIUM is registered once, earlier in this dict.
    "MERLIN_TOPK_EXACT_DECODE": lambda: os.getenv("MERLIN_TOPK_EXACT_DECODE", None),
    "MERLIN_TOPK_DET_LARGE": lambda: os.getenv("MERLIN_TOPK_DET_LARGE", None),
    # Large-path fault injectors. TEST ONLY, hashed for
    # the reason in the declaration block: both are capture-baked.
    "MERLIN_TOPK_LARGE_FAULT_INJECT": lambda: os.getenv(
        "MERLIN_TOPK_LARGE_FAULT_INJECT", None
    ),
    "MERLIN_TOPK_LARGE_FAULT_KIND": lambda: os.getenv(
        "MERLIN_TOPK_LARGE_FAULT_KIND", None
    ),
    # In-kernel canonical emission; see the declaration block. Hashed: it
    # changes the captured sequence.
    "MERLIN_TOPK_SORTED_EMIT": lambda: os.getenv(
        "MERLIN_TOPK_SORTED_EMIT",
        SHIPPED_KERNEL_TOGGLE_DEFAULTS["MERLIN_TOPK_SORTED_EMIT"],
    ),
    # Marlin MoE per-shape block size / thread config. Raw
    # passthrough; marlin_moe_cfg.py parses it.
    "MERLIN_MARLIN_MOE_CFG": lambda: os.getenv("MERLIN_MARLIN_MOE_CFG", None),
    # Prefix-cache thrash fixes (default ON). Read in cache_thrash.py.
    "VLLM_E2_SCHED_MISS_DEFER_S": lambda: _e2_nonneg_int(
        "VLLM_E2_SCHED_MISS_DEFER_S", default=120
    ),
    "VLLM_E2_SCHED_COHERENT_YIELD": lambda: _e2_strict_switch(
        "VLLM_E2_SCHED_COHERENT_YIELD", default=True
    ),
    # Same semantics as the former os.environ read: only "0" disables.
    "VLLM_SPEC_PROFILE_SHAPES": lambda: (
        os.getenv("VLLM_SPEC_PROFILE_SHAPES", "1") != "0"
    ),
    "MERLIN_ALLOW_PROFILING_KV_RETENTION": lambda: _e2_strict_switch(
        "MERLIN_ALLOW_PROFILING_KV_RETENTION"
    ),
}


MERLIN_IGNORED: frozenset[str] = frozenset(
    {
    # Eager prompt scoring after forward; no compiled or captured operations.
    "VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE",
    "VLLM_MERLIN_PROMPT_LOGPROBS_TENSOR_BUDGET_MIB",
    # AOT: no -- row slicing is eager, outside compiled decoder regions.
    # CAPTURE: no -- encoder graphs omit the grid and keep plain execution.
    "VLLM_MERLIN_VISION_CHUNK_ROWS",
    "VLLM_MERLIN_VISION_CHUNK_MIN_PIXELS",
    # AOT: no -- eager tower dispatch; compiled vision blocks keep the plain path.
    # CAPTURE: no -- encoder graphs omit the grid and keep plain execution.
    "VLLM_MERLIN_VISION_PEAK",
    # AOT: no -- buffer ownership only. CAPTURE: the drafter's graphs record the shared
    # pointer instead of a second buffer; same kernels and shapes.
    "MERLIN_SHARE_DRAFT_WORKSPACE",
    # AOT: no -- read only by the FlashInfer metadata builder (cudagraph-support
    # class and reorder threshold), outside compiled regions. CAPTURE: yes, like
    # VLLM_DRAFT_BOUNDED_PLAN.
    "MERLIN_MOE_UNIFORM_VERIFY",
    # AOT: no -- eager tower path only. CAPTURE: no -- encoder graphs keep the plain
    # path.
    "VLLM_MERLIN_VISION_ROTARY_INPLACE",
    # --- Flash-Next fork knobs that are runtime-only ------
    # Registering a name in environment_variables makes it a compile
    # factor by default (this function is a deny-list), so every fork knob
    # that cannot change a compiled or captured artifact is listed here
    # with its reason. Hashing them would invalidate the whole AOT cache on
    # a log path, a timeout or a guard-mode flip.
    #
    # Static QSA K/V scales: fill_()ed into existing fp32 device buffers
    # after load and before capture; the kernel reads them with
    # tl.load(k_scale_ptr), never as a constexpr.
    "VLLM_QSA_KV_SCALES",
    # Guard telemetry: counter() returns a real [3] int64 view in every
    # mode, so the Triton arguments and the captured graph are identical;
    # the mode only gates the side reader thread and the check_pending
    # raise, and the interval is that thread's poll cadence; the idle
    # switch only decides whether that thread parks while idle.
    # Boot-time integrity check on the draft vocab file; the table content
    # is already keyed through DRAFT_VOCAB_TABLE_FACTOR.
    "VLLM_DRAFT_VOCAB_SHA256",
    "VLLM_E2_GUARD_MODE",
    "VLLM_E2_GUARD_INTERVAL",
    # Transport only moves the counter buffer to mapped host memory: the
    # guarded kernels, their arguments, Triton specialisation (16-byte
    # aligned pointer either way) and the captured graph topology are
    # unchanged; a clean step touches no counter. Not a compile factor.
    "VLLM_E2_GUARD_TRANSPORT",
    # Debug telemetry cadence and output locations.
    "VLLM_E2_MEMSTATS",
    "VLLM_E2_MEMSTATS_DIR",
    # Scheduler-side admission order and yield policy; the scheduler is never
    # compiled and the worker sees ordinary block tables either way, so an A/B
    # flip reuses one AOT cache.
    "VLLM_E2_SCHED_MISS_DEFER_S",
    "VLLM_E2_SCHED_COHERENT_YIELD",
    # Startup profiling only: extra dummy forwards before KV sizing, and the
    # severity of a post-profiling teardown check; neither touches a compiled
    # or captured artifact.
    "VLLM_SPEC_PROFILE_SHAPES",
    "MERLIN_ALLOW_PROFILING_KV_RETENTION",
    # Placement only: rewrites CUDA_VISIBLE_DEVICES for the PLE offload
    # child, and CUDA_VISIBLE_DEVICES is already ignored above.
    "VLLM_PLE_OFFLOAD_HOME_DEVICE",
    # Timeouts and deadlines, in seconds. VLLM_PLE_OFFLOAD_READY_TIMEOUT
    # is not a compile factor: a launcher may set it (e.g. to 1800),
    # and hashing it would make every boot that changed it
    # miss the AOT cache for no reason.
    "VLLM_PLE_OFFLOAD_READY_TIMEOUT",
    "VLLM_PLE_PROGRESS_DEADLINE_S",
    # MERLIN_TOPK_EXACT_MEDIUM is deliberately NOT here: it is a capture
    # configuration factor and is hashed conservatively.
    # PLE offload process only (no CUDA context there): selects how the
    # host computes n-gram row IDs, not what any kernel or graph runs.
    "VLLM_E2_PLE_FAST_IDS",
    # Eager host ordering of the real-batch PLE request only: capture and
    # dummy forwards keep the late issue, so the compiled artifact and
    # every captured graph are the same with it on or off.
    "VLLM_E2_PLE_EARLY_REQUEST",
    # Eager allocation size of the QSA prefill logits (rounded up, reused
    # across chunks). The op runs outside capture, the view handed to the
    # kernel has the same shape, strides and dtype, so no compiled artifact
    # or captured graph depends on it.
    "VLLM_E2_QSA_LOGITS_GRANULE_MB",
    # Draft top-k/top-p: AOT: no (sample_draft is outside the compiled regions). CAPTURE: yes
    # (FULL draft graphs record the truncation), but the flag is read once at
    # speculator init, before capture, and graphs are never persisted, so there
    # is no cross-process artifact to key; hashing it would only force an AOT
    # miss on every configuration. Kept ignored.
    "VLLM_SPEC_DRAFT_UNDER_TARGET_TOPK",
    # Step-keyed draws: identical classification to the draft top-k/top-p switch above. AOT: no (the rejection
    # kernels and sample_draft are outside the compiled regions). CAPTURE: yes
    # (the FULL draft graphs record the STEP_KEYED draft sampler), but the flag
    # is read once at init, before capture, and graphs are never persisted, so
    # no cross-process artifact depends on it; hashing it would force an AOT
    # miss on every configuration for an RNG-keying A/B.
    "VLLM_SPEC_STEP_KEYED_RNG",
    # Logging cadence for an already-compiled kernel's counters: it gates a
    # host-side timer and a snapshot op, and is read by no compiled code, so
    # hashing it would invalidate every compile cache for a log interval.
    "MERLIN_TOPK_EXACT_COUNTERS_LOG",
    # Host-side reader fallback (buffer length for a snapshot op outside
    # capture); dead on a build that answers the _slots query.
    "MERLIN_TOPK_COUNTER_SLOTS",
    # Marlin MoE per-shape block size / thread config. AOT: no. The
    # choice is made in the Python body of fused_marlin_moe, which runs inside
    # the opaque vllm.moe_forward custom op (moe_runner.py: "opaque custom ops is
    # a load-bearing assumption"), so the Inductor/AOT artifact contains no
    # trace of it, and the kernels it selects among are all compiled into the
    # extension already. CAPTURE: yes -- a cudagraph captured for a given M bakes
    # in the block size and launch config -- but the value is fixed for the
    # process, read on every call, and cudagraphs are never persisted, so there
    # is no cross-process artifact to key. Hashing it would only move the
    # release AOT cache key (even when unset) for no protection. Same argument
    # as VLLM_SPEC_DRAFT_UNDER_TARGET_TOPK above.
    "MERLIN_MARLIN_MOE_CFG",
    # --- 27B series knobs: the same ignored
    # set the series' compile_factors() listed, with its reasons. ---
    # RecoverSSM commit order (after vs before the MTP proposal): eager Python
    # call order after sampling; nothing compiled or captured depends on it.
    "VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE",
    # RecoverSSM commit execution mode: read only after sampling, outside the
    # compiled forward and its capture; the commit graphs it selects are
    # in-process and bit-identical to the eager chain (see its declaration).
    "VLLM_RECOVERSSM_COMMIT_GRAPH",
    # GDN metadata host trims: host-side builder work only. The staged
    # buffers keep their addresses and get the same values, and nothing
    # inside the compiled forward or a captured graph reads the flags.
    "VLLM_GDN_SHARED_ALIGN_INDEX",
    "VLLM_GDN_DECODE_FASTPATH",
    # MTP draft bounded plan: decides on the host whether a draft build plans
    # FlashInfer from CPU bounds (and writes the page table with a Triton kernel
    # outside capture); nothing compiled or captured reads either flag, and
    # registering them must not move the release AOT cache key.
    "VLLM_DRAFT_BOUNDED_PLAN",
    "VLLM_DRAFT_BOUNDED_PLAN_VERIFY",
    # Target seq_lens async snapshot: host-side pinned D2H + event before the drafts; read by the
    # FlashInfer metadata build on the host only -- nothing compiled or captured reads it, and it must not
    # move the release AOT cache key (flags off/on compiled separate artefacts -> token drift).
    "VLLM_TARGET_SEQ_LENS_ASYNC",
    "VLLM_TARGET_SEQ_LENS_ASYNC_VERIFY",
    # Family-name aliases: resolved to architecture / model_type names at
    # config and registry time; no compiled or captured artifact depends on it.
    "VLLM_FLASHNEXT_FAMILY_ALIASES",
    # ------------------------------------------------------------------
    }
)


def add_merlin_compile_factors(factors: dict[str, object]) -> None:
    """Add the fork's derived compile factors to ``factors`` (in place)."""
    # Draft vocab: the path alone does not identify the table (it can be replaced in
    # place), so key its content too. The fingerprint is the one recorded when
    # the table was loaded, else it is read from the file now.
    draft_vocab_file = factors.get("VLLM_DRAFT_VOCAB_FILE")
    if draft_vocab_file:
        from vllm.models.qwen4_exp.nvidia.draft_head import draft_vocab_table_factor

        factors[DRAFT_VOCAB_TABLE_FACTOR] = draft_vocab_table_factor(
            str(draft_vocab_file)
        )


def read_registered(name: str) -> Any:
    """Resolve one registered fork knob through its getter (no module cache)."""
    return MERLIN_ENV[name]()
