#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one Merlin launcher (Merlin 3.0.0): (checkpoint, profile, env overrides) -> ``exec vllm serve``.

    python3 tools/merlin/launch.py --profile <id> [--kv fp8|bf16] [--checkpoint <name>] -- <checkpoint-dir> [extra vllm args]

Every ``models/<m>/serve*.sh`` and every file under ``tools/merlin/launchers/`` is a shim that calls this module with
its profile id; ``merlin serve`` resolves the profile from the hardware and calls it too. The per-profile values come
from ``tools/merlin/profiles/*.json`` (registry.py); the family argv shapes (Flash-Next, 27B, Super VL, Ornith) and the
shared start-up logic (toolchain and CUDA_HOME default, build-product check, PLE readiness, image cap, topology
selection) live here, once.

Standard library only (Python >= 3.10); no GPU needed until ``vllm`` starts. ``SERVE_PRINT_ENV_AND_EXIT=1`` prints
the final platform decision and the vllm flags instead of starting the server (test hook).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry as R  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# checkpoint name -> (registry model, recipe directory)
CHECKPOINTS: dict[str, tuple[str, str]] = {
    "flash-next": ("flash-next", "models/flash-next"),
    "swift1.5": ("flash-next", "models/swift1.5"),
    "swift1.5-huihui": ("flash-next", "models/swift1.5-huihui"),
    "27b": ("27b", "models/qwen38-27b"),
    "27b-thinkingcap": ("27b", "models/thinkingcap"),
    "nemotron-super-vl": ("svl", "models/nemotron-super-vl"),
    "ornith1.5": ("ornith", "models/ornith1.5"),
}
FAMILY = {"flash-next": "flash-next", "27b": "27b", "svl": "svl", "ornith": "ornith"}

# Family constants: values no profile or checkpoint has ever varied (the engine's CLI contract per family).
FN_GENERATION = '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
CKPT_SCALES = "qsa_kv_scales.json"  # Flash-Next family: a checkpoint's own FP8 KV scales, next to its weights
FN_CHAT_TEMPLATE ="models/qwen38-27b/templates/froggeric-qwen-v22.5.jinja"  # shared with 27B / ThinkingCap
FN_KERNEL_CONFIG = '{"qsa_warmup_use_serving_bounds":true,"reuse_gdn_prefill_warmup":true,"persist_triton_autotune":true}'
FN_UNSET = (
    "VLLM_PLE_CPU_OFFLOAD VLLM_PLE_FP8_GLOBAL_SCALE VLLM_PLE_TEST_FAULT VLLM_PLE_TEST_FAULT_GATE VLLM_PLE_TEST_FAULT_LOG "
    "VLLM_QSA_KV_COLLECT VLLM_QSA_KV_CLIP_COUNT VLLM_E2_ALLOW_BREAKABLE VLLM_E2_EXPECTED_CUDAGRAPH_MODE "
    "VLLM_E2_MAMBA_SLOT_OWNERSHIP VLLM_E2_MEMSTATS VLLM_E2_MEMSTATS_DIR VLLM_E2_PLE_FLAG_CHECK VLLM_E2_SCHED_LOG "
    "VLLM_E2_ZERO_OWNERSHIP_CHECK VLLM_PLE_OFFLOAD_WAIT_TIMEOUT VLLM_PLE_PROGRESS_DEADLINE_S "
    "VLLM_PLE_STARTUP_DEADLINE_S PYTORCH_CUDA_ALLOC_CONF"
).split()
B27_INDUCTOR = '{"test_configs.force_filter_reduction_configs":true,"dynamic_scale_rblock":false}'
B27_GENERATION = FN_GENERATION
# Ornith 1.5: sampling comes from the checkpoint's generation_config (T 0.6 / top_p 0.95 / top_k 20); only the penalties are set
ORN_GENERATION = '{"presence_penalty":0.0,"repetition_penalty":1.0}'
ORN_KERNEL_CONFIG = '{"reuse_gdn_prefill_warmup":true,"persist_triton_autotune":true}'
# Flash-Next host memory: the ~48 GiB n-gram table + ~4.2 GiB pinned embeddings per server.
FN_RAM_QUALIFIED_KB = 100663296  # 96 GB
FN_RAM_FLOOR_KB = 72351744  # ~69 GiB resident floor
FN_SHM_MIN_KB = 1048576
HOOK_VARS = (
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_SKIP_P2P_CHECK", "VLLM_PP_LAYER_PARTITION",
    "NCCL_P2P_LEVEL", "NCCL_P2P_DISABLE", "NCCL_PROTO", "NCCL_ALGO", "VLLM_QSA_KV_SCALES",
    "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC",
)


class LaunchError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


@dataclass
class Plan:
    model: str
    checkpoint: str  # checkpoint name
    profile: dict[str, Any]
    ckpt: str  # checkpoint directory
    env: dict[str, str]
    args: list[str]  # the vllm flags between the checkpoint and the extra args
    extra: list[str]
    repo: Path
    venv: str
    tree: str | None
    cache_root: str
    status: str
    p2p_mode: str | None = None
    prefix: list[str] = field(default_factory=list)  # topology launch prefix (numactl ...), selector on only
    notes: list[str] = field(default_factory=list)  # stderr notes produced while resolving defaults
    files: list[tuple[str, str]] = field(default_factory=list)  # (path, refusal message) checked before exec
    draft_vocab: tuple[str, str] | None = None  # (table path, expected sha256) under REDUCED_DRAFT_VOCAB=1
    topo: dict[str, Any] = field(default_factory=dict)  # Flash-Next: AUTO_TOPO decision inputs
    caller_env: dict[str, str] = field(default_factory=dict)  # the environment as the caller gave it

    @property
    def family(self) -> str:
        return FAMILY[self.model]

    def argv(self) -> list[str]:
        return [*self.prefix, f"{self.venv}/bin/vllm", "serve", self.ckpt, *self.args, *self.extra]


RETIRED_KNOBS = {  # read by nothing; a value in the environment is a stale shell, say so
    "VLLM_E2_GUARD_IDLE": "guard idle parking is built in (v2.5.2)",
    "VLLM_E2_PLE_PULL_TRANSPORT": "the pull transport is the only PLE transport",
    "VLLM_E2_SIDE_CACHE_LAYOUT": "the side-cache layout is always on",
    "VLLM_E2_MAMBA_RETAIN_CHECKPOINT": "checkpoint retention is always on",
    "VLLM_MROPE_CACHE_1X": "the 1x M-RoPE cache is automatic for images",
}
# Knobs a user's shell may carry that the Flash-Next profile sets itself (or leaves at the engine default):
# dropped on top of MERLIN_* and FN_UNSET so a stale export cannot change a qualified profile.
FN_RESET_EXTRA = ("VLLM_DRAFT_BOUNDED_PLAN", "VLLM_MERLIN_VISION_PEAK", "VLLM_GDN_BA_AUX_STREAM", "VLLM_E2_GUARD_TRANSPORT")


def distributed_ports(port: str) -> tuple[int, int]:
    """MASTER_PORT / VLLM_PORT for one server from its HTTP PORT (vLLM also takes VLLM_PORT + 1).

    PORT 8000..8355 maps to 29500+d / 30000+100d (the values the shipped profiles use). Any other PORT p, with r = p % 10000:
    below 20000 -> 20000+r / 40000+2r; 20000..39999 -> 10000+r / 40000+2r; 40000 and up -> 20000+r / 30000+2(p % 5000).
    Each band lies outside the HTTP range it serves, so the three ports never meet the HTTP port, and consecutive
    PORTs (``merlin serve --instances``) get disjoint sets; all stay below 65536.
    """
    try:
        p = int(port)
    except ValueError:
        raise LaunchError(f"PORT must be a whole number (got '{port}')", 2) from None
    if not 1024 <= p <= 65535:
        raise LaunchError(f"PORT must be between 1024 and 65535 (got {p})", 2)
    d, r = p - 8000, p % 10000
    if 0 <= d < 356:
        return 29500 + d, 30000 + d * 100
    if p < 20000:
        return 20000 + r, 40000 + 2 * r
    if p < 40000:
        return 10000 + r, 40000 + 2 * r
    return 20000 + r, 30000 + 2 * (p % 5000)


def warn_retired(env: dict[str, str]) -> None:
    for k, why in RETIRED_KNOBS.items():
        if k in env:
            print(f"note: {k}={env[k]!r} is a retired knob and is ignored ({why})", file=sys.stderr)


def _knob(env: dict[str, str], name: str, default: str, assign: bool, keep_empty: bool = False) -> str:
    """Read an env knob with bash's semantics: ``assign`` = ``NAME=${NAME:-default}`` (an exported variable keeps its
    export with the resolved value), else inline ``${NAME:-default}`` (environment untouched); ``keep_empty`` =
    ``${NAME-default}`` (set-but-empty stays empty)."""
    present = name in env
    raw = env.get(name)
    if raw is None or (raw == "" and not keep_empty):
        val = default
    else:
        val = raw
    if assign and present:
        env[name] = val
    return val


def _digits(v: str) -> bool:
    return v != "" and v.isdigit()


# --------------------------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------------------------
def load_profile(model: str, profile_id: str, kv: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    reg = R.load_registry(model)
    p = R.get_profile(reg, profile_id)
    if kv:
        dtype = {v: k for k, v in R.KV_TAG.items()}[kv]
        p = R.variant_profile(p, dtype)
    return reg, p


def default_checkpoint(model: str, profile: dict[str, Any]) -> str:
    if model == "flash-next":
        return profile.get("checkpoint") or "flash-next"
    return {"27b": "27b", "svl": "nemotron-super-vl", "ornith": "ornith1.5"}[model]


def build(
    model: str,
    profile_id: str,
    ckpt: str,
    environ: dict[str, str],
    extra: list[str] | None = None,
    kv: str | None = None,
    checkpoint: str | None = None,
    repo: Path | None = None,
    status: str | None = None,
) -> Plan:
    """Resolve env + argv for one profile from the registry and the caller's environment. ``status`` is the label
    merlin serve evaluated on the actual cards (``untested`` for a tested row on other names or without P2P); the
    registry row's own status otherwise.

    Pure apart from the directory-existence reads behind the Flash-Next VENV / CACHE_ROOT fallbacks; nothing is
    executed and nothing is written. The host checks (tree, venv, toolchain, topology, files) run in ``run``.
    """
    repo = Path(repo or REPO)
    reg, p = load_profile(model, profile_id, kv)
    checkpoint = checkpoint or default_checkpoint(model, p)
    if checkpoint not in CHECKPOINTS or CHECKPOINTS[checkpoint][0] != model:
        raise LaunchError(f"checkpoint {checkpoint!r} does not belong to the {model} registry", 2)
    if model == "flash-next" and (p.get("checkpoint") or "flash-next") != checkpoint:
        raise LaunchError(f"profile {profile_id} serves {p.get('checkpoint') or 'flash-next'}, not {checkpoint}", 2)
    env = dict(environ)
    extra = list(extra or [])
    if model == "flash-next":
        plan = _build_flash_next(reg, p, checkpoint, ckpt, env, extra, repo)
    elif model == "27b":
        plan = _build_27b(reg, p, checkpoint, ckpt, env, extra, repo)
    elif model == "ornith":
        plan = _build_ornith(reg, p, checkpoint, ckpt, env, extra, repo)
    else:
        plan = _build_svl(reg, p, checkpoint, ckpt, env, extra, repo)
    if status in (R.LABEL_TESTED, R.LABEL_UNTESTED):
        plan.status = R.label_text(reg, p, status)
    return plan


def _abs(path: str, cwd: str) -> str:
    return path if os.path.isabs(path) else f"{cwd}/{path}"


def _default_dir(notes: list[str], var: str, static: str, old: str, advice: str) -> str:
    if not os.path.exists(static) and os.path.exists(old):
        notes.append(f"note: {var} defaults to {static}, which does not exist; using the pre-v3 directory {old} for now. The pre-v3 names are")
        notes.append(f"      read for one release (until v3.1): {advice}.")
        return old
    return static


def _img_cap(env: dict[str, str], default: int) -> str:
    v = env.get("IMG_CAP_PX") or str(default)
    if not _digits(v) or v.startswith("0"):
        raise LaunchError(f"IMG_CAP_PX must be a whole number of pixels, got '{v}'", 1)
    if len(v) > 10 or int(v) < 65536:
        raise LaunchError(f"IMG_CAP_PX must be between 65536 (the shortest_edge floor) and 9999999999 pixels, got {v}", 1)
    return v


def _max_pixels_27b(v: str) -> None:
    if v == "":
        return
    if v.startswith("0") or not _digits(v):
        raise LaunchError("MAX_PIXELS must be empty or a canonical decimal pixel count (no leading zero, digits only)", 2)
    if len(v) > 8 or int(v) < 65536 or int(v) > 33554432:
        raise LaunchError(f"MAX_PIXELS must be between 65536 and 33554432 (got {v})", 2)


def _int_knob(env: dict[str, str], name: str, assign: bool, default: str | None, code: int) -> str | None:
    """CHUNK / BLOCK_SIZE / LPT style knobs: unset -> default (None = no flag); a value must be a whole number."""
    v = _knob(env, name, default or "", assign) if default is not None else env.get(name, "")
    if v == "":
        return None
    if not _digits(v):
        raise LaunchError(f"{name} must be a whole number (got '{v}')", code)
    return v


def _guard_mode(env: dict[str, str]) -> None:
    # GUARD takes precedence even when set empty (invalid); otherwise the inherited mode, then warn.
    if "GUARD" in env:
        mode = env["GUARD"]
    else:
        mode = env.get("VLLM_E2_GUARD_MODE", "warn")
    env["VLLM_E2_GUARD_MODE"] = mode
    if mode.strip().lower() not in ("raise", "warn"):
        raise LaunchError(f"VLLM_E2_GUARD_MODE must be raise|warn, got {mode.strip().lower()!r}", 1)
    interval = env.get("VLLM_E2_GUARD_INTERVAL")
    if interval is not None:
        try:
            value = float(interval)
        except ValueError:
            value = float("nan")
        if value != value or value in (float("inf"), float("-inf")) or value <= 0:
            raise LaunchError(f"VLLM_E2_GUARD_INTERVAL must be a finite number > 0, got {interval!r}", 1)


def _build_flash_next(reg, p, checkpoint, ckpt, env, extra, repo: Path) -> Plan:
    environ_in = dict(env)
    base = dict(reg["base"])
    ck = reg.get("checkpoints", {}).get(p.get("checkpoint", "flash-next"), {})
    base.update(ck)
    lay, srv, ic = p["layout"], p["serve"], p["interconnect"]
    fp8 = srv["kv_cache_dtype"] == "fp8_e4m3"
    root = repo.as_posix()
    here = f"{root}/models/flash-next"  # the vanilla release assets (scales/, assets/, topo_select.py)
    cwd = os.getcwd()
    notes: list[str] = []
    kv_bytes = _knob(env, "KV_BYTES", str(srv["kv_cache_memory"]), assign=False)
    if not _digits(kv_bytes):
        raise LaunchError(f"KV_BYTES must be integer bytes (got {kv_bytes})", 2)
    tree = env.get("TREE") or root
    venv = env.get("VENV") or _default_dir(
        notes, "VENV", f"{root}/venv", f"{root}/venv-v2.5",
        f"rebuild with release/install-env.sh {root}/venv, which creates that environment")
    cache_root = env.get("CACHE_ROOT") or _default_dir(
        notes, "CACHE_ROOT", f"{root}/.vllm-cache", f"{root}/.vllm-cache-v2.5",
        "rename it to .vllm-cache (the cache is keyed by hash, so it carries over)")
    tree, venv = _abs(tree, cwd), _abs(venv, cwd)
    for var, val in (("TREE", tree), ("VENV", venv)):
        if var in env:
            env[var] = val
    # pure knobs first: refuse bad settings before any host access
    img_cap_px = _img_cap(env, base["img_cap_px_default"])
    draft_head = env.get("DRAFT_HEAD") or R.DRAFT_HEAD_DEFAULT
    if draft_head not in ("bf16", "int8", "int4"):
        raise LaunchError(f"DRAFT_HEAD must be bf16, int8 or int4, got '{draft_head}'", 1)
    draft_spec = "" if draft_head == "bf16" else f',"draft_lm_head_quant":"{draft_head}"'
    reduced = env.get("REDUCED_DRAFT_VOCAB") or "0"
    vocab_sha = base.get("draft_vocab_sha256")
    if vocab_sha:
        if reduced not in ("0", "1"):
            raise LaunchError(f"REDUCED_DRAFT_VOCAB must be 0 (default) or 1, got '{reduced}'", 1)
    elif reduced != "0":
        raise LaunchError("this checkpoint requires REDUCED_DRAFT_VOCAB=0 (no qualified draft-vocab table)", 1)
    warm = env.get("WARM") or str(p["warm"])
    if warm not in ("0", "1"):
        raise LaunchError(f"WARM must be 1 (default) or 0, got '{warm}'", 1)
    chunk = _int_knob(env, "CHUNK", False, str(srv["max_num_batched_tokens"]), 1)
    block = _int_knob(env, "BLOCK_SIZE", False, str(base.get("block_size", 4096)), 1)
    lpt = _int_knob(env, "LPT", False, None, 1)
    # topology mode (the selector itself runs in the host phase)
    auto_topo = env.get("AUTO_TOPO", "0")
    if auto_topo in ("0", ""):
        topo_on = False
    elif auto_topo in ("1", "strict"):
        topo_on = True
    else:
        notes.append(f"topo: note: AUTO_TOPO='{auto_topo}' is not 0, 1 or strict; the selector stays off")
        topo_on = False

    env["PYTHONPATH"] = tree
    env["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
    # platform: GPUs, interconnect, layout
    modes = ic["modes"]
    switch = len(modes) > 1
    p2p_mode: str | None = None
    if switch:
        p2p_mode = env.get("P2P") or ic["default"]
        if p2p_mode not in modes:
            raise LaunchError(f"P2P must be on or off, got '{p2p_mode}'", 1)
    elif len(p["cuda_visible_devices"].split(",")) > 1:
        p2p_mode = next(iter(modes))
    mode = modes[p2p_mode] if p2p_mode else next(iter(modes.values()))
    if topo_on and not (p["match"]["cards"] == 4 and (p2p_mode or "on") == "on"):
        # the selector is for the four-card P2P-on layout; a no-P2P or non-quad row never used it
        notes.append(f"topo: note: AUTO_TOPO applies to four-card P2P-on rows only; ignored for {p['id']} (P2P {p2p_mode or 'n/a'})")
        topo_on = False

    def platform_env(e: dict[str, str]) -> None:
        e["CUDA_VISIBLE_DEVICES"] = e.get("CUDA_VISIBLE_DEVICES") or p["cuda_visible_devices"]
        e["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        for var in mode.get("unset", []):
            e.pop(var, None)
        for k, v in mode.get("exports", {}).items():
            e[k] = str(v)

    platform_env(env)
    if lay["pp"] > 1:
        env["VLLM_PP_LAYER_PARTITION"] = ",".join(str(x) for x in lay["partition"])
    else:
        env.pop("VLLM_PP_LAYER_PARTITION", None)
    for v in [k for k in env if k.startswith("MERLIN_")] + FN_UNSET:
        env.pop(v, None)
    for v in FN_RESET_EXTRA:
        if v in env:
            notes.append(f"note: {v}={env[v]!r} dropped: the Flash-Next profile sets its own value (or the engine default) for it")
            env.pop(v)
    env["VLLM_PLE_OFFLOAD_READY_TIMEOUT"] = "1800"
    env["VLLM_PLE_OFFLOAD_HOME_DEVICE"] = env.get("PLE_HOME") or "0"
    env["VLLM_HOST_EMBED_TABLE"] = "1"
    env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"
    files: list[tuple[str, str]] = []
    # Froggeric v22.5 on every Qwen profile: it accepts non-leading system messages, so /v1/messages
    # keeps a coding agent's per-turn notes in place (the checkpoint template refuses them and vLLM merges them into the
    # leading prompt, which breaks the prefix cache). CHAT_TEMPLATE overrides; empty serves the checkpoint's own template.
    tpl = _knob(env, "CHAT_TEMPLATE", f"{root}/{FN_CHAT_TEMPLATE}", True, keep_empty=True)
    template_args = ["--chat-template", tpl] if tpl else []
    if tpl:
        files.append((tpl, f"chat template missing: {tpl} (CHAT_TEMPLATE=<path>, or CHAT_TEMPLATE= for the checkpoint's own)"))
    if fp8:
        # SCALES wins; then the checkpoint's own sidecar (a variant ships its scales with its weights); then the recipe's
        own = Path(ckpt) / CKPT_SCALES
        if not env.get("SCALES") and own.is_file():
            notes.append(f"note: FP8 KV scales from the checkpoint: {own.as_posix()}")
        scales = env.get("SCALES") or (own.as_posix() if own.is_file() else
                                       f"{root}/{ck['scales']}" if ck else f"{here}/{base['scales']}")
        env["VLLM_QSA_KV_SCALES"] = scales
        files.append((scales, f"scales sidecar missing: {scales} (without it FP8 KV runs at scale 1.0)"))
    env["VLLM_CACHE_ROOT"] = cache_root
    warm_args: list[str] = []
    if warm == "1":
        env["TRITON_CACHE_DIR"] = f"{cache_root}/triton"
        warm_args = ["--kernel-config", FN_KERNEL_CONFIG]
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    for k in ("MERLIN_FULL_K", "MERLIN_QSA_SORT", "MERLIN_TIE_DET", "MERLIN_TIE_RECENT", "MERLIN_TOPK_SORTED_EMIT"):
        env[k] = "1"
    _guard_mode(env)
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    env["VLLM_E2_PLE_FAST_IDS"] = "1"
    env["VLLM_E2_PLE_EARLY_REQUEST"] = "1"
    # One constant 256 MiB QSA prefill-logits block (granule = cap): exact-size per-chunk blocks ratchet the
    # allocator reserve towards the card wall at 262K.
    env["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = "256"
    env["VLLM_E2_QSA_LOGITS_GRANULE_MB"] = "256"
    if env.get("COUNTERS", "0") == "1":
        env["MERLIN_TOPK_EXACT_COUNTERS_LOG"] = "30"
        env["VLLM_QSA_KV_CLIP_COUNT"] = "300"
    env.pop("VLLM_DRAFT_VOCAB_FILE", None)
    env.pop("VLLM_DRAFT_VOCAB_SHA256", None)
    draft_vocab = None
    if vocab_sha and reduced == "1":
        table = env.get("DRAFT_VOCAB")
        if not table:
            raise LaunchError(
                "REDUCED_DRAFT_VOCAB=1 needs DRAFT_VOCAB=<path to a reduced draft-vocab table>: no table ships with this "
                "release (the default is REDUCED_DRAFT_VOCAB=0, the full-vocabulary draft head)", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", str(vocab_sha)):
            raise LaunchError(
                f"draft_vocab_sha256 in the registry is not a sha256 ('{vocab_sha}'): this launcher was not packaged", 1)
        files.append((table, f"draft-vocab table missing: {table} (DRAFT_VOCAB=<path>, or REDUCED_DRAFT_VOCAB=0)"))
        draft_vocab = (table, vocab_sha)
        env["VLLM_DRAFT_VOCAB_FILE"] = table
        env["VLLM_DRAFT_VOCAB_SHA256"] = vocab_sha
    names = (env.get("MODEL_NAME") or base["served_names"]).split()
    spec = f'{{"method":"mtp","num_speculative_tokens":{R.MTP_K},"draft_sample_method":"probabilistic"{draft_spec}}}'
    args = [
        "--served-model-name", *names,
        "--host", env.get("HOST") or "127.0.0.1", "--port", env.get("PORT") or str(base.get("port", 8000)),
        "--tensor-parallel-size", str(lay["tp"]), "--pipeline-parallel-size", str(lay["pp"]),
    ]
    if lay["ep"]:
        args.append("--enable-expert-parallel")
    if mode.get("disable_custom_all_reduce"):
        args.append("--disable-custom-all-reduce")
    args += [
        "--speculative-config", spec,
        "--max-model-len", str(base["max_model_len"]), "--max-num-seqs", str(srv["max_num_seqs"]),
        "--max-num-batched-tokens", chunk,
        "--gpu-memory-utilization", str(base["gpu_memory_utilization"]),
        "--kv-cache-memory-bytes", kv_bytes, "--kv-cache-dtype", srv["kv_cache_dtype"],
        "--limit-mm-per-prompt", '{"image":42,"video":0}', "--mm-processor-cache-gb", "0",
        "--mm-processor-kwargs", f'{{"size":{{"longest_edge":{img_cap_px},"shortest_edge":65536}}}}',
        "--compilation-config",
        '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[' + ",".join(map(str, srv["capture_sizes"])) + "]}",
        "--enable-prefix-caching", "--mamba-cache-mode", "align", "--enable-prompt-tokens-details",
    ]
    if lpt is not None:
        args += ["--long-prefill-token-threshold", lpt]
    args += [
        "--no-enable-flashinfer-autotune",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
        "--override-generation-config", FN_GENERATION,
        *warm_args,
        *template_args,
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": True, "reasoning_effort": base.get("reasoning_effort", R.REASONING_EFFORT_DEFAULT)}),
        "--shutdown-timeout", "60",
        "--block-size", block, "--use-replayssm",
        "--prefix-match-unit", "64",
    ]
    plan = Plan("flash-next", checkpoint, p, ckpt, env, args, extra, repo, venv, tree, cache_root,
                R.label_text(reg, p), p2p_mode, notes=notes, files=files, draft_vocab=draft_vocab)
    plan.topo = {"on": topo_on, "strict": auto_topo == "strict", "platform_env": platform_env, "here": here}
    plan.caller_env = dict(environ_in)
    return plan


def _build_27b(reg, p, checkpoint, ckpt, env, extra, repo: Path) -> Plan:
    L = reg["launch"]
    shape = L["shapes"][p["launch_shape"]]
    ck = L["checkpoints"][checkpoint]
    srv = p["serve"]
    cwd = os.getcwd()
    root = repo.as_posix()
    here = f"{root}/models/qwen38-27b"
    port = _knob(env, "PORT", str(L["port"]), True)
    host = _knob(env, "HOST", "127.0.0.1", True)
    served = _knob(env, "SERVED", ck["served"], True)
    venv = _abs(_knob(env, "VENV", f"{root}/venv", True), cwd)
    if "VENV" in env:
        env["VENV"] = venv
    gpus = _knob(env, "GPUS", env.get("CUDA_VISIBLE_DEVICES") or shape["gpus"], True)
    kv_bytes = _knob(env, "KV_BYTES", str(srv["kv_cache_memory"]), True)
    lpt = _knob(env, "LPT", str(shape["lpt"]), True)
    if not _digits(lpt):
        raise LaunchError(f"LPT must be a whole number of tokens (0 turns the cap off; got '{lpt}')", 2)
    if shape.get("spec_k_fixed"):
        spec_k = str(srv["spec_k"])
        if "SPEC_K" in env:
            env["SPEC_K"] = spec_k
    else:
        spec_k = _knob(env, "SPEC_K", str(srv["spec_k"]), True)
    max_pixels = _knob(env, "MAX_PIXELS", str(shape.get("max_pixels", L["max_pixels"])), True, keep_empty=True)
    if shape.get("vision_peak"):
        env["VLLM_MERLIN_VISION_PEAK"] = _knob(env, "VLLM_MERLIN_VISION_PEAK", "1", True, keep_empty=True)
    levers = shape.get("image_levers")
    if levers:  # one card: the image headroom settings (output-identical; each env variable overrides)
        env["VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE"] = _knob(
            env, "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(levers["flashinfer_workspace_bytes"]), True
        )
        env["MERLIN_SHARE_DRAFT_WORKSPACE"] = _knob(env, "MERLIN_SHARE_DRAFT_WORKSPACE", "1", True)
        env["VLLM_MERLIN_VISION_ROTARY_INPLACE"] = _knob(env, "VLLM_MERLIN_VISION_ROTARY_INPLACE", "1", True)
    chat_template = _knob(env, "CHAT_TEMPLATE", f"{here}/{L['chat_template']}", True, keep_empty=True)
    thinking = _knob(env, "THINKING", ck["thinking"], True)
    if ck.get("export_thinking"):  # the ThinkingCap wrappers export THINKING before forwarding
        env["THINKING"] = thinking
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
    p2p_mode = None
    if shape.get("p2p_switch"):
        # a no-P2P row (27b-tp2/tp4-3090-nop2p) defaults to P2P off; the others default to on and take P2P=on|off
        p2p_mode = _knob(env, "P2P", "off" if p["match"].get("p2p") == "off" else "on", True)
        if p2p_mode == "on":
            env["VLLM_SKIP_P2P_CHECK"] = "1"
            env["NCCL_P2P_LEVEL"] = "SYS"
        elif p2p_mode == "off":
            env["VLLM_SKIP_P2P_CHECK"] = "1"
            env["NCCL_P2P_DISABLE"] = "1"
            for v in ("NCCL_P2P_LEVEL", "NCCL_PROTO", "NCCL_ALGO"):
                env.pop(v, None)
        else:
            raise LaunchError(f"P2P must be on or off (got '{p2p_mode}')", 2)
    smallq = _knob(env, "FLASHINFER_SM8X_FP8_SMALLQ", "cta32", True, keep_empty=True)
    env["FLASHINFER_SM8X_FP8_SMALLQ"] = smallq
    if smallq not in ("", "0", "cta32"):
        raise LaunchError(f"FLASHINFER_SM8X_FP8_SMALLQ must be empty, 0 or cta32 (got '{smallq}')", 2)
    env["VLLM_NO_USAGE_STATS"] = _knob(env, "VLLM_NO_USAGE_STATS", "1", True)
    env["DO_NOT_TRACK"] = _knob(env, "DO_NOT_TRACK", "1", True)

    def commit_env() -> None:
        env["VLLM_RECOVERSSM_COMMIT_GRAPH"] = _knob(env, "VLLM_RECOVERSSM_COMMIT_GRAPH", "graph", True, keep_empty=True)
        env["VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE"] = _knob(env, "VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE", "1", True, keep_empty=True)

    if shape["rssm_commit_env"] == "always":
        commit_env()
    rssm: list[str] = []
    if shape.get("recoverssm_fixed"):
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        rssm = ["--use-replayssm", "--mamba-backend", "triton"]
    else:
        default = "1" if srv["recoverssm"] else "0"
        rs = _knob(env, "RECOVERSSM", default, True)
        if rs == "1":
            env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
            if shape["rssm_commit_env"] == "when_on":
                commit_env()
            rssm = ["--use-replayssm", "--mamba-backend", "triton"]
        elif rs != "0":
            raise LaunchError(f"RECOVERSSM must be {default} (default) or {'0' if default == '1' else '1'} (got '{rs}')", 2)
    extra_flags: list[str] = []
    if chat_template:
        extra_flags += ["--chat-template", chat_template]
    _max_pixels_27b(max_pixels)
    if max_pixels:
        extra_flags += ["--mm-processor-kwargs", f'{{"size":{{"longest_edge":{max_pixels},"shortest_edge":65536}}}}']
    if spec_k not in ("1", "3", "4"):
        raise LaunchError("SPEC_K must be 1, 3 or 4 (the measured values; 4 is the default)", 2)
    if shape.get("spec_k_fixed"):
        capture = srv["capture_sizes"]
    else:
        capture = srv["capture_sizes"] if spec_k == "4" else shape["capture_sizes_k_other"]
    lpt_flags = [] if lpt == "0" else ["--long-prefill-token-threshold", lpt]
    kvf = [] if kv_bytes == "auto" else ["--kv-cache-memory-bytes", kv_bytes]
    chunk = _int_knob(env, "CHUNK", True, str(srv["max_num_batched_tokens"]), 2)
    block = _int_knob(env, "BLOCK_SIZE", True, str(srv["block_size"]) if "block_size" in srv else None, 2)
    args = [
        "--served-model-name", served,
        "--tensor-parallel-size", str(srv["tensor_parallel"]), "--attention-backend", "FLASHINFER",
        "--max-model-len", str(L["max_model_len"]), "--gpu-memory-utilization", "0.96",
        "--kv-cache-dtype", srv.get("kv_cache_dtype", "fp8_e4m3"), *kvf, "--mamba-ssm-cache-dtype", "float16",
        "--max-num-seqs", str(srv["max_num_seqs"]), "--max-num-batched-tokens", chunk,
        "--enable-prefix-caching", "--mamba-cache-mode", "align", "--enable-prompt-tokens-details", *lpt_flags,
        "--prefix-match-unit", "16", "--enable-mamba-shared-prefix-checkpoint", *rssm,
    ]
    if block is not None:
        args += ["--block-size", block]
    if shape.get("disable_custom_all_reduce"):
        args.append("--disable-custom-all-reduce")
    args += [
        "--speculative-config", f'{{"method":"mtp","num_speculative_tokens":{spec_k},"draft_sample_method":"probabilistic"}}',
        "--compilation-config",
        '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[' + ",".join(map(str, capture))
        + '],"inductor_compile_config":' + B27_INDUCTOR + "}",
        "--override-generation-config", B27_GENERATION,
        "--default-chat-template-kwargs", thinking,
        "--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
    ]
    if shape.get("cpu_offload_visual_gb"):
        args += ["--cpu-offload-gb", str(shape["cpu_offload_visual_gb"]), "--cpu-offload-params", "visual"]
    args += [
        "--limit-mm-per-prompt", f'{{"image":{shape["images_per_prompt"]},"video":0}}', "--mm-processor-cache-gb", "0",
        *extra_flags,
        "--host", host, "--port", port, "--shutdown-timeout", "60",
    ]
    # the 27B launchers export no cache variables of their own: vLLM's default applies unless the caller sets one.
    # CACHE_ROOT=<dir> (what merlin serve's second-server hints carry) means the same as for the other families:
    # VLLM_CACHE_ROOT=<dir> and TRITON_CACHE_DIR=<dir>/triton, unless those are set explicitly.
    if env.get("CACHE_ROOT"):
        env.setdefault("VLLM_CACHE_ROOT", env["CACHE_ROOT"])
        env.setdefault("TRITON_CACHE_DIR", f"{env['VLLM_CACHE_ROOT']}/triton")
    return Plan("27b", checkpoint, p, ckpt, env, args, extra, repo, venv, None, env.get("VLLM_CACHE_ROOT") or "(vLLM default)",
                R.label_text(reg, p), p2p_mode)


def _build_svl(reg, p, checkpoint, ckpt, env, extra, repo: Path) -> Plan:
    L = reg["launch"]
    srv = p["serve"]
    cwd = os.getcwd()
    root = repo.as_posix()
    gpus = _knob(env, "GPUS", env.get("CUDA_VISIBLE_DEVICES") or p.get("cuda_visible_devices", "0,1,2,3"), True)
    port = _knob(env, "PORT", str(L["port"]), True)
    host = _knob(env, "HOST", "127.0.0.1", True)
    served = _knob(env, "SERVED", L["served"], True)
    venv = _abs(_knob(env, "VENV", f"{root}/venv", True), cwd)
    if "VENV" in env:
        env["VENV"] = venv
    cache_root = _knob(env, "CACHE_ROOT", f"{root}/.vllm-cache", True)
    kv_bytes = _knob(env, "KV_BYTES", str(srv["kv_cache_memory"]), True)
    mnbt = _knob(env, "MNBT", str(srv["max_num_batched_tokens"]), True)
    chunk = _int_knob(env, "CHUNK", True, mnbt, 2)
    seqs = _knob(env, "SEQS", str(srv["max_num_seqs"]), True)
    maxlen = _knob(env, "MAXLEN", str(L["max_model_len"]), True)
    blk = _knob(env, "BLOCK", str(srv["block_size"]), True)
    block = _int_knob(env, "BLOCK_SIZE", True, blk, 2)
    lpt = _int_knob(env, "LPT", True, None, 2)
    partition = _knob(env, "PP_PARTITION", ",".join(map(str, p["layout"]["partition"])), True, keep_empty=True)
    max_pixels = _knob(env, "MAX_PIXELS", "", True, keep_empty=True)
    if max_pixels != "":
        if max_pixels.startswith("0") or not _digits(max_pixels):
            raise LaunchError("MAX_PIXELS must be empty or a canonical decimal pixel count (digits only, no leading zero)", 2)
        if len(max_pixels) > 10 or int(max_pixels) < 8388608:
            raise LaunchError(f"MAX_PIXELS must be >= 8388608 (the 8 MP floor) (got {max_pixels})", 2)
    if not _digits(kv_bytes) or int(kv_bytes) < 1073741824:
        raise LaunchError(f"KV_BYTES must be integer bytes >= 1 GiB (never a util fraction): {kv_bytes}", 2)
    if not re.fullmatch(r"|[0-9][^,]*,[0-9].*", partition):
        raise LaunchError(f"PP_PARTITION must be empty or 'n,m' layer counts (got '{partition}')", 2)
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # a no-P2P row (svl-quad-3090-nop2p) defaults to P2P off; the others default to on and take P2P=on|off
    p2p_mode = _knob(env, "P2P", "off" if p["match"].get("p2p") == "off" else "on", True)
    p2p_args: list[str] = []
    if p2p_mode == "on":
        # Super VL keeps custom all-reduce with NCCL LL: the hybrid all-reduce changed its temperature-0 output.
        env["NCCL_P2P_LEVEL"], env["NCCL_PROTO"], env["VLLM_SKIP_P2P_CHECK"] = "SYS", "LL", "1"
        env.pop("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", None)
    elif p2p_mode == "off":
        env["VLLM_SKIP_P2P_CHECK"], env["NCCL_P2P_DISABLE"] = "1", "1"
        for v in ("NCCL_P2P_LEVEL", "NCCL_PROTO", "NCCL_ALGO", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC"):
            env.pop(v, None)
        p2p_args = ["--disable-custom-all-reduce"]
    else:
        raise LaunchError(f"P2P must be on or off (got '{p2p_mode}')", 2)
    env["VLLM_CACHE_ROOT"] = cache_root
    env["TRITON_CACHE_DIR"] = f"{cache_root}/triton"
    master, dist = distributed_ports(port)
    env["MASTER_PORT"], env["VLLM_PORT"] = str(master), str(dist)
    env["VLLM_NO_USAGE_STATS"] = _knob(env, "VLLM_NO_USAGE_STATS", "1", True)
    env["DO_NOT_TRACK"] = _knob(env, "DO_NOT_TRACK", "1", True)
    env["HF_HUB_OFFLINE"] = _knob(env, "HF_HUB_OFFLINE", "1", True)
    env["TRANSFORMERS_OFFLINE"] = _knob(env, "TRANSFORMERS_OFFLINE", "1", True)
    if partition:
        env["VLLM_PP_LAYER_PARTITION"] = partition
    if max_pixels:
        env["VLLM_MAX_IMAGE_PIXELS"] = max_pixels
    tree = env.get("TREE") or None
    if tree:
        env["PYTHONPATH"] = tree + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    lpt_flags = [] if lpt is None else ["--long-prefill-token-threshold", lpt]
    args = [
        "--served-model-name", served, "--trust-remote-code",
        "--tensor-parallel-size", str(p["layout"]["tp"]), "--pipeline-parallel-size", str(p["layout"]["pp"]),
        "--enable-expert-parallel",
        "--max-model-len", maxlen, "--max-num-seqs", seqs, "--max-num-batched-tokens", chunk, *lpt_flags,
        "--kv-cache-dtype", L["kv_cache_dtype_flag"] if srv.get("kv_cache_dtype", "fp8_e4m3") == "fp8_e4m3" else srv["kv_cache_dtype"],
        "--kv-cache-memory-bytes", kv_bytes, "--gpu-memory-utilization", "0.96",
        "--mamba-ssm-cache-dtype", "float32", "--attention-backend", "FLASHINFER",
        "--block-size", block, "--enable-prefix-caching", "--prefix-match-unit", blk, "--mamba-cache-mode", "align",
        "--use-replayssm", "--mamba-backend", "triton",
        "--speculative-config", f'{{"method":"mtp","num_speculative_tokens":{srv["spec_k"]}}}',
        "--reasoning-parser", "nemotron_v3", "--enable-prompt-tokens-details",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",  # the checkpoint card's vLLM recipe
        "--limit-mm-per-prompt", '{"image":1,"video":0}', "--mm-processor-cache-gb", "0",
        *p2p_args, "--host", host, "--port", port,
    ]
    return Plan("svl", checkpoint, p, ckpt, env, args, extra, repo, venv, tree, cache_root, R.label_text(reg, p), p2p_mode)


def _build_ornith(reg, p, checkpoint, ckpt, env, extra, repo: Path) -> Plan:
    """Ornith 1.5 (Qwen3.5-122B-A10B MoE fine-tune): TP2 x PP2 + EP, FP8 KV, MTP with RecoverSSM on Model Runner V2, the
    checkpoint's own chat template and sampling. Needs the engine's Qwen3.5-MoE RecoverSSM / host-embed / uniform-verify
    support in this engine."""
    L = reg["launch"]
    srv = p["serve"]
    cwd = os.getcwd()
    root = repo.as_posix()
    gpus = _knob(env, "GPUS", env.get("CUDA_VISIBLE_DEVICES") or p.get("cuda_visible_devices", "0,1,2,3"), True)
    port = _knob(env, "PORT", str(L["port"]), True)
    host = _knob(env, "HOST", "127.0.0.1", True)
    served = _knob(env, "SERVED", L["served"], True)
    venv = _abs(_knob(env, "VENV", f"{root}/venv", True), cwd)
    if "VENV" in env:
        env["VENV"] = venv
    cache_root = _knob(env, "CACHE_ROOT", f"{root}/.vllm-cache", True)
    kv_bytes = _knob(env, "KV_BYTES", str(srv["kv_cache_memory"]), True)
    chunk = _int_knob(env, "CHUNK", True, str(srv["max_num_batched_tokens"]), 2)
    seqs = _knob(env, "SEQS", str(srv["max_num_seqs"]), True)
    maxlen = _knob(env, "MAXLEN", str(L["max_model_len"]), True)
    block = _int_knob(env, "BLOCK_SIZE", True, str(srv["block_size"]), 2)
    lpt = _int_knob(env, "LPT", True, None, 2)
    partition = _knob(env, "PP_PARTITION", ",".join(map(str, p["layout"]["partition"])), True, keep_empty=True)
    max_pixels = _knob(env, "MAX_PIXELS", str(srv["image_max_pixels"]), True)
    uv = _knob(env, "UV", "1" if srv.get("uniform_verify") else "0", True)
    if not _digits(max_pixels) or max_pixels.startswith("0") or len(max_pixels) > 10 or int(max_pixels) < 8388608:
        raise LaunchError(f"MAX_PIXELS must be a pixel count >= 8388608 (the 8 MP floor) (got {max_pixels})", 2)
    if not _digits(kv_bytes) or int(kv_bytes) < 1073741824:
        raise LaunchError(f"KV_BYTES must be integer bytes >= 1 GiB (never a util fraction): {kv_bytes}", 2)
    if block is None or int(block) < 4608:
        raise LaunchError(f"BLOCK_SIZE must be >= 4608 (the engine needs >= 4512 at fp32 SSM state) (got {block})", 2)
    if partition and not (re.fullmatch(r"[1-9][0-9]*,[1-9][0-9]*", partition) and sum(map(int, partition.split(","))) == 48):
        raise LaunchError(f"PP_PARTITION must be empty or two layer counts that sum to 48, e.g. 25,23 (got '{partition}')", 2)
    if uv not in ("0", "1"):
        raise LaunchError(f"UV must be 1 or 0 (got '{uv}')", 2)
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    row_p2p = "off" if p["match"].get("p2p") == "off" else "on"
    p2p_mode = _knob(env, "P2P", row_p2p, True)
    if p2p_mode in ("on", "off") and p2p_mode != row_p2p:
        other = "ornith-quad-3090" if p2p_mode == "on" else "ornith-quad-3090-nop2p"
        raise LaunchError(f"P2P={p2p_mode} does not match {p['id']} (qualified with P2P {row_p2p}; its KV pin was measured on that "
                          f"interconnect): use --profile {other}", 2)
    if p2p_mode == "on":
        env["VLLM_SKIP_P2P_CHECK"], env["NCCL_P2P_LEVEL"], env["NCCL_PROTO"] = "1", "SYS", "Simple"
        env["VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC"] = "1"
        env.pop("NCCL_P2P_DISABLE", None)
    elif p2p_mode == "off":
        env["NCCL_P2P_DISABLE"] = "1"
        for v in ("VLLM_SKIP_P2P_CHECK", "NCCL_P2P_LEVEL", "NCCL_PROTO", "NCCL_ALGO", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC"):
            env.pop(v, None)
    else:
        raise LaunchError(f"P2P must be on or off (got '{p2p_mode}')", 2)
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    env["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
    env["MERLIN_MOE_UNIFORM_VERIFY"] = uv
    env["VLLM_CACHE_ROOT"] = cache_root
    env["TRITON_CACHE_DIR"] = f"{cache_root}/triton"
    master, dist = distributed_ports(port)
    env["MASTER_PORT"], env["VLLM_PORT"] = str(master), str(dist)
    env["VLLM_NO_USAGE_STATS"] = _knob(env, "VLLM_NO_USAGE_STATS", "1", True)
    env["DO_NOT_TRACK"] = _knob(env, "DO_NOT_TRACK", "1", True)
    if partition:
        env["VLLM_PP_LAYER_PARTITION"] = partition
    else:
        env.pop("VLLM_PP_LAYER_PARTITION", None)  # PP_PARTITION="" = the engine's even split, never an inherited value
    tree = env.get("TREE") or None
    if tree:  # launcher convention (as for Super VL): the tree goes first, an inherited PYTHONPATH stays behind it
        env["PYTHONPATH"] = tree + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    lpt_flags = [] if lpt is None else ["--long-prefill-token-threshold", lpt]
    args = [
        "--served-model-name", served,
        "--tensor-parallel-size", str(p["layout"]["tp"]), "--pipeline-parallel-size", str(p["layout"]["pp"]),
        "--disable-custom-all-reduce",
        "--max-model-len", maxlen, "--max-num-seqs", seqs, "--max-num-batched-tokens", chunk, *lpt_flags,
        "--gpu-memory-utilization", "0.96", "--kv-cache-memory-bytes", kv_bytes,
        "--kv-cache-dtype", srv.get("kv_cache_dtype", "fp8_e4m3"), "--block-size", block,
        "--limit-mm-per-prompt", f'{{"image":{L["images_per_prompt"]},"video":0}}', "--mm-processor-cache-gb", "0",
        "--mm-processor-kwargs", f'{{"size":{{"longest_edge":{max_pixels},"shortest_edge":65536}}}}',
        "--enable-prefix-caching", "--mamba-cache-mode", "align", "--prefix-match-unit", str(srv["prefix_match_unit"]),
        "--enable-prompt-tokens-details", "--no-enable-flashinfer-autotune",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
        "--override-generation-config", ORN_GENERATION, "--kernel-config", ORN_KERNEL_CONFIG,
        "--mamba-backend", "triton", "--shutdown-timeout", "60",
    ]
    if p["layout"].get("ep"):
        args.append("--enable-expert-parallel")
    args += [
        "--speculative-config", f'{{"method":"mtp","num_speculative_tokens":{srv["spec_k"]},"draft_sample_method":"probabilistic"}}',
        "--use-replayssm",
        "--compilation-config", '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":['
        + ",".join(map(str, L["capture_sizes"])) + "]}",
        "--host", host, "--port", port,
    ]
    return Plan("ornith", checkpoint, p, ckpt, env, args, extra, repo, venv, tree, cache_root, R.label_text(reg, p), p2p_mode)


# --------------------------------------------------------------------------------------------
# Host phase: checks, toolchain, topology, then exec
# --------------------------------------------------------------------------------------------
def _say(msg: str) -> None:
    print(msg, flush=True)


def _venv_python(venv: str, code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([f"{venv}/bin/python", "-c", code, *args], capture_output=True, text=True)


def _verify_products(tree: str) -> None:
    sums = f"{tree}/.build-products.sha256"
    if not os.path.isfile(f"{tree}/.build-products") or not (os.path.isfile(sums) and os.path.getsize(sums) > 0):
        raise LaunchError(f"TREE={tree} has no verified build products (release/install-env.sh)", 1)
    ok = True
    with open(f"{tree}/.build-products.sha256", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    for ln in lines:  # sha256sum -c --strict: every line must parse and match
        m = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", ln)
        if not m:
            ok = False
            break
        path = os.path.join(tree, m.group(2))
        try:
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            ok = False
            break
        if digest.lower() != m.group(1).lower():
            ok = False
            break
    if not ok:
        raise LaunchError(f"TREE={tree}: build products changed since release/install-env.sh verified them", 1)


def _check_shadow(tree: str) -> None:
    pwd_vllm = os.path.join(os.getcwd(), "vllm")
    if not os.path.isdir(pwd_vllm):
        return
    real = os.path.realpath(pwd_vllm)
    if real == os.path.realpath(os.path.join(tree, "vllm")):
        return
    if real == os.path.realpath(tree) and not os.path.exists(os.path.join(tree, "__init__.py")):
        return
    raise LaunchError(
        f"refusing to start: {pwd_vllm} would shadow {tree}/vllm.\n"
        "  The registry-inspection child puts the current directory before PYTHONPATH on sys.path, so that copy\n"
        "  would be imported instead of the built tree. Start the server from a directory without a vllm/ beside it,\n"
        "  or pass TREE=<that checkout> if it really is the built tree.", 1)


def nvcc_release(cuda_home: str) -> str | None:
    try:
        out = subprocess.run([f"{cuda_home}/bin/nvcc", "--version"], capture_output=True, text=True).stdout
    except OSError:
        return None
    m = re.search(r"release (\d+\.\d+),", out)
    return m.group(1) if m else None


def check_inherited_cuda_home(cuda_home: str) -> None:
    """A CUDA_HOME from the caller's shell replaces the venv's locked CUDA 13.0 toolkit; JIT kernels built with
    another major version do not load against torch's CUDA 13 build."""
    rel = nvcc_release(cuda_home)
    if rel is None:
        _say(f"merlin: warning: CUDA_HOME={cuda_home} (from the environment) has no working nvcc; "
             "unset CUDA_HOME to use the venv's locked toolkit")
    elif rel.split(".")[0] != "13":
        raise LaunchError(
            f"CUDA_HOME={cuda_home} (from the environment) is CUDA {rel}; this engine needs a CUDA 13 toolkit. "
            "Unset CUDA_HOME to use the venv's locked toolkit.", 1)


def _toolchain(plan: Plan, full: bool) -> None:
    """CUDA_HOME default = the venv's locked CUDA 13.0 toolkit (release/install-0.32.md); ``full`` adds the
    first-serve checks of the Flash-Next recipe (cc/c++/ninja, Python.h, nvcc, libcudart, nvcc-vs-driver)."""
    env, venv = plan.env, plan.venv
    if not env.get("CUDA_HOME"):
        cp = _venv_python(venv, 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
        if cp.returncode != 0:
            raise LaunchError(f"VENV={venv}: cannot read the venv's purelib for the CUDA_HOME default ({cp.stderr.strip()})", 1)
        env["CUDA_HOME"] = cp.stdout.strip().splitlines()[-1] + "/nvidia/cu13"
    else:
        check_inherited_cuda_home(env["CUDA_HOME"])
    cuda_home = env["CUDA_HOME"]
    env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")
    if not full:
        return
    import shutil

    for t in ("cc", "c++", "ninja"):
        if shutil.which(t, path=env["PATH"]) is None:
            raise LaunchError(f"first serve needs {t} (Debian/Ubuntu: apt install gcc g++ ninja-build)", 1)
    cp = _venv_python(
        venv,
        'import os,sys,sysconfig; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"],"Python.h")) else 1)',
    )
    if cp.returncode != 0:
        raise LaunchError(
            "first serve needs the Python 3.13 headers: Triton compiles a C module against Python.h (Debian 13, or Ubuntu "
            "with the deadsnakes PPA: apt install python3.13-dev; uv and pyenv Pythons ship Python.h; the venv must come "
            "from that interpreter)", 1)
    nvcc = f"{cuda_home}/bin/nvcc"
    if not os.access(nvcc, os.X_OK):
        raise LaunchError(f"no nvcc at CUDA_HOME={cuda_home} (release/install-env.sh installs the CUDA 13.0 compiler wheels)", 1)
    if not os.path.exists(f"{cuda_home}/lib64/libcudart.so"):
        raise LaunchError(f"{cuda_home}/lib64/libcudart.so missing (release/install-env.sh creates the lib64 links)", 1)
    nv_rel = nvcc_release(cuda_home)
    if not nv_rel:
        raise LaunchError(f"could not read a CUDA release from `{nvcc} --version` (see its error above)", 1)
    min_drv = "13.0" if nv_rel.split(".")[0] == "13" else nv_rel
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, env=env).stdout
    except OSError:
        smi = ""
    dm = re.search(r"CUDA (?:UMD )?Version: *(\d+\.\d+)", smi)
    if not dm:
        _say("note: could not read the driver's CUDA version from nvidia-smi; the nvcc-versus-driver check is skipped")
        return
    drv = dm.group(1)
    if tuple(map(int, drv.split("."))) < tuple(map(int, min_drv.split("."))):
        raise LaunchError(
            f"nvcc at {cuda_home} is CUDA {nv_rel} but the driver supports CUDA {drv}; this toolkit requires CUDA {min_drv} or later.\n"
            "  Upgrade the driver or set CUDA_HOME to a compatible toolkit (see release/install-0.32.md).", 1)


_TOPO_VAL = r"([A-Za-z0-9_@%+=:,./-]+|'[^']*'(\"'\"'[^']*')*)"
_TOPO_LINES = [
    r"# topo_select v1 status=(ok|legacy) reason=[A-Za-z0-9_.-]+",
    rf"export CUDA_VISIBLE_DEVICES={_TOPO_VAL}",
    rf"export CUDA_DEVICE_ORDER={_TOPO_VAL}",
    rf"export VLLM_SKIP_P2P_CHECK={_TOPO_VAL}",
    rf"(export NCCL_P2P_LEVEL={_TOPO_VAL}|unset NCCL_P2P_LEVEL)",
    rf"(export NCCL_PROTO={_TOPO_VAL}|unset NCCL_PROTO)",
    r"export TOPO_LAUNCH_PREFIX=('numactl --physcpubind=[0-9,-]+ --preferred=[0-9]+ --'|'')",
    rf"export TOPO_DECISION_JSON={_TOPO_VAL}",
]


def _topo_env_file(path: str) -> list[str] | None:
    """The selector's env file must be exactly its 8-line format; anything else means 'do not source it'."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
    except OSError:
        return None
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != len(_TOPO_LINES):
        return None
    for ln, pat in zip(lines, _TOPO_LINES):
        if not re.fullmatch(pat, ln):
            return None
    return lines


def _apply_topo_lines(env: dict[str, str], lines: list[str]) -> str:
    prefix = ""
    for ln in lines[1:]:
        if ln.startswith("unset "):
            env.pop(ln[6:], None)
            continue
        name, val = ln[7:].split("=", 1)
        val = shlex.split(val)[0] if val.startswith("'") else val
        env[name] = val
        if name == "TOPO_LAUNCH_PREFIX":
            prefix = val
    return prefix


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError, AttributeError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


def _topology(plan: Plan) -> None:
    """The AUTO_TOPO selector integration (models/flash-next/docs/topology-selector.md)."""
    t = plan.topo
    env = plan.env
    plan.prefix = []
    if not t["on"]:
        return
    snapshot: dict[str, str | None] = {}
    for v in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_SKIP_P2P_CHECK", "NCCL_P2P_LEVEL", "NCCL_PROTO", "PLE_HOME"):
        caller = plan.caller_env.get(v)
        snapshot[v] = caller
        if caller is not None:
            env[f"TOPO_CALLER_SET_{v}"], env[f"TOPO_CALLER_{v}"] = "1", caller
        else:
            env[f"TOPO_CALLER_SET_{v}"] = "0"
    env["TOPO_LAUNCH_PREFIX"] = ""
    env["TOPO_DECISION_JSON"] = ""
    timeout = 20
    raw = env.get("TOPO_TIMEOUT", "")
    if raw != "":
        if re.fullmatch(r"[1-9][0-9]{0,2}", raw):
            if int(raw) <= 600:
                timeout = int(raw)
            else:
                _say(f"topo: note: TOPO_TIMEOUT={raw} is above 600 s; using 20")
        else:
            _say(f"topo: note: TOPO_TIMEOUT='{raw}' is not a whole number of seconds from 1 to 600; using 20")

    def legacy(fallback: bool) -> None:
        if fallback:
            cvd = snapshot["CUDA_VISIBLE_DEVICES"]
            if cvd is None or cvd.strip() == "":
                env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                env["CUDA_VISIBLE_DEVICES"] = cvd
            env["TOPO_LAUNCH_PREFIX"] = ""
            env["TOPO_DECISION_JSON"] = ""
            plan.prefix = []
        t["platform_env"](env)

    def unavailable(why: str) -> None:
        if t["strict"]:
            _say(f"topo: {why}; AUTO_TOPO=strict refuses to start without a selector decision (AUTO_TOPO=1 falls back to the legacy settings)")
            raise LaunchError("", 1)
        _say(f"topo: {why}, using legacy settings")
        legacy(True)

    # the platform block ran in build() with the caller's CUDA_VISIBLE_DEVICES; the selector decides from the snapshot
    try:
        # like mktemp(1): honour TMPDIR as given (a missing TMPDIR is a failure, not a fallback)
        fd, tmp = tempfile.mkstemp(dir=os.environ.get("TMPDIR") or None)
        os.close(fd)
    except OSError:
        unavailable("could not create a temporary file for the selector")
        return
    try:
        sel_env = dict(env)
        sel_env["TOPO_TIMEOUT"] = str(timeout)
        for v, val in snapshot.items():  # the selector sees the caller's values, not our defaults
            if val is None:
                sel_env.pop(v, None)
            else:
                sel_env[v] = val
        cmd = [f"{plan.venv}/bin/python", f"{t['here']}/topo_select.py", "--env-out", tmp, "--json-dir", f"{plan.cache_root}/topology"]
        rc = 0
        try:
            # its own process group, so the watchdog ends the selector AND anything it spawned (as `timeout` does)
            proc = subprocess.Popen(cmd, env=sel_env, start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout + 5)
            except subprocess.TimeoutExpired:
                _signal_group(proc, 15)
                try:
                    proc.wait(timeout=5)
                    rc = 124
                except subprocess.TimeoutExpired:
                    _signal_group(proc, 9)
                    proc.wait()
                    rc = 137
        except OSError:
            rc = 127
        if rc == 3:
            raise LaunchError("", 1)  # strict mode refused to guess; the selector already put its reason on stderr
        if rc != 0:
            unavailable(f"selector unavailable (rc={rc})")
            return
        lines = _topo_env_file(tmp)
        if lines is None:
            unavailable("selector unavailable (rc=0, unusable env file)")
            return
        plan.prefix = shlex.split(_apply_topo_lines(env, lines))
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _cgroup_bytes(path: str) -> int | None:
    """A cgroup memory file as an integer; None when missing, ``max``, or a limit so large it means unlimited."""
    try:
        with open(path, encoding="utf-8") as f:
            v = f.read().strip()
    except OSError:
        return None
    if not v.isdigit() or int(v) >= 1 << 60:
        return None
    return int(v)


def _cgroup_stat_bytes(path: str, key: str) -> int | None:
    """One ``key value`` line of a cgroup memory.stat; None when the file or the key is missing."""
    try:
        with open(path, encoding="utf-8") as f:
            m = re.search(rf"(?m)^{re.escape(key)}\s+(\d+)\s*$", f.read())
    except OSError:
        return None
    return int(m.group(1)) if m else None


def _host_memory_warnings(workers: int = 4) -> None:
    # Warn only: the recipe shims keep the hand-written launchers' behaviour.
    # merlin serve refuses on the same reading (serve_cli.py). ``workers`` = TP x PP of the layout: the resident floor
    # is the measured four-worker figure plus ~2 GiB per worker above four (preflight.ram_floor_gib).
    floor_kb = FN_RAM_FLOOR_KB + max(0, workers - 4) * 2097152
    floor_gib = floor_kb // 1048576
    mem_kb: int | None = None
    avail_kb: int | None = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("MemTotal:"):
                    mem_kb = int(ln.split()[1])
                elif ln.startswith("MemAvailable:"):
                    avail_kb = int(ln.split()[1])
    except (OSError, ValueError):
        pass
    # MemAvailable capped by what the cgroup has left: max - (current - inactive_file), the reclaimable page cache
    # taken out of the usage; the same rule as preflight.mem_available_mib (kept inline: the launcher runs from
    # trees without preflight.py)
    for lim_f, use_f, stat_f, key in (
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.stat", "inactive_file"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes",
             "/sys/fs/cgroup/memory/memory.stat", "total_inactive_file")):
        lim, use = _cgroup_bytes(lim_f), _cgroup_bytes(use_f)
        if lim is not None and use is not None:
            use = max(0, use - (_cgroup_stat_bytes(stat_f, key) or 0))
            left_kb = max(0, lim - use) // 1024
            if avail_kb is None or left_kb < avail_kb:
                avail_kb = left_kb
    for fpath in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(fpath, encoding="utf-8") as f:
                lim = f.read().strip()
        except OSError:
            continue
        if not lim.isdigit():
            continue
        lim_kb = int(lim) // 1024
        if mem_kb is None or lim_kb < mem_kb:
            mem_kb = lim_kb
    if mem_kb is not None and (mem_kb < FN_RAM_QUALIFIED_KB or mem_kb < floor_kb):
        _say(f"WARNING: host RAM (MemTotal or the cgroup limit) is {mem_kb // 1048576} GiB; the qualified configuration is 96 GB")
        _say(f"         (measured resident floor ~{floor_gib} GiB for {workers} workers: ~48 GiB n-gram table + ~4.2 GiB pinned "
             "embeddings + workers).")
        if mem_kb < floor_kb:
            _say(f"         Below the ~{floor_gib} GiB resident floor the model load is expected to OOM.")
    if avail_kb is not None and avail_kb < floor_kb and (mem_kb is None or mem_kb >= floor_kb):
        _say(f"WARNING: host memory available now is {avail_kb // 1048576} GiB (MemAvailable, capped by what the cgroup has left), "
             f"below the ~{floor_gib} GiB resident floor for {workers} workers")
        _say("         (another Merlin server or a large page-cache user?); the model load is expected to OOM.")
    try:
        st = os.statvfs("/dev/shm")
        shm_kb = st.f_blocks * st.f_frsize // 1024
    except (OSError, AttributeError):
        shm_kb = None
    if shm_kb is not None and shm_kb < FN_SHM_MIN_KB:
        _say(f"WARNING: /dev/shm is {shm_kb // 1024} MB; the PLE transport needs >= 1 GB (docker run --shm-size=8g, compose shm_size: 8g)")


def banner(plan: Plan) -> None:
    """The resolved values a launch trusts from the environment, printed before anything runs."""
    env = plan.env
    host, port = effective_host_port(plan.args, plan.extra)
    print(
        f"merlin launch: {plan.checkpoint} profile {plan.profile.get('variant_of') or plan.profile['id']} [{plan.status}] "
        f"cards {env.get('CUDA_VISIBLE_DEVICES', '<unset>')} -> http://{host}:{port}; VENV={plan.venv} "
        f"TREE={plan.tree or '(installed engine)'} CACHE_ROOT={plan.cache_root}"
        + (f" SCALES={env['VLLM_QSA_KV_SCALES']}" if env.get("VLLM_QSA_KV_SCALES") else ""),
        file=sys.stderr,
    )


def check_remote_code(ckpt: str, pinned: Path) -> None:
    """Super VL loads its config and processor with --trust-remote-code, which runs the checkpoint's Python files:
    every one of them must match the reviewed hash list."""
    want: dict[str, str] = {}
    with open(pinned, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                digest, name = ln.split(None, 1)
                want[name.strip()] = digest
    for py in sorted(Path(ckpt).glob("*.py")):
        if want.get(py.name) != _sha256_file(py):
            raise LaunchError(
                f"{py} does not match the reviewed remote code ({pinned.name}); this checkpoint runs it with "
                "--trust-remote-code, so re-download the checkpoint before serving", 1)


def host_phase(plan: Plan) -> None:
    """Every check that touches the host; raises LaunchError instead of starting."""
    for n in plan.notes:
        print(n, file=sys.stderr)
    banner(plan)
    warn_retired(plan.caller_env)
    env, venv = plan.env, plan.venv
    fam = plan.family
    if fam == "flash-next":
        tree = plan.tree or plan.repo.as_posix()
        if not os.path.isfile(f"{tree}/vllm/__init__.py"):
            msg = f"TREE={tree} is not a built tree (release/install-env.sh)"
            if os.path.exists(f"{plan.repo.as_posix()}/vllm-v2.5"):
                msg += f"\n       ({plan.repo.as_posix()}/vllm-v2.5 is a pre-v3 build of the v2.5.1 source and is not used; see release/install-env.sh)"
            raise LaunchError(msg, 1)
        _verify_products(tree)
        if not os.access(f"{venv}/bin/python", os.X_OK):
            raise LaunchError(f"VENV={venv} has no bin/python (release/install-env.sh creates it)", 1)
        marker = plan.profile.get("checkpoint") and R.load_registry("flash-next")["checkpoints"][plan.profile["checkpoint"]].get("checkpoint_marker")
        if marker and not os.path.isfile(os.path.join(plan.ckpt, marker)):
            raise LaunchError(f"wrong checkpoint: missing {marker} in {plan.ckpt}", 1)
        # the key must sit INSIDE text_config (the engine reads its text config; a top-level copy does nothing)
        try:
            with open(os.path.join(plan.ckpt, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            ok = cfg.get("text_config", {}).get("ple_embedding_dtype") == "float8_e4m3fn"
        except (OSError, ValueError):
            ok = False
        if not ok:
            raise LaunchError(
                f"config.json lacks text_config.ple_embedding_dtype = float8_e4m3fn: run models/flash-next/make-e1-config.py {plan.ckpt}", 1)
        _check_shadow(tree)
        _toolchain(plan, full=True)
        _topology(plan)
    else:
        if not os.access(f"{venv}/bin/python", os.X_OK):
            raise LaunchError(f"VENV={venv} has no bin/python (release/install-env.sh creates it)", 1)
        if fam == "svl":
            check_remote_code(plan.ckpt, plan.repo / CHECKPOINTS["nemotron-super-vl"][1] / "remote-code.sha256")
        _toolchain(plan, full=False)
    for path, msg in plan.files:
        if not os.path.isfile(path):
            raise LaunchError(msg, 1)
    if plan.draft_vocab:
        table, want = plan.draft_vocab
        with open(table, "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != want:
            raise LaunchError(f"draft-vocab table {table} has sha256 {got}, not the release's {want}", 1)
    if fam == "flash-next":
        lay = plan.profile.get("layout") or {}
        _host_memory_warnings(int(lay.get("tp", 1)) * int(lay.get("pp", 1)) if lay else 4)
    if fam in ("svl", "ornith"):
        os.makedirs(env["VLLM_CACHE_ROOT"], exist_ok=True)
        os.makedirs(env["TRITON_CACHE_DIR"], exist_ok=True)


def effective_host_port(args: list[str], extra: list[str]) -> tuple[str, int]:
    """The host and port vllm will bind: the profile's flags, overridden by extra args (they come last and win)."""
    host, port = "127.0.0.1", 8000
    for seq in (args, extra):
        i = 0
        while i < len(seq):
            a = seq[i]
            if a in ("--host", "--port") and i + 1 < len(seq):
                if a == "--host":
                    host = seq[i + 1]
                else:
                    port = int(seq[i + 1]) if seq[i + 1].isdigit() else port
                i += 2
                continue
            if a.startswith("--host="):
                host = a[7:]
            elif a.startswith("--port=") and a[7:].isdigit():
                port = int(a[7:])
            i += 1
    return host, port


def port_free(host: str, port: int) -> bool:
    """Whether ``host``:``port`` can be bound now, over IPv4 or IPv6 as the host resolves. False when the port is taken;
    LaunchError when the host is not an address this machine can bind."""
    import errno
    import socket

    h = host.strip("[]")
    try:
        if h in ("", "0.0.0.0"):
            family, addr = socket.AF_INET, ("", port)
        else:
            family, _, _, _, addr = socket.getaddrinfo(h, port, type=socket.SOCK_STREAM)[0]
    except (socket.gaierror, UnicodeError) as e:
        raise LaunchError(f"--host {host}: not a resolvable address ({e})", 2) from None
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(addr)
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                return False
            raise LaunchError(f"--host {host} --port {port}: cannot bind ({e.strerror or e})", 2) from None
    return True


def print_hook(plan: Plan) -> None:
    env = plan.env
    _say("SERVE_PRINT_ENV_AND_EXIT:")
    _say(f"PROFILE={plan.profile.get('variant_of') or plan.profile['id']}"
         + (f"/kv-{R.KV_TAG[plan.profile['serve']['kv_cache_dtype']]}" if plan.profile.get("variant_of") else ""))
    _say(f"STATUS={plan.status}")
    if plan.p2p_mode:
        _say(f"P2P={plan.p2p_mode}")
    for var in HOOK_VARS:
        _say(f"{var}={env.get(var, '<unset>')}")
    _say(f"TOPO_LAUNCH_PREFIX={env.get('TOPO_LAUNCH_PREFIX', '')}")
    _say(f"TOPO_DECISION_JSON={env.get('TOPO_DECISION_JSON', '')}")
    _say("EXEC_ARGV_PREFIX:" + "".join(f" [{w}]" for w in plan.prefix))
    _say("ARGV:")
    for a in plan.args:
        _say(a)


def unauthenticated_exposure(host: str, env: dict[str, str], args: list[str], extra: list[str]) -> str:
    """A warning when the API listens beyond loopback with no key; empty otherwise. Inside a container the
    in-container bind is expected; the published port decides the exposure."""
    if host in ("127.0.0.1", "::1", "localhost") or os.path.exists("/.dockerenv"):
        return ""
    if env.get("VLLM_API_KEY") or "--api-key" in args or any(a.startswith("--api-key") for a in extra):
        return ""
    return (f"merlin: warning: HOST={host} serves every client that can reach this host without authentication; "
            "set VLLM_API_KEY (bearer key) or HOST=127.0.0.1")


def run(plan: Plan) -> int:
    host_phase(plan)
    if os.environ.get("SERVE_PRINT_ENV_AND_EXIT", "0") == "1":
        print_hook(plan)
        return 0
    host, port = effective_host_port(plan.args, plan.extra)
    unauth = unauthenticated_exposure(host, plan.env, plan.args, plan.extra)
    if unauth:
        _say(unauth)
    if os.environ.get("MERLIN_SKIP_PORT_CHECK") != "1" and not port_free(host, port):
        raise LaunchError(
            f"port {port} on {host} is already in use (another server? the Flash-Next / Super VL / Ornith default is 8000, the 27B "
            f"default 8100); set PORT={port + 1} or stop the other server", 2)
    argv = plan.argv()
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(argv[0], argv, plan.env)
    return 0  # pragma: no cover


# --------------------------------------------------------------------------------------------
# Checkpoint identification (merlin serve ./ckpt with no name)
# --------------------------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_list(repo: Path, name: str) -> dict[str, str]:
    path = repo / CHECKPOINTS[name][1] / "checkpoint.sha256"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"([0-9a-fA-F]{64})\s+\*?(.+)$", ln.strip())
        if m:
            out[m.group(2).strip()] = m.group(1).lower()
    return out


def identify_checkpoint(ckpt: str, repo: Path | None = None) -> tuple[str, list[str]]:
    """Name the checkpoint in ``ckpt`` from config.json and the recipes' checkpoint.sha256 lists.

    Returns (name, notes). Small listed files are hashed to tell apart recipes whose config.json agrees; no safetensors
    shard is ever hashed. Raises LaunchError when the directory is not a Merlin checkpoint or stays ambiguous.
    """
    repo = Path(repo or REPO)
    d = Path(ckpt)
    notes: list[str] = []
    try:
        with open(d / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        raise LaunchError(f"{ckpt}: no readable config.json ({e}); pass the checkpoint name (merlin list)", 2)
    mt = str(cfg.get("model_type", "")) + " " + " ".join(cfg.get("architectures") or [])
    if "nemotron" in mt.lower():
        return "nemotron-super-vl", notes
    if "qwen3_5_moe" in mt.lower():
        # one Qwen3.5-MoE recipe (Ornith 1.5): name it only when config.json is the release's, byte for byte
        want = _sha_list(repo, "ornith1.5").get("config.json")
        if want and _sha256_file(d / "config.json") == want:
            return "ornith1.5", notes
        raise LaunchError(f"{ckpt}: a Qwen3.5-MoE checkpoint that is not the release's Ornith 1.5 pack (its config.json differs from "
                          "models/ornith1.5/checkpoint.sha256); pass the checkpoint name to serve it with that recipe (merlin list)", 2)
    cands = [n for n, (model, _) in CHECKPOINTS.items() if model == ("flash-next" if "qwen4_exp" in mt or "exp" in mt.lower() else "27b")]
    if "27b" in cands and "qwen" not in mt.lower():
        raise LaunchError(f"{ckpt}: model_type {cfg.get('model_type')!r} is not a Merlin checkpoint; pass the name (merlin list)", 2)
    if (d / "EDITED_TENSORS.txt").is_file() and "swift1.5-huihui" in cands:
        return "swift1.5-huihui", notes
    cands = [c for c in cands if c != "swift1.5-huihui"]
    # score by the listed small files that exist, then by their hashes
    lists = {c: _sha_list(repo, c) for c in cands}
    small = {c: {f: h for f, h in lst.items() if not f.endswith(".safetensors")} for c, lst in lists.items()}
    present = {c: {f for f in files if (d / f).is_file()} for c, files in small.items()}
    best = max((len(present[c]) / max(len(small[c]), 1), c) for c in cands)[0] if cands else 0
    top = [c for c in cands if small[c] and len(present[c]) / len(small[c]) == best and best > 0]
    if len(top) == 1:
        return top[0], notes
    if not top:
        raise LaunchError(f"{ckpt}: none of the recipes' checkpoint.sha256 lists matches this directory; pass the name (merlin list)", 2)
    hashed: dict[str, str] = {}
    for c in top:
        for fname in sorted(present[c]):
            if fname not in hashed:
                hashed[fname] = _sha256_file(d / fname)
    match = [c for c in top if all(small[c][f] == hashed[f] for f in present[c])]
    if len(match) == 1:
        return match[0], notes
    hint = "thinkingcap" in d.resolve().name.lower()
    if {"27b", "27b-thinkingcap"} <= set(top):
        pick = "27b-thinkingcap" if hint else "27b"
        notes.append(f"note: 27b and 27b-thinkingcap share every small file; taking {pick} from the directory name "
                     "(pass 27b or 27b-thinkingcap to be explicit)")
        return pick, notes
    raise LaunchError(f"{ckpt}: ambiguous between {', '.join(sorted(match or top))}; pass the checkpoint name", 2)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="launch.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=("flash-next", "27b", "svl", "ornith"), help="registry (default: the profile's)")
    ap.add_argument("--profile", required=True, help="profile id (tools/merlin/profiles/*.json)")
    ap.add_argument("--kv", choices=("fp8", "bf16"), help="KV variant of the profile")
    ap.add_argument("--checkpoint", choices=list(CHECKPOINTS), help="checkpoint name (default: the profile's)")
    ap.add_argument("--status", choices=(R.LABEL_TESTED, R.LABEL_UNTESTED),
                    help="the label merlin serve evaluated on these cards (default: the registry row's)")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="[--] <checkpoint-dir> [extra vllm args]")
    a = ap.parse_args(argv)
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    model = a.model or ("27b" if a.profile.startswith("27b-") else "svl" if a.profile.startswith("svl-")
                        else "ornith" if a.profile.startswith("ornith-") else "flash-next")
    # <checkpoint-dir> positional, or MODEL= from the environment (then every argument is an extra vllm arg)
    if rest and not rest[0].startswith("-"):
        ckpt, extra = rest[0], rest[1:]
    else:
        ckpt, extra = os.environ.get("MODEL") or os.environ.get("MERLIN_CHECKPOINT_DIR") or "", rest
    if not ckpt:
        print(f"usage: launch.py --profile {a.profile} -- <checkpoint-dir> [extra vllm args]   (or MODEL=<dir>)", file=sys.stderr)
        return 2
    try:
        plan = build(model, a.profile, ckpt, dict(os.environ), extra, a.kv, a.checkpoint, status=a.status)
        return run(plan)
    except LaunchError as e:
        if str(e):
            print(str(e), flush=True)
        return e.code
    except R.ProfileError as e:
        print(f"launch: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
