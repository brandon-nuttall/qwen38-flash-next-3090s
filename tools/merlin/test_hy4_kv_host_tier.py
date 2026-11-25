#!/usr/bin/env python3
"""Unit and simulation tests for Hy4 Flash host-RAM KV cache offloading.

Covers:
1. Registry sizing, launcher plans, memory warnings, and setup CLI dry-runs
   (`tools.merlin.registry`, `launch`, `hy4_setup`, `setup_cli`) with no engine stubs.
2. Non-last PP rank commit-ack handling (`vllm.merlin.kv_commit_acks`) comparing
   non-deferred `DeepseekV41ModelState` (Hy4 Flash) vs deferred `MambaHybridModelState`.
3. End-to-end Hy4 Flash unit lifecycle (`kv_host_tier_units`, `kv_host_tier_writeback`,
   `kv_tier_lookup`, `BlockTierState`, `HybridKVCacheCoordinator`, `KVCacheManager`,
   `SimpleCPUOffloadScheduler`, `MerlinKVHostTierConnector`) across DCP2 and DCP4:
   - unaligned boundary alignment (512 / 1024), multi-FULL anchoring (MLA + IndexPool),
     and `CircularBufferSpec` exclusion
   - running SWA dependency retention across decode block releases and shift on advance
   - immediate publication of host-complete units (`record_unit` / `record_preempted`)
   - full and mixed GPU+host union restore across MLA + IndexPool + all SWA groups
   - multi-turn supersession (`2048 -> 4096 -> 6144`) reclaiming prior turn SWA rows
   - grace pin activation and hit-triggered release with alignment-aware reach
   - backward compatibility on a Mamba + FA recurrent layout
4. Validation exemptions in `HYV4FlashForCausalLM` (`validate_runtime`) and
   `HYV4DFlySpeculator.__init__` when `kv_host_tier.is_active` is True vs Nixl PD.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.machinery
import importlib.metadata
import inspect
import io
import json
import os
import sys
import tempfile
import types
import typing
import unittest
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

TOOLS_MERLIN = Path(__file__).resolve().parent
REPO = TOOLS_MERLIN.parents[1]
for p in (str(REPO), str(TOOLS_MERLIN)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import tooling before installing any CPU-only vLLM stubs so tooling tests run pure.
import hy4_setup as H  # noqa: E402
import launch as L  # noqa: E402
import preflight as pf  # noqa: E402
import registry as R  # noqa: E402
import setup_cli as SC  # noqa: E402


# ---------------------------------------------------------------------------
# CPU-only stub importer for running vLLM scheduler/tier modules without torch
# ---------------------------------------------------------------------------
_STUBS_INSTALLED = False


class _StubBase:
    def __init__(self, *a, **kw):
        pass

    def __init_subclass__(cls, **kw):
        pass

    def __getattr__(self, name):
        return _Attr(name)


class _Attr(types.ModuleType):
    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Attr(f"{self.__name__}.{name}")

    def __call__(self, *a, **kw):
        if (
            len(a) == 3
            and isinstance(a[0], str)
            and isinstance(a[1], tuple)
            and isinstance(a[2], dict)
        ):
            return type(a[0], (object,), a[2])
        return _Attr(self.__name__)

    def __mro_entries__(self, bases):
        return (_StubBase,)

    def __getitem__(self, item):
        return self

    def __or__(self, other):
        return typing.Any

    def __ror__(self, other):
        return typing.Any

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


class _TensorMeta(type):
    def __getattr__(cls, name):
        return lambda *a, **kw: None


class _Tensor(metaclass=_TensorMeta):
    def __getattr__(self, name):
        return lambda *a, **kw: self


class _Struct:
    def __init_subclass__(cls, **kwargs):
        pass

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _pydantic_dataclass(*args, config=None, **kwargs):
    def wrap(cls):
        anns = getattr(cls, "__annotations__", {})
        for k, v in list(cls.__dict__.items()):
            ann = anns.get(k)
            if ann is None or "ClassVar" in str(ann):
                continue
            if isinstance(v, (list, dict, set)):
                setattr(
                    cls,
                    k,
                    dataclasses.field(
                        default_factory=lambda val=v: copy.deepcopy(val)
                    ),
                )
        return dataclasses.dataclass(cls, **kwargs)

    return wrap(args[0]) if (args and callable(args[0]) and len(args) == 1) else wrap


def _pydantic_field(
    *args, default=dataclasses.MISSING, default_factory=None, **kwargs
):
    if args and args[0] is not ...:
        default = args[0]
    if default_factory is not None:
        return dataclasses.field(default_factory=default_factory)
    if isinstance(default, (list, dict, set)):
        return dataclasses.field(
            default_factory=lambda val=default: copy.deepcopy(val)
        )
    if default is dataclasses.MISSING or (
        type(default).__name__ == "PydanticUndefinedType"
    ):
        return dataclasses.field(default=None)
    return dataclasses.field(default=default)


class _Subscriptable:
    def __class_getitem__(cls, item):
        return item


def _dec_or_noop(*a, **kw):
    return (
        a[0]
        if (a and callable(a[0]) and not kw and len(args_or_empty(a)) == 1)
        else (lambda fn: fn)
    )


def args_or_empty(a):
    return a


def _fake_infer_schema(fn, mutates_args=(), op_name=None):
    sig_params = list(inspect.signature(fn).parameters)
    params = ", ".join(f"Tensor {p}" for p in sig_params)
    ret_ann = str(fn.__annotations__.get("return", "Tensor"))
    if "tuple" in ret_ann.lower():
        n_ret = max(1, ret_ann.count("Tensor"))
    else:
        n_ret = len(mutates_args) or sum(
            1 for p in sig_params if p.startswith("x")
        )
    n_ret = max(1, n_ret)
    rets = ", ".join(["Tensor"] * n_ret)
    return f"({params}) -> ({rets})" if n_ret > 1 else f"({params}) -> Tensor"


class _StubFinder:
    PREFIXES = frozenset(
        {
            "aiohttp",
            "av",
            "blake3",
            "cachetools",
            "cbor2",
            "cloudpickle",
            "compressed_tensors",
            "cpuinfo",
            "cuda",
            "cv2",
            "decord",
            "einops",
            "fastapi",
            "filelock",
            "flashinfer",
            "gguf",
            "huggingface_hub",
            "jinja2",
            "librosa",
            "llguidance",
            "llvmlite",
            "lm_format_enforcer",
            "mistral_common",
            "msgspec",
            "numba",
            "nvidia",
            "openai",
            "openai_harmony",
            "outlines",
            "partial_json_parser",
            "PIL",
            "prometheus_client",
            "prometheus_fastapi_instrumentator",
            "protobuf",
            "pybase64",
            "pydantic",
            "pydantic_core",
            "pyzmq",
            "ray",
            "regex",
            "safetensors",
            "scipy",
            "sentencepiece",
            "setproctitle",
            "soundfile",
            "starlette",
            "tiktoken",
            "tokenizers",
            "torch",
            "tqdm",
            "transformers",
            "triton",
            "typing_extensions",
            "uvicorn",
            "uvloop",
            "vllm._C",
            "watchfiles",
            "xgrammar",
            "xxhash",
            "zmq",
        }
    )

    def find_spec(self, fullname, path, target=None):
        if fullname in self.PREFIXES or fullname.split(".")[0] in self.PREFIXES:
            return importlib.machinery.ModuleSpec(
                fullname, self, is_package=True
            )
        return None

    def create_module(self, spec):
        m = _Attr(spec.name)
        if spec.name == "torch":
            m.__version__ = "2.6.0"
            m.Tensor = _Tensor
            m.dtype = type("dtype", (), {})
            m.bfloat16 = "bfloat16"
            m.float16 = "float16"
            m.float32 = "float32"
            m.uint8 = "uint8"
            m.float8_e4m3fn = "float8_e4m3fn"
            m.finfo = lambda *a, **kw: types.SimpleNamespace(
                max=448.0, min=-448.0, eps=1e-6
            )
        elif spec.name == "torch.cuda":
            m.is_available = lambda: False
        elif spec.name == "torch.distributed":
            m.is_initialized = lambda: False
        elif spec.name == "torch.library":
            m.infer_schema = _fake_infer_schema
            m.custom_op = _dec_or_noop
            m.register_fake = _dec_or_noop
        elif spec.name == "torch.nn":
            m.Module = type("Module", (), {})
        elif spec.name == "pydantic":
            m.BaseModel = type(
                "BaseModel",
                (),
                {
                    "__init__": lambda self, **kw: [
                        setattr(self, k, v) for k, v in kw.items()
                    ]
                    and None
                },
            )
            m.ConfigDict = lambda **kw: kw
            m.Field = _pydantic_field
            m.SkipValidation = _Subscriptable
            m.TypeAdapter = lambda *a, **kw: type(
                "TA", (), {"validate_python": staticmethod(lambda x: x)}
            )()
            m.field_validator = _dec_or_noop
            m.model_validator = _dec_or_noop
        elif spec.name == "pydantic.dataclasses":
            m.dataclass = _pydantic_dataclass
        elif spec.name == "msgspec":
            m.Struct = _Struct
        elif spec.name == "regex":
            import re

            m.compile, m.match, m.search = re.compile, re.match, re.search
        elif spec.name == "typing_extensions":
            for k in dir(typing):
                if not hasattr(m, k):
                    setattr(m, k, getattr(typing, k))
            m.Self = typing.Self
            m.TypeAlias = typing.TypeAlias
        elif spec.name == "blake3":
            m.blake3 = hashlib.sha256
        elif spec.name == "cachetools":
            m.LRUCache = dict
        elif spec.name == "cpuinfo":
            m.get_cpu_info = lambda: {"brand_raw": "CPU"}
        return m

    def exec_module(self, module):
        pass


def ensure_vllm_stubs() -> None:
    global _STUBS_INSTALLED
    if _STUBS_INSTALLED:
        return
    try:
        import torch  # noqa: F401
        import msgspec  # noqa: F401
        import pydantic  # noqa: F401
    except ImportError:
        orig_ver = importlib.metadata.version
        importlib.metadata.version = (
            lambda name: "2.6.0"
            if name in ("torch", "vllm", "pydantic", "msgspec")
            else orig_ver(name)
        )
        sys.meta_path.append(_StubFinder())
    _STUBS_INSTALLED = True


# ---------------------------------------------------------------------------
# Group 1: Registry, Launch, Setup CLI, Profiles (Pure stdlib)
# ---------------------------------------------------------------------------
class TestRegistryLaunchSetup(unittest.TestCase):
    def test_all_registries_valid_and_hy4flash_restore_unit(self):
        for model in ("flash-next", "27b", "svl", "ornith", "hy4flash"):
            reg = R.load_registry(model)
            self.assertEqual(reg["model"], model)

        hy4 = R.load_registry("hy4flash")
        by_id = {p["id"]: p for p in hy4["profiles"]}
        self.assertEqual(R.restore_unit_rows("hy4flash", by_id["hy4flash-quad-3090"]), 2632)
        self.assertEqual(R.restore_unit_rows("hy4flash", by_id["hy4flash-octo-3090"]), 2632)
        self.assertEqual(R.restore_unit_rows("hy4flash", by_id["hy4flash-quad-3090-tp4"]), 1424)
        self.assertEqual(
            R.restore_unit_min_gib_per_rank("hy4flash", by_id["hy4flash-quad-3090"], 4), 1
        )
        self.assertEqual(
            R.restore_unit_min_gib_per_rank("hy4flash", by_id["hy4flash-quad-3090-tp4"], 4), 1
        )
        self.assertEqual(
            R.restore_unit_min_gib_per_rank("hy4flash", by_id["hy4flash-octo-3090"], 8), 1
        )

        # Regression: Flash-Next & 27B calculations unchanged
        fn_reg = R.load_registry("flash-next")
        self.assertEqual(R.restore_unit_rows("flash-next", fn_reg["profiles"][0]), 68)
        b27_reg = R.load_registry("27b")
        self.assertEqual(R.restore_unit_rows("27b", b27_reg["profiles"][0]), 277)
        self.assertEqual(R.restore_unit_min_gib_per_rank("27b", b27_reg["profiles"][0], 1), 9)

    def test_launch_build_hy4flash_kv_tier_and_swarm_refusal(self):
        dummy = "/tmp/nonexistent-hy4-ckpt"
        for pid, tier, cards in (
            ("hy4flash-quad-3090", "8", 4),
            ("hy4flash-quad-3090-tp4", "4", 4),
            ("hy4flash-octo-3090", "6", 8),
        ):
            plan = L.build(
                "hy4flash",
                pid,
                dummy,
                environ={"MERLIN_KV_HOST_TIER_GIB": tier},
                repo=REPO,
            )
            self.assertEqual(plan.env[L.FN_KV_HOST_TIER], tier)
            self.assertIn(L.KV_TIER_UNTESTED_HY4, plan.notes)
            self.assertTrue(
                any(
                    f"{L.FN_KV_HOST_TIER}={tier}: {tier} GiB per rank "
                    f"({int(tier) * cards} GiB across {cards} ranks)"
                    in n
                    for n in plan.notes
                ),
                plan.notes,
            )
            # DFly2 remains enabled by default alongside KV RAM cache
            self.assertIn("--speculative-config", plan.args)
            idx = plan.args.index("--speculative-config")
            spec = json.loads(plan.args[idx + 1])
            self.assertEqual(spec["method"], "dfly")
            self.assertEqual(spec["num_speculative_tokens"], L.HY4_DFLY_K)

        # Tier=0 disables and pops env var
        plan0 = L.build(
            "hy4flash",
            "hy4flash-quad-3090",
            dummy,
            environ={"MERLIN_KV_HOST_TIER_GIB": "0"},
            repo=REPO,
        )
        self.assertNotIn(L.FN_KV_HOST_TIER, plan0.env)
        self.assertNotIn(L.KV_TIER_UNTESTED_HY4, plan0.notes)

        # BF16 Engram + DFLY=0 + KV RAM cache
        plan_bf16 = L.build(
            "hy4flash",
            "hy4flash-quad-3090",
            dummy,
            environ={
                "MERLIN_KV_HOST_TIER_GIB": "8",
                "ENGRAM": "bf16",
                "DFLY": "0",
                "MERLIN_KV_HOST_TIER__COPY_CREDITS_D": "64",
                "MERLIN_KV_HOST_TIER_CENSUS": "1",
            },
            repo=REPO,
        )
        self.assertEqual(plan_bf16.engram, "bf16")
        self.assertEqual(plan_bf16.env["MERLIN_KV_HOST_TIER__COPY_CREDITS_D"], "64")
        self.assertEqual(plan_bf16.env["MERLIN_KV_HOST_TIER_CENSUS"], "1")
        self.assertNotIn("--speculative-config", plan_bf16.args)
        # 140 + 58 (bf16 diff) + 32 (4x8g tier) = 230 GiB
        self.assertTrue(
            any("ENGRAM=bf16" in n and "~230 GiB" in n for n in plan_bf16.notes),
            plan_bf16.notes,
        )

        # Invalid tier strings refused cleanly (including unicode digit & >6 chars)
        for bad in ("abc", "²", "1234567"):
            with self.assertRaises(L.LaunchError) as ctx:
                L.build(
                    "hy4flash",
                    "hy4flash-quad-3090",
                    dummy,
                    environ={"MERLIN_KV_HOST_TIER_GIB": bad},
                    repo=REPO,
                )
            self.assertEqual(ctx.exception.code, 2)

        # Pacing/census rejected when tier is 0
        with self.assertRaises(L.LaunchError):
            L.build(
                "hy4flash",
                "hy4flash-quad-3090",
                dummy,
                environ={"MERLIN_KV_HOST_TIER_GIB": "0", "MERLIN_KV_HOST_TIER_CENSUS": "1"},
                repo=REPO,
            )

        # SWARM=1 remains refused cleanly
        with self.assertRaises(L.LaunchError) as ctx:
            L.build(
                "hy4flash",
                "hy4flash-quad-3090",
                dummy,
                environ={"SWARM": "1"},
                repo=REPO,
            )
        self.assertEqual(ctx.exception.code, 2)
        self.assertEqual(str(ctx.exception), L.swarm_not_qualified("hy4flash-quad-3090"))

    def test_hy4_memory_warnings_and_serve_hints(self):
        fake_mem = "MemAvailable: 150000000 kB\n"  # ~143 GiB
        with mock.patch("resource.getrlimit", return_value=(4 * 2**30, 4 * 2**30)):
            # Need 140 + 32 = 172 GiB > 143 GiB; soft memlock 4 GiB < 8 GiB & < 60 GiB
            warns = L._hy4_memory_warnings(
                "fp8",
                tier_total_gib=32,
                tier_per_rank_gib=8,
                read=lambda path: fake_mem if "meminfo" in path else "CapEff:\t0\n",
            )
        self.assertEqual(len(warns), 3)
        self.assertIn("+ KV RAM cache 32 GiB", warns[0])
        self.assertIn("KV RAM cache (8 GiB per rank)", warns[1])
        self.assertIn("Engram tables", warns[2])

        hints_on = H.serve_hints(dfly_on=True, engram="fp8")
        hints_off = H.serve_hints(dfly_on=False, engram="fp8")
        self.assertIn("DFly2 speculative decoding is on by default", hints_on[1])
        self.assertIn("DFLY=0", hints_off[1])
        self.assertNotIn("KV RAM cache keeps it off", hints_off[1])

    def test_setup_cli_hy4_dry_runs(self):
        args_ok = SC.parse_args(
            ["--model", "hy4flash", "--simulate", "4x3090", "--kv-ram-cache", "32g", "--yes"]
        )
        self.assertEqual(args_ok.kv_ram_cache_gib, 32)
        self.assertIsNone(SC.hy4_flag_problem(args_ok))

        # Verify simulated dry-run passes on 4x3090:ram=256 with --kv-ram-cache 32g
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(SC, "git_problem", return_value=None),
            mock.patch.object(SC, "disk_free_for", return_value=lambda _: 1 << 40),
        ):
            args = SC.parse_args(
                [
                    "--model",
                    "hy4flash",
                    "--simulate",
                    "4x3090:ram=256",
                    "--models-dir",
                    tmp,
                    "--kv-ram-cache",
                    "32g",
                    "--dry-run",
                    "--yes",
                ]
            )
            inv = pf.simulate(args.simulate)
            buf = io.StringIO()
            ui = SC.Asker(yes=True, out=buf)
            a = SC.Answers(model="hy4flash", models_dir=tmp)
            key_path = Path(tmp) / "api-key"
            rc = SC.hy4_main(
                args, a, inv, ui, REPO, SC.sizes_table(REPO), key_path, lambda _: None, {}
            )
            out = buf.getvalue()
            self.assertEqual(rc, 0, out)
            self.assertIn("KV RAM cache", out)
            self.assertIn("32g per server (8 GiB per rank", out)
            self.assertIn("MERLIN_KV_HOST_TIER_GIB=8", out)

            # 8x3090:ram=320 with --kv-ram-cache 16g -> 2 GiB/rank
            args8 = SC.parse_args(
                [
                    "--model",
                    "hy4flash",
                    "--simulate",
                    "8x3090:ram=320",
                    "--models-dir",
                    tmp,
                    "--kv-ram-cache",
                    "16g",
                    "--dry-run",
                    "--yes",
                ]
            )
            inv8 = pf.simulate(args8.simulate)
            buf8 = io.StringIO()
            rc8 = SC.hy4_main(
                args8, SC.Answers(model="hy4flash", models_dir=tmp), inv8,
                SC.Asker(yes=True, out=buf8),
                REPO, SC.sizes_table(REPO), key_path, lambda _: None, {},
            )
            out8 = buf8.getvalue()
            self.assertEqual(rc8, 0, out8)
            self.assertIn("MERLIN_KV_HOST_TIER_GIB=2", out8)

            # Too small (<1 GiB per rank): returns FAIL cleanly without duplicate launch error
            args_small = SC.parse_args(
                [
                    "--model",
                    "hy4flash",
                    "--simulate",
                    "4x3090:ram=256",
                    "--models-dir",
                    tmp,
                    "--kv-ram-cache",
                    "2g",
                    "--dry-run",
                    "--yes",
                ]
            )
            buf_small = io.StringIO()
            rc_small = SC.hy4_main(
                args_small, SC.Answers(model="hy4flash", models_dir=tmp), inv,
                SC.Asker(yes=True, out=buf_small), REPO, SC.sizes_table(REPO),
                key_path, lambda _: None, {},
            )
            self.assertEqual(rc_small, 1)
            out_small = buf_small.getvalue()
            self.assertIn("needs at least 1 GiB per rank (4g per server)", out_small)
            self.assertNotIn("FAIL  launch", out_small)

        # --swarm refused cleanly on parse_args and SWARM=1 refused in hy4_main
        with self.assertRaises(SC.SetupError):
            SC.parse_args(["--model", "hy4flash", "--swarm", "--yes"])
        with tempfile.TemporaryDirectory() as tmp:
            args0 = SC.parse_args(
                ["--model", "hy4flash", "--simulate", "4x3090:ram=256", "--models-dir", tmp, "--dry-run", "--yes"]
            )
            inv = pf.simulate(args0.simulate)
            with self.assertRaises(SC.SetupError) as ctx:
                SC.hy4_main(
                    args0, SC.Answers(model="hy4flash", models_dir=tmp), inv,
                    SC.Asker(yes=True), REPO, SC.sizes_table(REPO), Path(tmp) / "k",
                    lambda _: None, {"SWARM": "1"},
                )
            self.assertEqual(ctx.exception.code, 2)


# ---------------------------------------------------------------------------
# Group 2: Non-last PP rank commit acks
# ---------------------------------------------------------------------------
class TestHy4CommitAcks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_vllm_stubs()

    def test_pp2_commit_acks_hy4_vs_mamba(self):
        from vllm.merlin import kv_commit_acks
        from vllm.v1.worker.gpu.model_states.interface import ModelState

        class FakeDeepseekV41State(ModelState):
            def prepare_attn(self, *args, **kwargs):
                return None

            def prepare_dummy_inputs(self, *args, **kwargs):
                return None

            def prepare_inputs(self, *args, **kwargs):
                return None

            def prepare_inputs_embeds(self, *args, **kwargs):
                return None

        class FakeMambaState(FakeDeepseekV41State):
            def postprocess_state(self, *args, **kwargs):
                return None

        class FakeEvent:
            def __init__(self, ready: bool = True):
                self.ready = ready

            def query(self) -> bool:
                return self.ready

        class FakeRunner:
            def __init__(self, model_state):
                self.model_state = model_state
                tracker = kv_commit_acks.WorkerCommitAcks(record_event=FakeEvent)
                worker = types.SimpleNamespace(commit_acks=tracker)
                inner = types.SimpleNamespace(worker_handler=worker)
                self.kv_connector = types.SimpleNamespace(_disabled=False, kv_connector=inner)
                self.pp_handler = types.SimpleNamespace(
                    is_last_rank=False, queue=[types.SimpleNamespace()]
                )
                self.tracker = tracker

        # 1) Hy4 Flash (no deferred writer): never holds or masks on PP0
        hy4_runner = FakeRunner(FakeDeepseekV41State.__new__(FakeDeepseekV41State))
        sched_out = types.SimpleNamespace(
            total_num_scheduled_tokens=64,
            kv_connector_metadata=types.SimpleNamespace(scheduled_step=1, acked_step=0),
        )
        kv_commit_acks.begin_step(hy4_runner, sched_out)
        self.assertIsNone(hy4_runner.tracker._undecided)
        hy4_runner.tracker.finish_forward()
        kv_commit_acks.defer_step(hy4_runner, input_batch=types.SimpleNamespace(req_ids=["r1"]))
        self.assertIsNone(getattr(hy4_runner.pp_handler.queue[-1], "commit_ack", None))
        committed, masked = hy4_runner.tracker.report()
        self.assertEqual(committed, 1)
        self.assertEqual(masked, set())

        # 2) MambaHybridModelState (deferred writer): holds until defer_step & still masks
        mamba_runner = FakeRunner(FakeMambaState.__new__(FakeMambaState))
        kv_commit_acks.begin_step(mamba_runner, sched_out)
        self.assertEqual(mamba_runner.tracker._undecided, 1)
        mamba_runner.tracker.finish_forward()
        # Held: cannot advance while undecided
        self.assertEqual(mamba_runner.tracker.report()[0], 0)
        kv_commit_acks.defer_step(
            mamba_runner, input_batch=types.SimpleNamespace(req_ids=["r1"])
        )
        ack = getattr(mamba_runner.pp_handler.queue[-1], "commit_ack", None)
        self.assertIsNotNone(ack)
        ack.mask([True])
        ack.finish()
        committed_m, masked_m = mamba_runner.tracker.report()
        self.assertEqual(committed_m, 1)
        self.assertEqual(masked_m, {("r1", 1)})

        # 3) None fallback: conservatively deferred
        none_runner = FakeRunner(None)
        kv_commit_acks.begin_step(none_runner, sched_out)
        self.assertEqual(none_runner.tracker._undecided, 1)


# ---------------------------------------------------------------------------
# Group 3: End-to-End Hy4 Flash Unit Lifecycle, Retention, Eviction, Restore
# ---------------------------------------------------------------------------
class _SimRequest:
    def __init__(
        self,
        request_id: str,
        block_hashes: list,
        num_tokens: int,
        num_prompt_tokens: int | None = None,
    ):
        from vllm.v1.request import RequestStatus

        self.request_id = request_id
        self.block_hashes = block_hashes
        self.num_tokens = num_tokens
        self.num_prompt_tokens = num_tokens if num_prompt_tokens is None else num_prompt_tokens
        self.num_computed_tokens = 0
        self.num_in_flight_tokens = 0
        self.num_output_placeholders = 0
        self.num_stale_output_tokens = 0
        self.drop_stale_output = False
        self.shared_prefix_boundary = 0
        self.replay_start = 0
        self.is_prefill_chunk = False
        self.resume_needs_rewind = False
        self.next_decode_eligible_step = 0
        self.last_sched_seq = 0
        self.spec_token_ids: list[int] = []
        self.skip_reading_prefix_cache = False
        self.num_output_tokens = max(0, self.num_tokens - self.num_prompt_tokens)
        self.status = RequestStatus.RUNNING
        self.sampling_params = types.SimpleNamespace(max_tokens=512)
        self.pooling_params = None
        self.mm_features: list = []
        self.kv_cache_report_mode = "incremental"
        self.cache_salt = None
        self.priority = 0

    def is_finished(self) -> bool:
        from vllm.v1.request import RequestStatus

        return RequestStatus.is_finished(self.status)


def _make_hashes(prefix_seed: bytes, num_blocks: int) -> list:
    from vllm.v1.core.kv_cache_utils import BlockHash

    out = []
    cur = hashlib.sha256(prefix_seed).digest()
    for i in range(num_blocks):
        cur = hashlib.sha256(cur + i.to_bytes(4, "big")).digest()
        out.append(BlockHash(cur))
    return out


class TestHy4UnitLifecycleAndRestore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_vllm_stubs()

    def _build_hy4_config(
        self, *, num_gpu_blocks: int, dcp: int, pp: int
    ):
        from vllm.v1.kv_cache_interface import (
            CircularBufferSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
            MLAAttentionSpec,
            SlidingWindowMLASpec,
            SparseCacheRole,
        )

        # Sparse MLA + IndexPool (tokens_per_state=4) + SWA groups (9 per PP2 stage) + Ring
        groups: list[KVCacheGroupSpec] = [
            KVCacheGroupSpec(
                layer_names=["mla.0"],
                kv_cache_spec=MLAAttentionSpec(
                    block_size=64,
                    num_kv_heads=1,
                    head_size=576,
                    dtype="float8_e4m3fn",
                    cache_dtype_str="fp8_ds_mla",
                    dcp_sharded=True,
                    tokens_per_state=1,
                    state_content_bytes=656,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["idx.0"],
                kv_cache_spec=MLAAttentionSpec(
                    block_size=256,
                    num_kv_heads=1,
                    head_size=128,
                    dtype="uint8",
                    cache_dtype_str="fp8_ds_mla",
                    dcp_sharded=True,
                    tokens_per_state=4,
                    cache_role=SparseCacheRole.INDEXER,
                    state_content_bytes=128,
                ),
            ),
        ]
        n_swa = 9 * max(1, 2 // pp)
        for s in range(n_swa):
            groups.append(
                KVCacheGroupSpec(
                    layer_names=[f"swa.{s}"],
                    kv_cache_spec=SlidingWindowMLASpec(
                        block_size=64,
                        num_kv_heads=1,
                        head_size=512,
                        dtype="float8_e4m3fn",
                        cache_dtype_str="fp8_ds_mla",
                        sliding_window=512,
                        dcp_sharded=False,
                        state_content_bytes=512,
                    ),
                )
            )
        groups.append(
            KVCacheGroupSpec(
                layer_names=["ring.0"],
                kv_cache_spec=CircularBufferSpec(
                    block_size=8,
                    num_kv_heads=1,
                    head_size=128,
                    dtype="bfloat16",
                    dcp_sharded=False,
                    state_content_bytes=128,
                ),
            )
        )
        cfg = KVCacheConfig(
            num_blocks=num_gpu_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=num_gpu_blocks * 251904,
                    layers=[ln for g in groups for ln in g.layer_names],
                    layer_stride=251904,
                    block_stride=251904,
                )
            ],
            kv_cache_groups=groups,
            hash_block_size=64,
            cache_hit_alignment_tokens=64 * dcp * 4,
        )
        return cfg

    def _build_stack(self, *, dcp: int, pp: int, num_gpu_blocks: int = 800, num_cpu_blocks: int = 3200):
        from vllm.merlin import kv_host_tier, kv_host_tier_units, kv_host_tier_writeback
        from vllm.merlin import kv_restore_dependencies
        from vllm.merlin.kv_tier_lookup import closure_alignment
        from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

        cfg = self._build_hy4_config(num_gpu_blocks=num_gpu_blocks, dcp=dcp, pp=pp)
        sched_bs = 64 * 4 * dcp  # 512 for DCP2, 1024 for DCP4
        kv_mgr = KVCacheManager(
            kv_cache_config=cfg,
            max_model_len=32768,
            scheduler_block_size=sched_bs,
            hash_block_size=64,
            enable_caching=True,
            use_eagle=False,
            dcp_world_size=dcp,
            pcp_world_size=1,
            causal_sliding_window_kv=True,
        )
        cpu_cfg = copy.deepcopy(cfg)
        cpu_cfg.num_blocks = num_cpu_blocks
        cpu_cfg.kv_cache_tensors = [
            dataclasses.replace(cfg.kv_cache_tensors[0], size=num_cpu_blocks * 251904)
        ]
        cpu_coord = get_kv_cache_coordinator(
            kv_cache_config=cpu_cfg,
            max_model_len=32768,
            max_in_flight_tokens=32768,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=dcp,
            pcp_world_size=1,
            scheduler_block_size=sched_bs,
            hash_block_size=64,
            allow_partial_hash_hits=False,
            causal_sliding_window_kv=True,
        )
        # Verify alignment properties
        self.assertFalse(cpu_coord.enable_partial_hash_hits)
        self.assertEqual(cpu_coord.scheduler_block_size, sched_bs)
        align = closure_alignment(cpu_coord, cpu_coord.scheduler_block_size)
        self.assertEqual(align, sched_bs)

        units = kv_host_tier_units.mount(
            kv_mgr.block_pool,
            cpu_coord.block_pool,
            cpu_cfg,
            cpu_cfg.prefix_cacheable_group_ids,
            cpu_coord.group_block_sizes,
            64,
            cpu_coord.eagle_group_ids,
            kv_restore_dependencies.fine_grained_group_ids(cpu_coord),
            alignment_tokens=align,
        )
        wb = kv_host_tier_writeback.WriteBack(
            cpu_coord.block_pool,
            cpu_cfg.prefix_cacheable_group_ids,
            reserve_blocks=16,
            alignment_groups=len(cpu_cfg.prefix_cacheable_group_ids),
            world_size=1,
            credits_cap=128,
        )
        wb.bind(kv_mgr.block_pool, units)

        # Exercise SimpleCPUOffloadScheduler.bind_kv_cache_manager and record_unit
        fake_sched = SimpleCPUOffloadScheduler.__new__(SimpleCPUOffloadScheduler)
        fake_sched.tier_runtime = types.SimpleNamespace(
            units=units,
            row_bytes=0,
            on_request_finished=lambda req: None,
            on_registered=lambda b, wk: None,
        )
        fake_sched.writeback = wb
        fake_sched.tier_plans = None
        fake_sched.cpu_kv_cache_config = cpu_cfg
        fake_sched.tier_hooks = None
        fake_sched._sched_step = 1
        fake_sched._gpu_block_pool = kv_mgr.block_pool
        fake_sched.cpu_block_pool = cpu_coord.block_pool
        fake_sched.bind_kv_cache_manager(kv_mgr)

        # Confirm grace pins enabled by bind_kv_cache_manager even without partial hash hits!
        self.assertTrue(kv_mgr.enable_grace_pins)
        # Verify min_tier_rows exact DCP scaling
        self.assertEqual(
            kv_host_tier.min_tier_rows(cpu_cfg, cpu_coord.group_block_sizes),
            2632 if pp == 2 else 1424,
        )
        return kv_mgr, cpu_coord, units, wb, fake_sched

    def _run_prefill_and_commit_gpu(self, kv_mgr, tier, req, total_tokens: int, step_start: int = 1) -> int:
        step = step_start
        tier.advance_step(step)
        blocks = kv_mgr.allocate_slots(req, num_new_tokens=total_tokens)
        self.assertIsNotNone(blocks)
        req.num_computed_tokens = total_tokens
        kv_mgr.coordinator.cache_blocks(req, total_tokens)
        tier.set_acked_step(step)
        kv_mgr.new_step_starts()
        return step

    def _commit_unit_to_host(self, gpu_pool, cpu_pool, units, unit, step: int) -> None:
        gpu_blocks = units.gpu_members(unit)
        host_blocks = cpu_pool.get_new_blocks(len(gpu_blocks))
        tier = gpu_pool.tier
        for g_blk, h_blk in zip(gpu_blocks, host_blocks, strict=True):
            keys = [g_blk.block_hash, *gpu_pool.cached_block_hashes_by_block.get(g_blk.block_id, ())]
            cpu_pool._insert_block_hash(keys[0], h_blk, num_tokens=g_blk.block_hash_num_tokens)
            for sec in keys[1:]:
                cpu_pool._insert_block_hash(sec, h_blk, num_tokens=None)
            stamp = tier.stamp_of(g_blk)
            tier.note_host_copy(h_blk.block_id, stamp.cid if stamp is not None else 0)
        units.on_host_registered(host_blocks)
        cpu_pool.free_blocks(host_blocks)
        units.publish_unit(unit.uid, step)

    def test_dcp2_lifecycle_swa_retention_restore_supersede_grace(self):
        from vllm.merlin import kv_tier_lookup
        from vllm.v1.core.kv_tier_state import BlockState
        from vllm.v1.request import RequestStatus

        kv_mgr, cpu_coord, units, wb, fake_sched = self._build_stack(dcp=2, pp=2, num_gpu_blocks=900, num_cpu_blocks=3200)
        gpu_pool = kv_mgr.block_pool
        cpu_pool = cpu_coord.block_pool
        tier = gpu_pool.tier
        self.assertEqual(units.alignment_tokens, 512)

        # Turn 1: prompt 2240 -> decode to 2300 (between 2048 and 2560 aligned boundaries)
        hashes = _make_hashes(b"conv1", 120)  # 7680 tokens
        req1 = _SimRequest("r1", hashes, num_tokens=2240)
        self._run_prefill_and_commit_gpu(kv_mgr, tier, req1, 2240, step_start=1)
        wb.account_running(kv_mgr.coordinator, (), [req1], {"r1": 1}, token_budget=1024, max_model_len=32768)

        # Simulate decode steps up to 2300; SWA blocks in [1536..2048) drop off active window
        for cur in range(2241, 2301):
            req1.num_tokens = cur
            req1.num_output_tokens = cur - 2240
            tier.advance_step(cur)
            blk = kv_mgr.allocate_slots(req1, num_new_tokens=1)
            self.assertIsNotNone(blk)
            req1.num_computed_tokens = cur
            kv_mgr.coordinator.cache_blocks(req1, cur)
            kv_mgr.coordinator.remove_skipped_blocks("r1", cur)
            tier.set_acked_step(cur)
            wb.account_running(kv_mgr.coordinator, (), [req1], {"r1": 1}, token_budget=1024, max_model_len=32768)

        # Crucial check: every freed SWA block in [1536..2048) stays retained (not CLEAN!)
        # and circular buffer (ring.0, gid=11) is NOT in prefix-cacheable groups.
        manifest_mid = units.manifest(req1)
        self.assertIsNotNone(manifest_mid)
        self.assertEqual(manifest_mid.boundary, 2048)
        for gid in range(2, 11):  # 9 SWA groups
            for tok in range(1600, 2049, 64):
                k = make_key = (hashes[tok // 64 - 1], gid)
                from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
                bk = make_block_hash_with_group_id(make_key[0], make_key[1])
                b = gpu_pool.cached_block_hash_to_block.get_one_block(bk)
                self.assertIsNotNone(b, f"SWA block missing gid={gid} tok={tok}")
                self.assertTrue(units.retains(b), f"SWA block lost retention gid={gid} tok={tok}")
                self.assertIs(tier.state(b), BlockState.DIRTY)

        # Advancing past 2560 shifts _running_sw_keys forward to [2048..2560) and releases <2048 SWA blocks
        for cur in range(2301, 2561):
            req1.num_tokens = cur
            req1.num_output_tokens = cur - 2240
            tier.advance_step(cur)
            kv_mgr.allocate_slots(req1, num_new_tokens=1)
            req1.num_computed_tokens = cur
            kv_mgr.coordinator.cache_blocks(req1, cur)
            kv_mgr.coordinator.remove_skipped_blocks("r1", cur)
            tier.set_acked_step(cur)
            wb.account_running(kv_mgr.coordinator, (), [req1], {"r1": 1}, token_budget=1024, max_model_len=32768)

        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        old_swa_key = make_block_hash_with_group_id(hashes[1600 // 64 - 1], 2)
        old_swa_blk = gpu_pool.cached_block_hash_to_block.get_one_block(old_swa_key)
        self.assertIsNotNone(old_swa_blk)
        self.assertFalse(units.retains(old_swa_blk))

        # Rewind test setup to Turn 1 finishing at 2300 -> boundary 2048 recorded + host-published
        req1_fin = _SimRequest("r1_fin", hashes, num_tokens=2240)
        kv_mgr2, cpu_coord2, units2, wb2, fake_sched2 = self._build_stack(dcp=2, pp=2, num_gpu_blocks=1600, num_cpu_blocks=4000)
        gpu2, cpu2, tier2 = kv_mgr2.block_pool, cpu_coord2.block_pool, kv_mgr2.block_pool.tier
        self._run_prefill_and_commit_gpu(kv_mgr2, tier2, req1_fin, 2240, step_start=1)
        wb2.account_running(kv_mgr2.coordinator, (), [req1_fin], {"r1_fin": 1}, token_budget=1024, max_model_len=32768)
        for cur in range(2241, 2301):
            req1_fin.num_tokens = cur
            req1_fin.num_output_tokens = cur - 2240
            tier2.advance_step(cur)
            kv_mgr2.allocate_slots(req1_fin, num_new_tokens=1)
            req1_fin.num_computed_tokens = cur
            kv_mgr2.coordinator.cache_blocks(req1_fin, cur)
            kv_mgr2.coordinator.remove_skipped_blocks("r1_fin", cur)
            tier2.set_acked_step(cur)
            wb2.account_running(kv_mgr2.coordinator, (), [req1_fin], {"r1_fin": 1}, token_budget=1024, max_model_len=32768)

        # Clear _running_sw_keys once running set is empty (turn finished)
        wb2.account_running(kv_mgr2.coordinator, (), [], {}, token_budget=1024, max_model_len=32768)
        req1_fin.status = RequestStatus.FINISHED_STOPPED
        unit1 = fake_sched2.record_unit(req1_fin)
        self.assertIsNotNone(unit1)
        self.assertEqual(unit1.boundary, 2048)

        # Free Turn 1 via KVCacheManager -> takes grace pin!
        kv_mgr2.free(req1_fin)
        self.assertEqual(kv_mgr2.num_grace_pins, 1)

        # Test _release_grace_for_hit with align=512 reach (`2048 + 512 >= 2240`):
        req1_resumed = _SimRequest("r1_resume", hashes, num_tokens=2350)
        computed_blks, num_hit, _ = kv_mgr2.get_computed_blocks(req1_resumed)
        self.assertEqual(num_hit, 2048)
        kv_mgr2._release_grace_for_hit(computed_blks.blocks, num_hit)
        self.assertEqual(kv_mgr2.num_grace_pins, 0)

        # Commit Unit 1 to host and verify evicting all GPU blocks still restores 2048 from host!
        self._commit_unit_to_host(gpu2, cpu2, units2, unit1, step=2301)
        self.assertTrue(units2.is_published(unit1.uid))
        self.assertTrue(units2.is_host_complete(unit1))

        # Re-record equivalent finished request whose blocks are already on host -> supersedes unit1 with immediate publish
        req1_dup = _SimRequest("r1_dup", hashes, num_tokens=2048)
        req1_dup.num_computed_tokens = 2048
        req1_dup.status = RequestStatus.FINISHED_STOPPED
        unit_dup = fake_sched2.record_unit(req1_dup)
        self.assertIsNotNone(unit_dup)
        self.assertTrue(units2.is_published(unit_dup.uid))
        self.assertNotIn(unit1.uid, units2.units)

        # Test MerlinKVHostTierConnector.record_preempted delegates to record_unit
        from vllm.distributed.kv_transfer.kv_connector.v1.merlin_kv_host_tier_connector import (
            MerlinKVHostTierConnector,
        )
        fake_conn = MerlinKVHostTierConnector.__new__(MerlinKVHostTierConnector)
        called = []
        fake_mgr_stub = types.SimpleNamespace(
            tier_runtime=types.SimpleNamespace(units=types.SimpleNamespace(claim_waiting=lambda r, front=True: called.append(("claim", r.request_id, front)))),
            record_unit=lambda r: called.append(("record_unit", r.request_id)),
        )
        fake_conn.scheduler_manager = fake_mgr_stub
        fake_conn.record_preempted(req1_dup)
        self.assertEqual(called, [("record_unit", "r1_dup"), ("claim", "r1_dup", True)])

        # Evict every GPU block and verify full host restore across all 11 prefix-cacheable groups!
        gpu2.evict_blocks({b.block_id for b in gpu2.blocks if b.block_hash is not None})
        plan = kv_tier_lookup.find_longest_union_hit(
            kv_mgr2.coordinator,
            hashes,
            max_length=2350 - 1,
            host_pool=cpu2,
            version=kv_tier_lookup.tier_version,
            epoch=tier2.epoch,
        )
        self.assertEqual(plan.boundary, 2048)
        self.assertEqual(plan.gpu_boundary, 0)
        # All 11 prefix-cacheable groups restored from HOST (group 11 CircularBuffer is empty)
        self.assertEqual(len(plan.groups), 12)
        for gid in range(11):
            self.assertTrue(any(s is not None and s.tier is kv_tier_lookup.Tier.HOST for s in plan.groups[gid]))
        self.assertEqual(plan.groups[11], ())

        # Multi-turn supersession & mixed GPU+host restore: Turn 2 (4096) and Turn 3 (6144)
        req2 = _SimRequest("r2", hashes, num_tokens=4096)
        self._run_prefill_and_commit_gpu(kv_mgr2, tier2, req2, 4096, step_start=2302)
        req2.status = RequestStatus.FINISHED_STOPPED
        unit2 = fake_sched2.record_unit(req2)
        self.assertEqual(unit2.boundary, 4096)
        self._commit_unit_to_host(gpu2, cpu2, units2, unit2, step=2303)
        kv_mgr2.free(req2)
        kv_mgr2.release_grace_pin("r2", "hit")
        # Turn 2 keeps immediate predecessor unit_dup (2048) until Turn 3 publishes
        self.assertIn(unit_dup.uid, units2.units)

        # Mixed GPU+host lookup: after Turn 2 prefill (0..4096) and evicting > 2048 on GPU,
        # GPU still holds [0..2048) of MLA+IndexPool while HOST holds [2048..4096) + SWA.
        gpu2.evict_blocks({
            b.block_id
            for b in gpu2.blocks
            if b.block_hash is not None and (b.block_hash_num_tokens or 0) > 2048
        })
        plan_mixed = kv_tier_lookup.find_longest_union_hit(
            kv_mgr2.coordinator,
            hashes,
            max_length=4096 + 100,
            host_pool=cpu2,
            version=kv_tier_lookup.tier_version,
            epoch=tier2.epoch,
        )
        self.assertEqual(plan_mixed.boundary, 4096)
        self.assertTrue(any(s is not None and s.tier is kv_tier_lookup.Tier.GPU for s in plan_mixed.groups[0]))
        self.assertTrue(any(s is not None and s.tier is kv_tier_lookup.Tier.HOST for s in plan_mixed.groups[0]))
        self.assertTrue(any(s is not None and s.tier is kv_tier_lookup.Tier.HOST for s in plan_mixed.groups[2]))

        req3 = _SimRequest("r3", hashes, num_tokens=6144)
        self._run_prefill_and_commit_gpu(kv_mgr2, tier2, req3, 6144, step_start=2304)
        req3.status = RequestStatus.FINISHED_STOPPED
        unit3 = fake_sched2.record_unit(req3)
        self.assertEqual(unit3.boundary, 6144)
        self._commit_unit_to_host(gpu2, cpu2, units2, unit3, step=2305)
        # Turn 3 keeps Turn 2 AND drops unit_dup (across sliding-window + MLA key changes!)
        self.assertNotIn(unit_dup.uid, units2.units)
        self.assertIn(unit2.uid, units2.units)
        self.assertIn(unit3.uid, units2.units)
        self.assertGreaterEqual(units2.counters.supersede_completed, 1)

    def test_dcp4_pp1_alignment_1024_and_restore(self):
        from vllm.merlin import kv_tier_lookup
        from vllm.v1.request import RequestStatus

        kv_mgr, cpu_coord, units, wb, fake_sched = self._build_stack(
            dcp=4, pp=1, num_gpu_blocks=800, num_cpu_blocks=2400
        )
        self.assertEqual(units.alignment_tokens, 1024)
        hashes = _make_hashes(b"dcp4", 64)  # 4096 tokens
        req = _SimRequest("r_dcp4", hashes, num_tokens=2240)
        self._run_prefill_and_commit_gpu(kv_mgr, kv_mgr.block_pool.tier, req, 2240, step_start=1)
        req.status = RequestStatus.FINISHED_STOPPED
        unit = fake_sched.record_unit(req)
        self.assertIsNotNone(unit)
        self.assertEqual(unit.boundary, 2048)
        self._commit_unit_to_host(kv_mgr.block_pool, cpu_coord.block_pool, units, unit, step=2)
        kv_mgr.free(req)
        kv_mgr.release_grace_pin("r_dcp4", "hit")
        kv_mgr.block_pool.evict_blocks(
            {b.block_id for b in kv_mgr.block_pool.blocks if b.block_hash is not None}
        )
        plan = kv_tier_lookup.find_longest_union_hit(
            kv_mgr.coordinator,
            hashes,
            max_length=2239,
            host_pool=cpu_coord.block_pool,
            version=kv_tier_lookup.tier_version,
            epoch=kv_mgr.block_pool.tier.epoch,
        )
        self.assertEqual(plan.boundary, 2048)
        self.assertEqual(len(plan.groups), 21)  # 1 MLA + 1 IndexPool + 18 SWA + 1 CircularBuffer

    def test_recurrent_mamba_fa_backward_compat(self):
        from vllm.merlin import kv_host_tier_units, kv_restore_dependencies
        from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
            MambaSpec,
        )
        from vllm.v1.request import RequestStatus

        groups = [
            KVCacheGroupSpec(
                layer_names=["fa.0"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=256, num_kv_heads=2, head_size=64, dtype="bfloat16", state_content_bytes=256
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["mamba.0"],
                kv_cache_spec=MambaSpec(
                    shapes=((16, 16),),
                    dtypes=("bfloat16",),
                    block_size=256,
                    mamba_type="mamba2",
                    mamba_cache_mode="align",
                    num_speculative_blocks=0,
                ),
            ),
        ]
        cfg = KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=64 * 1024,
                    layers=["fa.0", "mamba.0"],
                    layer_stride=1024,
                    block_stride=1024,
                )
            ],
            kv_cache_groups=groups,
            hash_block_size=64,
            cache_hit_alignment_tokens=64,
        )
        kv_mgr = KVCacheManager(
            kv_cache_config=cfg,
            max_model_len=4096,
            scheduler_block_size=256,
            hash_block_size=64,
            enable_caching=True,
            use_eagle=False,
            dcp_world_size=1,
            pcp_world_size=1,
            causal_sliding_window_kv=False,
        )
        cpu_cfg = copy.deepcopy(cfg)
        cpu_coord = get_kv_cache_coordinator(
            kv_cache_config=cpu_cfg,
            max_model_len=4096,
            max_in_flight_tokens=4096,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            scheduler_block_size=256,
            hash_block_size=64,
            allow_partial_hash_hits=True,
            causal_sliding_window_kv=False,
        )
        units = kv_host_tier_units.mount(
            kv_mgr.block_pool,
            cpu_coord.block_pool,
            cpu_cfg,
            cpu_cfg.prefix_cacheable_group_ids,
            cpu_coord.group_block_sizes,
            64,
            cpu_coord.eagle_group_ids,
            kv_restore_dependencies.fine_grained_group_ids(cpu_coord),
            alignment_tokens=64,
        )
        self.assertEqual(units.alignment_tokens, 64)
        hashes = _make_hashes(b"mamba", 16)
        req = _SimRequest("rm1", hashes, num_tokens=512)
        self._run_prefill_and_commit_gpu(kv_mgr, kv_mgr.block_pool.tier, req, 512, step_start=1)
        req.status = RequestStatus.FINISHED_STOPPED
        u = units.record_finished(req)
        self.assertIsNotNone(u)
        self.assertEqual(u.boundary, 512)


# ---------------------------------------------------------------------------
# Group 4: HYV4Flash validate_runtime and HYV4DFlySpeculator exemptions
# ---------------------------------------------------------------------------
class _StopAfterGuard(Exception):
    pass


class TestHy4ModelAndDFlyExemptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_vllm_stubs()

    def _make_vllm_config(self, connector: str | None, *, active_tier: bool):
        from vllm.merlin import kv_host_tier

        ktc = None
        if connector is not None:
            ktc = types.SimpleNamespace(
                kv_connector=kv_host_tier.CONNECTOR if active_tier else connector,
                kv_connector_module_path=None,
                kv_role="kv_both",
                kv_parallel_size=1,
                kv_buffer_device="cpu",
                kv_connector_extra_config=(
                    kv_host_tier.extra_config(8, 4) if active_tier else {}
                ),
            )
        hf_cfg = types.SimpleNamespace(
            compressed_indexer_compress_ratio=4,
            num_attention_heads=32,
            swa_num_attention_heads=16,
            index_topk=512,
            sliding_window=512,
            engram_layer_ids=[],
            validate_flash_config=lambda: None,
            verify_parallel_config=lambda _pc: None,
        )
        return types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=hf_cfg, hf_text_config=hf_cfg, dtype="bfloat16", quantization="fp8"
            ),
            cache_config=types.SimpleNamespace(block_size=64, cache_dtype="fp8_ds_mla"),
            parallel_config=types.SimpleNamespace(
                world_size=4,
                tensor_parallel_size=2,
                decode_context_parallel_size=2,
                prefill_context_parallel_size=1,
                cp_kv_cache_interleave_size=64,
                use_ubatching=False,
            ),
            lora_config=None,
            kv_transfer_config=ktc,
            speculative_config=types.SimpleNamespace(
                method="dfly", kv_cache_dtype=None, num_speculative_tokens=3
            ),
            engram_config=types.SimpleNamespace(
                cpu_offload=True,
                embedding_across_dp=False,
                dp_shared_memory=False,
                use_thp=False,
            ),
        )

    def test_hy4_flash_validate_runtime_exempts_kv_host_tier(self):
        from vllm.merlin import kv_host_tier
        from vllm.models.hy_v4_flash.nvidia import model as hy4_model

        with (
            mock.patch.object(kv_host_tier, "tier_gib", return_value=8),
            mock.patch.object(hy4_model.current_platform, "is_cuda", return_value=True),
            mock.patch.object(hy4_model, "validate_hy4_dcp"),
            mock.patch.object(hy4_model, "check_dfly_stage"),
        ):
            tier_cfg = self._make_vllm_config("MerlinKVHostTierConnector", active_tier=True)
            pd_cfg = self._make_vllm_config("NixlConnector", active_tier=False)
            hy4_model.validate_runtime(tier_cfg)
            with self.assertRaisesRegex(ValueError, "LoRA or KV transfer"):
                hy4_model.validate_runtime(pd_cfg)

    def test_dfly_speculator_exempts_kv_host_tier(self):
        from vllm.merlin import kv_host_tier
        from vllm.models.hy_v4.nvidia import dfly_speculator as dfly_mod

        def _stop_init(self, vllm_config, device):
            raise _StopAfterGuard

        with (
            mock.patch.object(kv_host_tier, "tier_gib", return_value=8),
            mock.patch.object(dfly_mod.DraftModelSpeculator, "__init__", _stop_init),
        ):
            tier_cfg = self._make_vllm_config("MerlinKVHostTierConnector", active_tier=True)
            pd_cfg = self._make_vllm_config("NixlConnector", active_tier=False)
            with self.assertRaises(_StopAfterGuard):
                dfly_mod.HYV4DFlySpeculator(tier_cfg, device="cpu")
            with self.assertRaisesRegex(ValueError, "PD disaggregation"):
                dfly_mod.HYV4DFlySpeculator(pd_cfg, device="cpu")


if __name__ == "__main__":
    unittest.main()
