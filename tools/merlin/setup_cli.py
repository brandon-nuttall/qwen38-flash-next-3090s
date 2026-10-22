# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The logic behind ``./setup.sh``: host checks, a few questions, the download plan, install, checkpoint, start.

    setup.sh [--yes] [--model M] [--models-dir D] [--host H] [--api-key K] [--port P] [--tp N] [--instances N]
             [--from-source] [--no-start] [--dry-run] [--reconfigure] [--rehash] [--simulate SPEC]
    merlin pull <model> [--models-dir D] [--rehash]        (download + verify only; no GPU needed)

Answers live in ``.merlin/setup.json`` per model (a re-run asks nothing), the API key in ``.merlin/api-key`` (0600),
the install record in ``.merlin/state.json`` (wheel sha + lock hash + commit: a matching venv is reused), and a
verified checkpoint carries ``.merlin-verified.json`` (model, file list, sizes and mtimes: the full re-hash runs when
anything in that list changed, or with ``--rehash``). Standard library only; ``setup.sh`` runs it on the repo-local
Python 3.13 and the venv's ``hf`` does the (resumable) checkpoint download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch as L  # noqa: E402
import preflight as pf  # noqa: E402
import registry as R  # noqa: E402
import serve_cli as S  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODELS = list(L.CHECKPOINTS)  # flash-next, swift1.5, swift1.5-huihui, 27b, 27b-thinkingcap, nemotron-super-vl, ornith1.5
FAMILY = {k: v[0] for k, v in L.CHECKPOINTS.items()}
RECIPE = {k: v[1] for k, v in L.CHECKPOINTS.items()}
FAMILY_DIR = {k: Path(v[1]).name for k, v in L.CHECKPOINTS.items()}
TITLE = {
    "flash-next": "Qwen3.8 Flash-Next W4A16 (the reference recipe)",
    "swift1.5": "Swift 1.5 Flash-Next W4A16 (UkisAI fine-tune)",
    "swift1.5-huihui": "Swift 1.5 Huihui Flash-Next W4A16 (abliterated)",
    "27b": "Qwen3.8 27B W4A16 (dense; 1, 2 or 4 cards)",
    "27b-thinkingcap": "ThinkingCap Qwen3.8 27B W4A16 (BottleCap AI; separate weight licence)",
    "nemotron-super-vl": "Nemotron 3.5 Super VL 120B-A12B W4A16 (4 cards)",
    "ornith1.5": "Ornith 1.5 122B-A10B W4A16 (4 x 24 GB; with or without P2P)",
}
WHEEL_RELEASE_URL = "https://github.com/halt95/merlin-engine/releases/download/merlin-v{version}/{wheel}"
SETUP_JSON = "setup.json"
STATE_JSON = "state.json"
KEY_FILE = "api-key"
VERIFIED_MARKER = ".merlin-verified.json"
MIN_GCC = 13
GCC_CANDIDATES = ("14", "13", "15")  # gcc-N / g++-N looked up on PATH when cc is too old
RAM_OTHER_WARN_GIB = 32.0  # 27B / Super VL / Ornith: no measured host-RAM floor; below this the load is unproven
APT = "sudo apt install"
NOP2P_TP1_WHY = ("P2P is off among these cards: the measured best aggregate without P2P is one single-card server per card "
                 "(measured on 4 x CMP 170HX: TP1 x 4 = 873 tok/s, TP2 x 2 = 523, TP4 = 371), so setup recommends "
                 "--tp 1 --instances {n}; `merlin serve` alone would start one TP1 server on GPU {g}. Each single-card "
                 "server takes images up to 4 MP (the multi-card layouts take 8 MP)")


class SetupError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def gib(n: int | float | None) -> str:
    if n is None:
        return "size unknown"
    if n < 1 << 20:
        return f"{n / 1024:.0f} KB"
    if n < 1 << 30:
        return f"{n / (1 << 20):.0f} MB"
    return f"{n / (1 << 30):.1f} GiB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def release_version(repo: Path = REPO) -> str:
    """The release line of docs/CHANGELOG.md (``## Merlin 3.0.0``) names the GitHub release tag ``merlin-v<version>``."""
    env = os.environ.get("MERLIN_RELEASE_VERSION")
    if env:
        return env
    for line in (repo / "docs" / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## Merlin (\d+\.\d+\.\d+)", line)
        if m:
            return m[1]
    raise SetupError("docs/CHANGELOG.md names no `## Merlin <version>` release; set MERLIN_RELEASE_VERSION")


def wheel_record(repo: Path = REPO) -> tuple[str, str]:
    """(sha256, filename) from release/merlin-wheel.sha256."""
    text = (repo / "release" / "merlin-wheel.sha256").read_text(encoding="utf-8")
    m = re.search(r"(?m)^([0-9a-f]{64})\s+\*?(\S+\.whl)\s*$", text)
    if not m:
        raise SetupError("release/merlin-wheel.sha256 names no wheel")
    return m[1], m[2]


def wheel_urls(repo: Path = REPO) -> list[str]:
    """The release asset's URL and the forms GitHub may give a name with ``+`` (``%2B`` encoded, or ``.`` after an
    upload rename); MERLIN_WHEEL_URL replaces the first. The sha256 check applies to whichever answers."""
    _, wheel = wheel_record(repo)
    url = os.environ.get("MERLIN_WHEEL_URL") or WHEEL_RELEASE_URL.format(version=release_version(repo), wheel=wheel)
    base, _, name = url.rpartition("/")
    out = [url]
    for alt in (name.replace("+", "%2B"), name.replace("+", ".")):
        if alt != name and f"{base}/{alt}" not in out:
            out.append(f"{base}/{alt}")
    return out


def lock_hash(repo: Path = REPO) -> str:
    return sha256_file(repo / "release" / "requirements-0.32.lock")


def sizes_table(repo: Path = REPO) -> dict[str, Any]:
    return json.loads((repo / "release" / "download-sizes.json").read_text(encoding="utf-8"))


def hf_repo(model: str, repo: Path = REPO) -> str:
    return sizes_table(repo)["checkpoints"][model]["repo"]


def checksum_list(model: str, repo: Path = REPO) -> Path:
    return repo / RECIPE[model] / "checkpoint.sha256"


def parse_checksums(path: Path) -> tuple[dict[str, str], str | None]:
    """{filename: sha256} and the Hugging Face revision the header names (None when it names none)."""
    sums: dict[str, str] = {}
    rev = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            m = re.search(r"revision ([0-9a-f]{40})", line)
            if m:
                rev = m[1]
            continue
        m = re.match(r"^([0-9a-f]{64})\s+\*?(.+?)\s*$", line)
        if m:
            sums[m[2]] = m[1]
    return sums, rev


def pending_publication(model: str, repo: Path = REPO, table: dict[str, Any] | None = None) -> bool:
    """True for a checkpoint whose Hugging Face repository is not public yet (release/download-sizes.json
    ``pending_publication``). The marker is only valid with no recorded revision: a pinned revision means it is published."""
    ck = (table or sizes_table(repo))["checkpoints"][model]
    if not ck.get("pending_publication"):
        return False
    if ck.get("revision") or pinned_revision(model, repo, table):
        raise SetupError(f"{model}: release/download-sizes.json marks it pending_publication but a revision is recorded")
    return True


def not_published_msg(model: str) -> str:
    return (f"{model} is not on Hugging Face yet; pass --models-dir with a folder that holds a local copy of the checkpoint "
            f"(the files models/{FAMILY_DIR.get(model, model)}/checkpoint.sha256 lists)")


def pinned_revision(model: str, repo: Path = REPO, table: dict[str, Any] | None = None) -> str | None:
    """The one revision a model is downloaded at: the checksum list's header, else release/download-sizes.json; both
    present and different is an inconsistency in the repository. None for a repository whose revision is not recorded
    yet (pending_publication): the plan then resolves it through the Hugging Face API or refuses."""
    _, rev = parse_checksums(checksum_list(model, repo))
    table_rev = (table or sizes_table(repo))["checkpoints"][model].get("revision")
    if rev and table_rev and rev != table_rev:
        raise SetupError(f"{model}: checkpoint.sha256 names revision {rev} but release/download-sizes.json {table_rev}")
    return rev or table_rev


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def write_json(path: Path, data: dict[str, Any], mode: int = 0o644) -> None:
    write_private(path, json.dumps(data, indent=1, sort_keys=True) + "\n", mode)


def write_private(path: Path, text: str, mode: int = 0o600) -> None:
    """Create the file with ``mode`` from the first byte (no world-readable window) under a 0700 .merlin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.name == ".merlin":
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        os.chmod(tmp, mode)  # the creation mode is filtered by the umask; the file is ours, so set it exactly
    except OSError:
        pass
    os.replace(tmp, path)


def which(name: str, env: dict[str, str] | None = None) -> str | None:
    return shutil.which(name, path=(env or os.environ).get("PATH"))


def tool_major(tool: str, run: Callable[[list[str]], str | None]) -> int | None:
    out = run([tool, "-dumpversion"])
    if not out:
        return None
    head = out.strip().split(".")[0]
    return int(head) if head.isdigit() else None


def default_runner(cmd: list[str]) -> str | None:
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return cp.stdout if cp.returncode == 0 else None


def nearest_existing(path: Path) -> Path:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def disk_free(path: Path) -> int | None:
    """Free bytes on the filesystem that holds ``path`` (its nearest existing parent)."""
    try:
        return shutil.disk_usage(nearest_existing(path)).free
    except OSError:
        return None


def device_of(path: Path) -> Any:
    try:
        return os.stat(nearest_existing(path)).st_dev
    except OSError:
        return str(path)


def uv_cache_dir() -> Path:
    env = os.environ.get("UV_CACHE_DIR")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "uv"


# --------------------------------------------------------------------------------------------
# Arguments and saved answers (per model)
# --------------------------------------------------------------------------------------------
PER_MODEL = ("models_dir", "port", "tp", "instances")


@dataclass
class Answers:
    model: str | None = None
    models_dir: str | None = None
    host: str = "127.0.0.1"
    api_key: str | None = None  # never saved in setup.json: .merlin/api-key (0600)
    port: int | None = None
    tp: int | None = None
    instances: int | None = None
    from_source: bool = False
    cc: str | None = None
    cxx: str | None = None

    def checkpoint_dir(self) -> Path:
        return Path(self.models_dir or f"~/models/{self.model}").expanduser()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="setup.sh", description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every option and the update / uninstall flow: docs/SETUP.md",
    )
    ap.add_argument("--yes", "-y", action="store_true", help="no questions: the recommended (or given) answers, downloads confirmed")
    ap.add_argument("--model", choices=MODELS, help="checkpoint to install and serve")
    ap.add_argument("--models-dir", metavar="DIR", help="checkpoint folder (default ~/models/<model>; an existing verified copy is reused)")
    ap.add_argument("--host", help="127.0.0.1 (this machine only, default) or 0.0.0.0 (network: needs --api-key or one is generated)")
    ap.add_argument("--api-key", metavar="KEY", help="bearer key the server requires (stored in .merlin/api-key, 0600)")
    ap.add_argument("--port", type=int, help="HTTP port (default: the model's, 8000 Flash-Next / Super VL, 8100 27B)")
    ap.add_argument("--tp", type=int, help="pick the layout with this tensor-parallel size")
    ap.add_argument("--instances", type=int, help="start N servers on N disjoint card sets")
    ap.add_argument("--from-source", action="store_true", help="compile the engine instead of taking the release wheel")
    ap.add_argument("--no-start", action="store_true", help="install and download, then stop before `merlin serve`")
    ap.add_argument("--dry-run", action="store_true", help="print checks, answers, download plan and serve command; download and write nothing")
    ap.add_argument("--reconfigure", action="store_true", help="ask the questions again (saved answers are the defaults)")
    ap.add_argument("--rehash", action="store_true", help="hash every checkpoint file again even when the verified marker is current")
    ap.add_argument("--simulate", metavar="SPEC", help="preflight's hardware spec (4x3090, 2x3090:ram=64, ...): implies --dry-run unless --no-start")
    ap.add_argument("--pull-only", action="store_true", help=argparse.SUPPRESS)  # merlin pull
    return ap


def parse_args(argv: list[str]) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.simulate and not args.no_start:  # nothing ever starts on fabricated hardware; --no-start still installs
        args.dry_run = True
    if args.host and args.host not in ("127.0.0.1", "0.0.0.0", "localhost", "::1"):
        if not re.fullmatch(r"[0-9.]+|[0-9a-fA-F:]+", args.host):
            raise SetupError(f"--host {args.host!r}: give an IP address (127.0.0.1 or 0.0.0.0)", 2)
    if args.port is not None and not 1 <= args.port <= 65535:
        raise SetupError(f"--port {args.port}: 1..65535", 2)
    if args.tp is not None and args.tp < 1 or args.instances is not None and args.instances < 1:
        raise SetupError("--tp and --instances take whole numbers >= 1", 2)
    return args


def _valid_saved(saved: dict[str, Any] | None) -> dict[str, Any]:
    """The saved file with anything malformed dropped (a bad model name, port or block never raises later)."""
    if not saved:
        return {"models": {}}
    out: dict[str, Any] = {}
    if saved.get("model") in MODELS:
        out["model"] = saved["model"]
    if saved.get("host") in ("127.0.0.1", "0.0.0.0", "localhost", "::1"):
        out["host"] = saved["host"]
    if isinstance(saved.get("from_source"), bool):
        out["from_source"] = saved["from_source"]
    for k in ("cc", "cxx"):
        if isinstance(saved.get(k), str) and saved[k]:
            out[k] = saved[k]
    models = saved.get("models")
    out["models"] = {}
    if isinstance(models, dict):
        for m, block in models.items():
            if m in MODELS and isinstance(block, dict):
                b: dict[str, Any] = {}
                if isinstance(block.get("models_dir"), str) and block["models_dir"]:
                    b["models_dir"] = block["models_dir"]
                for k in ("port", "tp", "instances"):
                    v = block.get(k)
                    if isinstance(v, int) and not isinstance(v, bool) and v >= 1 and (k != "port" or v <= 65535):
                        b[k] = v
                out["models"][m] = b
    return out


def merge_answers(saved: dict[str, Any] | None, args: argparse.Namespace, key: str | None = None) -> Answers:
    """Saved answers for the chosen model (the saved model unless --model names another), then every flag given on the
    command line. The folder, port and layout belong to a model: another model starts from its own defaults."""
    s = _valid_saved(saved)
    model = args.model or s.get("model")
    block = s["models"].get(model, {}) if model else {}
    a = Answers(model=model, host=s.get("host", "127.0.0.1"), from_source=s.get("from_source", False), cc=s.get("cc"), cxx=s.get("cxx"))
    a.api_key = key
    for k in PER_MODEL:
        setattr(a, k, block.get(k))
    for k in ("models_dir", "host", "api_key", "port", "tp", "instances"):
        v = getattr(args, k)
        if v is not None:
            setattr(a, k, v)
    if args.from_source:
        a.from_source = True
    return a


def saved_block(a: Answers) -> dict[str, Any]:
    return {k: getattr(a, k) for k in PER_MODEL}


def save_answers(path: Path, saved: dict[str, Any] | None, a: Answers) -> dict[str, Any]:
    s = _valid_saved(saved)
    s.update({"model": a.model, "host": a.host, "from_source": a.from_source})
    for k in ("cc", "cxx"):
        if getattr(a, k):
            s[k] = getattr(a, k)
        else:
            s.pop(k, None)
    s["models"][a.model] = saved_block(a)
    write_json(path, s, 0o600)
    return s


def answers_complete(a: Answers, saved: dict[str, Any] | None) -> bool:
    """A saved set exists for this model (folder, port, layout) and network access has its key."""
    s = _valid_saved(saved)
    return bool(a.model) and a.model in s["models"] and a.port is not None and (a.host == "127.0.0.1" or bool(a.api_key))


def read_key(path: Path) -> str | None:
    try:
        key = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


# --------------------------------------------------------------------------------------------
# Host checks
# --------------------------------------------------------------------------------------------
@dataclass
class Problem:
    level: str  # FAIL or WARN
    code: str
    text: str
    fix: str = ""


def find_gcc(run: Callable[[list[str]], str | None], env: dict[str, str] | None = None) -> tuple[str, str] | None:
    """A ``gcc-N`` / ``g++-N`` pair (N >= 13) on PATH, newest first."""
    for n in sorted(GCC_CANDIDATES, reverse=True):
        cc, cxx = f"gcc-{n}", f"g++-{n}"
        if which(cc, env) and which(cxx, env) and (tool_major(cc, run) or 0) >= MIN_GCC and (tool_major(cxx, run) or 0) >= MIN_GCC:
            return cc, cxx
    return None


def gcc_problem(run: Callable[[list[str]], str | None], env: dict[str, str] | None = None,
                a: Answers | None = None) -> Problem | None:
    """GCC/G++ >= 13 for the install (release/install-env.sh checks CC / CXX, default cc / c++). The pair comes from
    CC / CXX in the environment, else the saved answers, else cc / c++; when those are too old a ``gcc-N`` / ``g++-N``
    pair on PATH is taken and remembered in the answers (so one fix lasts across re-runs)."""
    env = env or dict(os.environ)
    cc = env.get("CC") or (a.cc if a else None) or "cc"
    cxx = env.get("CXX") or (a.cxx if a else None) or "c++"
    fix = (f"{APT} gcc-{MIN_GCC} g++-{MIN_GCC}   (Debian 12 / Ubuntu 24.04 ship GCC 13 or 14; on Ubuntu 22.04 first "
           f"`sudo add-apt-repository ppa:ubuntu-toolchain-r/test`), then run ./setup.sh again (it finds gcc-{MIN_GCC} on PATH "
           f"and remembers it; CC=gcc-{MIN_GCC} CXX=g++-{MIN_GCC} ./setup.sh names another pair)")

    def problem(pair: tuple[str, str]) -> Problem | None:
        for t in pair:
            if which(t, env) is None:
                return Problem("FAIL", "compiler", f"{t} is missing (the install needs a C/C++ compiler >= GCC {MIN_GCC})", fix)
            major = tool_major(t, run)
            if major is None:
                return Problem("FAIL", "compiler", f"cannot read the version of {t} (GCC/G++ >= {MIN_GCC} required)", fix)
            if major < MIN_GCC:
                return Problem("FAIL", "compiler", f"{t} is GCC {major}; the engine's CUDA ops need GCC/G++ >= {MIN_GCC}", fix)
        return None

    p = problem((cc, cxx))
    if p is None:
        if a is not None and (env.get("CC") or env.get("CXX")):
            a.cc, a.cxx = cc, cxx  # an explicit pair is remembered
        return None
    if not (env.get("CC") or env.get("CXX")):
        found = find_gcc(run, env)
        if found:
            if a is not None:
                a.cc, a.cxx = found
            return None
    return p


def compiler_env(a: Answers, env: dict[str, str]) -> dict[str, str]:
    env = dict(env)
    if a.cc and not env.get("CC"):
        env["CC"] = a.cc
    if a.cxx and not env.get("CXX"):
        env["CXX"] = a.cxx
    return env


def git_problem(repo: Path, run: Callable[[list[str]], str | None], env: dict[str, str] | None = None) -> Problem | None:
    if which("git", env) is None:
        return Problem("FAIL", "git", "git is missing", f"{APT} git")
    dirty = run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"])
    if dirty is None:
        return Problem("FAIL", "git", f"{repo} is not a git checkout (a ZIP download?)",
                       "clone the repository with git (git clone ...) and run ./setup.sh inside it")
    if dirty.strip():
        return Problem("FAIL", "git", "tracked files are modified; the installer records a clean commit as the engine's source",
                       f"commit or revert them (cd {repo} && git status)")
    return None


def driver_problem(inv: pf.Inventory) -> Problem | None:
    fix = ("install the NVIDIA driver R580 or later (CUDA 13): Ubuntu `sudo apt install nvidia-driver-580-open` (or the "
           ".run installer from https://www.nvidia.com/drivers), reboot, run ./setup.sh again")
    if not inv.gpus:
        return Problem("FAIL", "gpu", "no NVIDIA GPU is visible (nvidia-smi is missing or lists none)", fix)
    d = inv.host.driver
    major = int(d.split(".")[0]) if d and d.split(".")[0].isdigit() else None
    if major is not None and major < pf.DRIVER_MIN_MAJOR:
        return Problem("FAIL", "driver", f"driver {d} is older than the {pf.DRIVER_MIN_MAJOR} series: CUDA 13 needs R580 or later", fix)
    return None


def source_build_problems(run: Callable[[list[str]], str | None], env: dict[str, str] | None = None) -> list[Problem]:
    out = []
    if which("cargo", env) is None:
        out.append(Problem("FAIL", "cargo", "--from-source needs cargo (Rust)", "curl https://sh.rustup.rs -sSf | sh   (per user, no sudo)"))
    if which("cmake", env) is None or which("ninja", env) is None:
        out.append(Problem("FAIL", "cmake", "--from-source needs cmake and ninja", f"{APT} cmake ninja-build"))
    return out


def ram_problem(model: str, inv: pf.Inventory, cards: int) -> Problem | None:
    ram = inv.host.ram_mib
    if ram is None:
        return Problem("WARN", "ram", "host RAM unknown")
    have = pf._gib(ram)
    if FAMILY[model] == "flash-next":
        floor = pf.ram_floor_gib(cards)
        if have < floor:
            return Problem("FAIL", "ram", f"host RAM {have:.0f} GiB is below Flash-Next's ~{floor:.0f} GiB resident floor "
                           "(a ~47.7 GiB pinned PLE table): the load would be OOM-killed",
                           "add RAM, or choose 27b (--model 27b), which needs no PLE table")
        if have < pf.RAM_WARN_GIB:
            return Problem("WARN", "ram", f"host RAM {have:.0f} GiB is below the qualified {pf.RAM_WARN_GIB:.0f} GiB "
                           f"(floor ~{floor:.0f} GiB; the boot peak is unmeasured)")
    elif have < RAM_OTHER_WARN_GIB:
        return Problem("WARN", "ram", f"host RAM {have:.0f} GiB: {model} is unmeasured below {RAM_OTHER_WARN_GIB:.0f} GiB")
    return None


# --------------------------------------------------------------------------------------------
# Recommendation and layouts
# --------------------------------------------------------------------------------------------
def _choose(model: str, inv: pf.Inventory, tp: int | None = None) -> S.Choice | None:
    try:
        return S.choose(S._load(model), model, inv, None, tp)
    except (S.CliError, R.ProfileError):
        return None


THINKINGCAP_WHY = ("the fastest 27B-family checkpoint on this hardware (2026-10-20 bench cards, 1 stream, thinking on, 4K / 131K / "
                   "262K: ThinkingCap TP4 253 / 255 / 248 tok/s vs 27B TP4 250 / 235 / 215); separate weight licence, see "
                   "models/thinkingcap/licenses/")
THINKINGCAP_LICENCE = ("ThinkingCap's weights are BottleCap AI's, under the PolyForm Small Business 1.0.0 licence plus their personal-use "
                       "permission (not Apache-2.0): models/thinkingcap/licenses/ and the Hugging Face card state the terms; "
                       "downloading them means accepting those terms.")


def thinkingcap_tp4(inv: pf.Inventory) -> S.Choice | None:
    """The ThinkingCap TP4 pick when its 4 x 24 GB 3090-class row fits this host with P2P (the 4 x 3090
    27B-family default); None elsewhere (2 cards, no P2P, other classes)."""
    tc = _choose("27b-thinkingcap", inv)
    if tc is None:
        return None
    m = tc.prof["match"]
    if m["cards"] == 4 and m["arch"] == ["sm_86"] and m["vram_gib"] == 24 and _profile_tp(tc.prof) == 4 and tc.rep.effective_p2p == "on":
        return tc
    return None


def recommend(inv: pf.Inventory) -> tuple[str | None, str]:
    """(model, why): Flash-Next when a Flash-Next layout fits the cards and the host has its qualified RAM; else 27B
    when a 27B layout fits; else Flash-Next under its RAM floor (an OOM is expected); else the first model with any
    layout; else None."""
    names = sorted({g.name.replace("NVIDIA ", "") for g in inv.gpus})
    cards = f"{len(inv.gpus)} x {' + '.join(names)}" if inv.gpus else "no GPU"
    ram = inv.host.ram_mib
    ram_s = f"{pf._gib(ram):.0f} GiB RAM" if ram is not None else "RAM unknown"
    # the card fit alone: preflight refuses Flash-Next under the RAM floor, which is the second question here
    fn = _choose("flash-next", replace(inv, host=replace(inv.host, ram_mib=int(pf.RAM_WARN_GIB * 1024))))
    if fn is not None and (ram is None or pf._gib(ram) >= pf.RAM_WARN_GIB):
        return "flash-next", (f"{cards}, P2P {fn.rep.effective_p2p}, {ram_s}: {fn.prof['id']} [{fn.rep.status}] fits"
                              + (" with the qualified RAM" if ram is not None else f" (check for >= {pf.RAM_WARN_GIB:.0f} GiB RAM)"))
    b27 = _choose("27b", inv)
    if b27 is not None:
        if fn is not None:
            why = f"{cards} fit Flash-Next ({fn.prof['id']}) but {ram_s} is below its qualified {pf.RAM_WARN_GIB:.0f} GiB"
        else:
            why = f"{cards} hold no Flash-Next layout (it needs 4 x 24 GB or 2 x 48 GB)"
        tc = thinkingcap_tp4(inv)
        if tc is not None:  # the 4 x 3090 27B-family default: ThinkingCap TP4 with P2P
            return "27b-thinkingcap", f"{why}; ThinkingCap {tc.prof['id']} [{tc.rep.status}] fits, {ram_s}: {THINKINGCAP_WHY}"
        return "27b", f"{why}; 27B {b27.prof['id']} [{b27.rep.status}] fits, {ram_s}"
    if fn is not None:
        return "flash-next", (f"{cards} fit {fn.prof['id']} but {ram_s} is below the ~{pf.ram_floor_gib(len(fn.cards)):.0f} GiB "
                              "floor: expect an OOM; 27B does not fit either")
    for m in MODELS:
        if _choose(m, inv) is not None:
            return m, f"{cards}: only {m} has a layout here"
    return None, f"{cards}, {ram_s}: no Merlin layout fits (run `merlin doctor` for the report)"


@dataclass
class Layout:
    label: str  # "default", "--tp 2", "--instances 2", "--tp 1 --instances 4"
    tp: int | None
    instances: int | None
    text: str
    recommended: bool = False
    why: str = ""


def _profile_tp(p: dict[str, Any]) -> int | None:
    return (p.get("layout") or {}).get("tp", p["serve"].get("tensor_parallel"))


def layouts(model: str, inv: pf.Inventory, repo: Path = REPO) -> list[Layout]:
    """What this host offers for ``model``: serve's own pick first, then every other tensor-parallel size whose row
    resolves (``--tp`` takes the row's real TP, not its card count), then every ``--instances`` count that partitions.
    One option carries ``recommended``: serve's pick, except a single-card 27B pick on a multi-card class without P2P,
    where setup recommends ``--tp 1 --instances N`` over all idle cards (the measured aggregate advice; serve's own
    default is unchanged)."""
    out: list[Layout] = []
    seen: set[str] = set()
    try:
        d = S.resolve(model, None, None, inv, [], repo, {})
    except S.CliError:
        return out
    out.append(Layout("default", None, None, f"{d.profile_id} [{d.status}] on GPUs {','.join(map(str, d.cards))}", True,
                      "what merlin serve picks"))
    seen.add(d.profile_id)
    reg = S._load(model)
    tps = sorted({t for t in (_profile_tp(p) for p in S._profiles_for(model, reg)) if t})
    for tp in tps:
        try:
            p = S.resolve(model, None, None, inv, [], repo, {}, tp=tp)
        except S.CliError:
            continue
        if p.profile_id in seen:
            continue
        seen.add(p.profile_id)
        out.append(Layout(f"--tp {tp}", tp, None, f"{p.profile_id} [{p.status}] on GPUs {','.join(map(str, p.cards))}"))
    for n in range(2, len(inv.gpus) + 1):
        try:
            plans = S.plan_instances(model, None, None, inv, [], n, repo, {})
        except S.CliError:
            continue
        sets = " / ".join(",".join(map(str, p.cards)) for p in plans)
        ports = f"ports {plans[0].env.get('PORT')}-{plans[-1].env.get('PORT')}"
        tp1 = all(len(p.cards) == 1 for p in plans)
        out.append(Layout(f"--tp 1 --instances {n}" if tp1 else f"--instances {n}", 1 if tp1 else None, n,
                          f"{n} x {plans[0].profile_id} on GPUs {sets}, {ports}"))
    # the no-P2P single-card case: all idle cards of the class, each its own server
    if FAMILY[model] == "27b" and len(d.cards) == 1:
        first = next(g for g in inv.gpus if g.index == d.cards[0])
        cls = [g for g in inv.gpus if S._class_of(g) == S._class_of(first)]
        idle = S._idle(cls)
        if len(idle) > 1 and inv.p2p_among([g.index for g in idle]) != "on":
            pick = next((o for o in out if o.instances == len(idle) and o.tp == 1), None)
            if pick is not None:
                out[0].recommended, out[0].why = False, ""
                pick.recommended, pick.why = True, NOP2P_TP1_WHY.format(n=len(idle), g=d.cards[0])
    return out


def serve_plan(a: Answers, inv: pf.Inventory, repo: Path = REPO) -> list[S.Plan]:
    env = serve_env(a)
    if a.instances and a.instances > 1:
        return S.plan_instances(a.model, str(a.checkpoint_dir()), None, inv, [], a.instances, repo, env, tp=a.tp)
    return [S.resolve(a.model, str(a.checkpoint_dir()), None, inv, [], repo, env, tp=a.tp)]


def serve_env(a: Answers) -> dict[str, str]:
    env = compiler_env(a, dict(os.environ))
    env["HOST"] = a.host
    if a.port is not None:
        env["PORT"] = str(a.port)
    if a.api_key:
        env["VLLM_API_KEY"] = a.api_key
    env["MERLIN_HOME"] = str(REPO)
    return env


def serve_argv(a: Answers, repo: Path = REPO) -> list[str]:
    argv = [str(repo / "venv" / "bin" / "python"), str(repo / "merlin_cli.py"), "serve", a.model or "", str(a.checkpoint_dir())]
    if a.tp:
        argv += ["--tp", str(a.tp)]
    if a.instances and a.instances > 1:
        argv += ["--instances", str(a.instances)]
    return argv


def serve_line(a: Answers, repo: Path = REPO) -> str:
    pre = f"HOST={a.host} PORT={a.port}" + (" VLLM_API_KEY=\"$(cat .merlin/api-key)\"" if a.api_key else "")
    return pre + " " + " ".join(shlex.quote(x) for x in serve_argv(a, repo))


def default_port(model: str) -> int:
    return S._default_port(S._load(model))


def served_name(model: str) -> str:
    reg = S._load(model)
    if FAMILY[model] == "flash-next":
        ck = reg.get("checkpoints", {}).get(model, {})
        return (ck.get("served_names") or reg["base"]["served_names"]).split()[0]
    if FAMILY[model] in ("svl", "ornith"):
        return reg["launch"]["served"]
    return reg["launch"]["checkpoints"][model]["served"]


# --------------------------------------------------------------------------------------------
# Install state and the verified-checkpoint marker
# --------------------------------------------------------------------------------------------
def head_commit(repo: Path, run: Callable[[list[str]], str | None] = default_runner) -> str | None:
    out = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    return out.strip() if out else None


def install_state(repo: Path, from_source: bool, commit: str | None) -> dict[str, Any]:
    sha, name = wheel_record(repo)
    return {"wheel_sha": "source" if from_source else sha, "wheel": name, "lock_sha": lock_hash(repo), "commit": commit,
            "venv": str(repo / "venv")}


def venv_current(repo: Path, want: dict[str, Any], saved: dict[str, Any] | None) -> bool:
    """The recorded install matches this checkout (wheel sha, lock hash, commit) and the venv is still there."""
    if not saved:
        return False
    if any(saved.get(k) != want[k] for k in ("wheel_sha", "lock_sha", "commit", "venv")):
        return False
    venv = Path(want["venv"])
    return (venv / "bin" / "vllm").exists() and (venv / "bin" / "python").exists()


def marker_data(ckpt_dir: Path, model: str, sums: dict[str, str], list_sha: str) -> dict[str, Any]:
    files = {}
    for name in sorted(sums):
        st = (ckpt_dir / name).stat()
        files[name] = [st.st_size, st.st_mtime_ns]
    return {"model": model, "list_sha256": list_sha, "files": files, "verified_at": int(time.time())}


def marker_ok(ckpt_dir: Path, model: str, sums: dict[str, str], list_sha: str) -> bool:
    """True when the marker records this model and checksum list and every file still has the recorded size and
    mtime. A trade-off for speed (a 115 GiB re-hash takes minutes): same-size, same-mtime corruption passes until
    ``--rehash``; a changed list, file set, size or mtime re-hashes."""
    m = read_json(ckpt_dir / VERIFIED_MARKER)
    if not m or m.get("model") != model or m.get("list_sha256") != list_sha or set(m.get("files", {})) != set(sums):
        return False
    for name, (size, mtime) in m["files"].items():
        try:
            st = (ckpt_dir / name).stat()
        except OSError:
            return False
        if st.st_size != size or st.st_mtime_ns != mtime:
            return False
    return True


def verify_checkpoint(ckpt_dir: Path, list_path: Path, say: Callable[[str], None] = print, model: str = "",
                      rehash: bool = False) -> tuple[bool, list[str]]:
    """Hash every listed file (skipped when the marker is current, unless ``rehash``). (ok, missing-or-wrong files)."""
    sums, _ = parse_checksums(list_path)
    list_sha = sha256_file(list_path)
    if not rehash and marker_ok(ckpt_dir, model, sums, list_sha):
        say(f"checkpoint {ckpt_dir}: verified earlier ({len(sums)} files unchanged; --rehash hashes them again)")
        return True, []
    bad = []
    for i, (name, want) in enumerate(sorted(sums.items()), 1):
        p = ckpt_dir / name
        if not p.is_file():
            bad.append(f"{name}: missing")
            continue
        say(f"  hashing {i}/{len(sums)} {name} ({gib(p.stat().st_size)})")
        if sha256_file(p) != want:
            bad.append(f"{name}: sha256 differs")
    if bad:
        try:
            (ckpt_dir / VERIFIED_MARKER).unlink()
        except OSError:
            pass
        return False, bad
    write_json(ckpt_dir / VERIFIED_MARKER, marker_data(ckpt_dir, model, sums, list_sha))
    return True, []


def checkpoint_missing(ckpt_dir: Path, list_path: Path) -> list[str]:
    sums, _ = parse_checksums(list_path)
    return [n for n in sums if not (ckpt_dir / n).is_file()]


def folder_conflict(ckpt_dir: Path, model: str, repo: Path = REPO) -> str | None:
    """Why ``ckpt_dir`` must not receive ``model``: it holds another checkpoint (its config.json or verified marker
    belongs to another model, or to no known revision). None when the folder is empty, missing, or this model's."""
    if not ckpt_dir.is_dir():
        return None
    m = read_json(ckpt_dir / VERIFIED_MARKER)
    if m and m.get("model") and m["model"] != model:
        return f"holds a verified {m['model']} checkpoint"
    cfg = ckpt_dir / "config.json"
    if cfg.is_file():
        digest = sha256_file(cfg)
        owners = [o for o in MODELS if parse_checksums(checksum_list(o, repo))[0].get("config.json") == digest]
        if model in owners:
            return None
        if owners:
            return f"holds a {' / '.join(owners)} checkpoint (its config.json)"
        return "holds a checkpoint that is not this release's revision of any Merlin model (its config.json matches none)"
    return None


# --------------------------------------------------------------------------------------------
# Download plan and disk
# --------------------------------------------------------------------------------------------
@dataclass
class Download:
    what: str
    source: str
    bytes: int | None  # None = unknown, 0 = nothing to fetch
    where: str  # "repo" or "models"
    note: str = ""


@dataclass
class DiskNeed:
    path: Path
    bytes: int
    what: str


@dataclass
class DownloadPlan:
    items: list[Download] = field(default_factory=list)
    need_install: bool = False
    need_checkpoint: bool = False
    revision: str | None = None  # the checkpoint revision hf download is pinned to
    disk: list[DiskNeed] = field(default_factory=list)

    def known_total(self) -> int:
        return sum(d.bytes or 0 for d in self.items)

    def unknown(self) -> int:
        return sum(1 for d in self.items if d.bytes is None)

    def pending(self) -> bool:
        return any(d.bytes is None or d.bytes > 0 for d in self.items)


@dataclass
class HfInfo:
    bytes: int
    files: int
    sha: str | None
    sizes: dict[str, int]


def hf_info(repo_id: str, revision: str | None, fetch: Callable[[str], bytes] | None = None, timeout: float = 10) -> HfInfo | None:
    """Sizes, file count and resolved commit of a Hugging Face repository (its API); None when unreachable or gated
    without HF_TOKEN."""
    url = f"https://huggingface.co/api/models/{repo_id}" + (f"/revision/{revision}" if revision else "") + "?blobs=true"
    try:
        if fetch is None:
            req = urllib.request.Request(url, headers={"User-Agent": "merlin-setup"})
            tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if tok:
                req.add_header("Authorization", f"Bearer {tok}")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
        else:
            raw = fetch(url)
        d = json.loads(raw)
        sib = d.get("siblings") or []
        sizes = {s["rfilename"]: int((s.get("lfs") or {}).get("size") or s.get("size") or 0) for s in sib}
        total = sum(sizes.values())
        return HfInfo(total, len(sib), d.get("sha"), sizes) if total else None
    except Exception:  # noqa: BLE001 - any network / JSON failure means "use the table"
        return None


def checkpoint_item(model: str, ckpt_dir: Path, repo: Path, table: dict[str, Any], fetch: Callable[[str], bytes] | None,
                    online: bool) -> tuple[Download, bool, str | None, int | None]:
    """The checkpoint's plan line: (item, needs download, pinned revision, remaining bytes or None when unknown)."""
    ck = table["checkpoints"][model]
    list_path = checksum_list(model, repo)
    sums, _ = parse_checksums(list_path)
    missing = [n for n in sums if not (ckpt_dir / n).is_file()] if ckpt_dir.is_dir() else list(sums)
    rev = pinned_revision(model, repo, table)
    if not missing:
        return Download(f"checkpoint {model}", str(ckpt_dir), 0, "models", "present; verified before serving"), False, rev, 0
    if pending_publication(model, repo, table):
        return (Download(f"checkpoint {model}", f"local copy only -> {ckpt_dir}", None, "models", not_published_msg(model)),
                True, None, None)
    info = hf_info(ck["repo"], rev, fetch) if online else None
    if rev is None and info is not None and info.sha:
        rev = info.sha  # a repository without a recorded revision: pinned to the commit the API resolves now
    if info is not None:
        remaining: int | None = sum(info.sizes.get(n, 0) for n in missing)  # the listed files still to fetch
        n_bytes, n_files, src = info.bytes, info.files, "Hugging Face API"
    elif ck.get("bytes"):
        present = sum((ckpt_dir / n).stat().st_size for n in sums if (ckpt_dir / n).is_file()) if ckpt_dir.is_dir() else 0
        remaining = max(0, ck["bytes"] - present)
        n_bytes, n_files, src = ck["bytes"], ck.get("files"), "release/download-sizes.json"
    else:
        remaining, n_bytes, n_files, src = None, None, ck.get("files"), None
    if n_bytes is None:
        note = "size unknown (the Hugging Face API did not answer)"
    else:
        note = f"{n_files} files, {gib(n_bytes)} in all, size from {src}"
    if len(missing) < len(sums):
        note += f"; {len(missing)} of {len(sums)} listed files still missing in {ckpt_dir} (the download resumes)"
    cmd = f"hf download {ck['repo']}" + (f" --revision {rev}" if rev else " (no pinned revision yet)") + f" -> {ckpt_dir}"
    return Download(f"checkpoint {model}", cmd, remaining, "models", note), True, rev, remaining


def download_plan(a: Answers, repo: Path, state: dict[str, Any] | None, commit: str | None,
                  fetch: Callable[[str], bytes] | None = None, online: bool = True,
                  table: dict[str, Any] | None = None, engine: bool = True) -> DownloadPlan:
    """What a run fetches and the disk it needs (per path; ``disk_problems`` groups by filesystem). ``engine=False``
    plans the checkpoint only (merlin pull)."""
    table = table or sizes_table(repo)
    plan = DownloadPlan()
    sha, wheel = wheel_record(repo)
    venv = repo / "venv"
    factor = table.get("venv_factor", 3)
    pk = table["packages"]
    if engine and not venv_current(repo, install_state(repo, a.from_source, commit), state):
        plan.need_install = True
        if a.from_source:
            plan.items.append(Download("engine source build", "compiles csrc/ (about 1.5 h on 8 cores)", 0, "repo"))
        else:
            local = os.environ.get("MERLIN_WHEEL")
            cached = repo / ".merlin" / "downloads" / wheel
            if local and Path(local).is_file():
                plan.items.append(Download("engine wheel", f"MERLIN_WHEEL={local}", 0, "repo", "local file, no download"))
            elif cached.is_file() and sha256_file(cached) == sha:
                plan.items.append(Download("engine wheel", str(cached), 0, "repo", "already downloaded"))
            else:
                plan.items.append(Download("engine wheel", wheel_urls(repo)[0], table["wheel"]["bytes"], "repo"))
                plan.disk.append(DiskNeed(repo / ".merlin" / "downloads", table["wheel"]["bytes"], "the wheel download"))
        plan.items.append(Download(f"Python packages ({pk['count']} from release/requirements-0.32.lock)", "PyPI via uv (cached after the first install)",
                                   pk["bytes"], "repo", f"about {gib(pk['bytes'] * factor)} installed"))
        plan.disk.append(DiskNeed(uv_cache_dir(), pk["bytes"], "uv's package cache (up to; less when it already holds them)"))
        plan.disk.append(DiskNeed(venv, pk["bytes"] * factor, "the new venv"))
        if venv.exists():
            plan.disk.append(DiskNeed(venv, pk["bytes"] * factor, "the previous venv, kept until the new one verifies"))
    item, need, rev, remaining = checkpoint_item(a.model, a.checkpoint_dir(), repo, table, fetch, online)
    plan.items.append(item)
    plan.need_checkpoint, plan.revision = need, rev
    if need and remaining:
        plan.disk.append(DiskNeed(a.checkpoint_dir(), remaining, f"the rest of the {a.model} checkpoint"))
    return plan


def disk_problems(plan: DownloadPlan, free: Callable[[Path], int | None] | None = None,
                  dev: Callable[[Path], Any] | None = None) -> list[Problem]:
    """One FAIL per filesystem whose free space is below everything the plan puts there."""
    free, dev = free or disk_free, dev or device_of  # looked up at call time (tests replace them)
    groups: dict[Any, list[DiskNeed]] = {}
    for n in plan.disk:
        groups.setdefault(dev(n.path), []).append(n)
    out = []
    for needs in groups.values():
        total = sum(n.bytes for n in needs)
        have = free(needs[0].path)
        if have is not None and have < total:
            parts = ", ".join(f"{n.what} {gib(n.bytes)}" for n in needs)
            where = sorted({str(nearest_existing(n.path)) for n in needs})
            out.append(Problem("FAIL", "disk", f"the filesystem of {' and '.join(where)} has {gib(have)} free; the plan puts {gib(total)} there ({parts})",
                               "free the space" + (" or choose another models folder (--models-dir)" if any("checkpoint" in n.what for n in needs) else "")))
    return out


def render_plan(plan: DownloadPlan) -> str:
    if not plan.pending():
        return "Nothing to download: the engine and the checkpoint are in place.\n"
    lines = ["Downloads:"]
    for d in plan.items:
        size = gib(d.bytes) if d.bytes else ("-" if d.bytes == 0 else "size unknown")
        lines.append(f"  {d.what:<58} {size:>12}")
        lines.append(f"      {d.source}" + (f"  ({d.note})" if d.note else ""))
    total = gib(plan.known_total()) + (f" + {plan.unknown()} of unknown size" if plan.unknown() else "")
    lines.append(f"  {'total':<58} {total:>12}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------------
def sh(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> int:
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.call(cmd, env=env, cwd=cwd)


def fetch_wheel(urls: list[str], dest: Path, sha: str, runner: Callable[[list[str]], int] = sh) -> Path:
    """The first URL that answers, resumed into ``dest.part``, checked against ``sha`` (a mismatch is deleted)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and sha256_file(dest) == sha:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    rc = 1
    for url in urls:
        rc = runner(["curl", "-fL", "--retry", "3", "-C", "-", "-o", str(part), url])
        if rc == 0:
            break
        print(f"setup: {url}: not available (rc {rc})" + ("; trying the next name" if url != urls[-1] else ""), flush=True)
    if rc != 0:
        raise SetupError("wheel download failed from every URL:\n  " + "\n  ".join(urls)
                         + "\n  MERLIN_WHEEL=<file> or MERLIN_WHEEL_URL=<url> override it")
    got = sha256_file(part)
    if got != sha:
        part.unlink()
        raise SetupError(f"downloaded wheel sha256 {got} differs from release/merlin-wheel.sha256 ({sha}); deleted, not installed")
    os.replace(part, dest)
    return dest


def install_engine(repo: Path, a: Answers, env: dict[str, str], runner: Callable[..., int] = sh) -> None:
    """install-env.sh into ``venv`` then ``--verify``. An existing venv is moved aside first (the editable install
    binds the venv to its path, so the new one must take that path), and comes back when the install or the verify
    fails: a failed update never leaves the host without its working engine. Success removes the old one and any
    ``venv-old-*`` left by an interrupted earlier run."""
    sha, wheel = wheel_record(repo)
    venv = repo / "venv"
    old_dir = repo / ".merlin"
    old = None
    if venv.exists():
        old = old_dir / f"venv-old-{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"setup: moving the working venv aside -> {old} (restored if the install fails, removed when it verifies)", flush=True)
        old_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(venv), str(old))
    env = compiler_env(a, env)
    env.pop("MERLIN_WHEEL", None)
    t0 = time.time()
    try:
        if not a.from_source:
            local = os.environ.get("MERLIN_WHEEL")
            if local and Path(local).is_file():
                path = Path(local)
            else:
                path = fetch_wheel(wheel_urls(repo), repo / ".merlin" / "downloads" / wheel, sha)
            env["MERLIN_WHEEL"] = str(path)
        rc = runner(["bash", str(repo / "release" / "install-env.sh"), str(venv)], env=env, cwd=repo)
        if rc != 0:
            raise SetupError(f"release/install-env.sh failed (rc {rc})")
        env.pop("MERLIN_WHEEL", None)  # --verify refuses it (an install-only variable)
        rc = runner(["bash", str(repo / "release" / "install-env.sh"), "--verify", str(venv)], env=env, cwd=repo)
        if rc != 0:
            raise SetupError(f"release/install-env.sh --verify failed (rc {rc})")
    except BaseException as e:  # noqa: BLE001 - restore on any failure, Ctrl+C included
        if old is not None:
            shutil.rmtree(venv, ignore_errors=True)
            shutil.move(str(old), str(venv))
            msg = "; the previous venv is back in place, the engine still serves"
        else:
            msg = f"; the partial venv is left at {venv} for diagnosis"
        if isinstance(e, SetupError):
            raise SetupError(str(e) + msg, e.code) from None
        print(f"setup: install interrupted{msg}", file=sys.stderr, flush=True)
        raise
    for stale in old_dir.glob("venv-old-*"):
        shutil.rmtree(stale, ignore_errors=True)
    print(f"setup: engine installed and verified in {time.time() - t0:.0f} s -> {venv}", flush=True)


def hf_tool(repo: Path, env: dict[str, str], runner: Callable[..., int] = sh) -> Path:
    """The ``hf`` CLI: the engine venv's, else a small ``.merlin/hf`` environment with the lock's huggingface-hub (so
    ``merlin pull`` works on a host that only downloads)."""
    hf = repo / "venv" / "bin" / "hf"
    if hf.exists():
        return hf
    hf = repo / ".merlin" / "hf" / "bin" / "hf"
    if hf.exists():
        return hf
    if which("uv", env) is None:
        raise SetupError("no `hf` (install the engine first, or run through ./setup.sh so uv is on PATH)")
    pins = {}
    for line in (repo / "release" / "requirements-0.32.lock").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^\s\\;]+)", line)
        if m:
            pins[m[1]] = m[2]
    cons = repo / ".merlin" / "constraints.txt"
    write_private(cons, "".join(f"{k}=={v}\n" for k, v in sorted(pins.items())), 0o644)
    venv = repo / ".merlin" / "hf"
    if runner(["uv", "venv", "--python", "3.13", str(venv)], env=env) != 0:
        raise SetupError("uv venv for the hf tool failed")
    if runner(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "--only-binary", ":all:", "-c", str(cons),
               f"huggingface-hub=={pins['huggingface-hub']}"], env=env) != 0:
        raise SetupError("installing huggingface-hub for the hf tool failed")
    return hf


def pull_checkpoint(repo: Path, a: Answers, env: dict[str, str], revision: str | None, rehash: bool = False,
                    runner: Callable[..., int] = sh) -> None:
    ckpt_dir = a.checkpoint_dir()
    list_path = checksum_list(a.model, repo)
    conflict = folder_conflict(ckpt_dir, a.model, repo)
    if conflict:
        raise SetupError(f"{ckpt_dir} {conflict}; nothing downloaded there. Choose another folder (--models-dir) or remove it.")
    sums, _ = parse_checksums(list_path)
    if checkpoint_missing(ckpt_dir, list_path):
        if pending_publication(a.model, repo):
            raise SetupError(not_published_msg(a.model))
        if not revision:
            raise SetupError(f"{a.model} has no pinned revision (record it in release/download-sizes.json when its Hugging Face "
                             "repository is published); nothing downloaded")
        hf = hf_tool(repo, env, runner)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        env = {**env, "HF_HUB_DISABLE_TELEMETRY": "1"}
        rc = runner([str(hf), "download", hf_repo(a.model, repo), "--revision", revision, "--local-dir", str(ckpt_dir)], env=env)
        if rc != 0:
            raise SetupError(f"hf download failed (rc {rc}); run again to resume")
    ok, bad = verify_checkpoint(ckpt_dir, list_path, model=a.model, rehash=rehash)
    if not ok:
        raise SetupError(f"checkpoint verification failed in {ckpt_dir}:\n  " + "\n  ".join(bad)
                         + "\n  delete the listed files and run again (hf download fetches them anew)")
    print(f"setup: checkpoint verified: {len(sums)} files in {ckpt_dir}", flush=True)


# --------------------------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------------------------
class Asker:
    def __init__(self, yes: bool, inp: Any = None, out: Any = None):
        self.yes = yes
        self.inp = inp or sys.stdin
        self.out = out or sys.stdout

    def say(self, text: str = "") -> None:
        self.out.write(text + "\n")
        self.out.flush()

    def ask(self, prompt: str, default: str, choices: list[str] | None = None) -> str:
        if self.yes:
            return default
        tty = getattr(self.inp, "isatty", lambda: False)()
        while True:
            self.out.write(f"{prompt} [{default}]: ")
            self.out.flush()
            line = self.inp.readline()
            if not line:  # EOF: the default
                self.out.write(default + "\n")
                return default
            v = line.strip() or default
            if choices and v.isdigit() and 1 <= int(v) <= len(choices):
                v = choices[int(v) - 1]
            if not tty:  # piped answers: show what was taken, as a terminal would
                self.out.write(v + "\n")
            if not choices or v in choices:
                return v
            self.say(f"  choose one of: {', '.join(choices)}")


def layout_label(a: Answers) -> str:
    if a.tp and a.instances and a.instances > 1:
        return f"--tp {a.tp} --instances {a.instances}"
    if a.tp:
        return f"--tp {a.tp}"
    if a.instances and a.instances > 1:
        return f"--instances {a.instances}"
    return "default layout"


def ask_questions(a: Answers, inv: pf.Inventory, ui: Asker, args: argparse.Namespace, repo: Path, table: dict[str, Any],
                  saved: dict[str, Any] | None, key_path: Path, only: set[str] | None = None) -> Answers:
    """Ask every question whose flag is unset; ``only`` limits it to those names (review_answers re-asks one)."""
    def want(name: str) -> bool:
        return only is None or name in only

    if want("model"):
        a = _ask_model(a, inv, ui, args, table, saved)
    if want("layout"):
        a = _ask_layout(a, inv, ui, args, repo)
    if want("folder") and args.models_dir is None:
        a.models_dir = ui.ask("Models folder (an existing download is reused)", a.models_dir or f"~/models/{a.model}")
    if want("folder"):
        ui.say(f"  checkpoint -> {a.checkpoint_dir()}")
    if want("access"):
        a = _ask_access(a, ui, args, key_path)
    if want("port") and args.port is None:
        p = ui.ask("Port", str(a.port or default_port(a.model)))
        if not p.isdigit() or not 1 <= int(p) <= 65535:
            raise SetupError(f"port {p!r}: 1..65535", 2)
        a.port = int(p)
    if ui.yes:
        ui.say(f"Answers: {a.model}, {layout_label(a)}, {a.checkpoint_dir()}, {a.host}:{a.port}" + (", API key" if a.api_key else ""))
    return a


def _ask_model(a: Answers, inv: pf.Inventory, ui: Asker, args: argparse.Namespace, table: dict[str, Any],
               saved: dict[str, Any] | None) -> Answers:
    rec, why = recommend(inv)
    menus = not ui.yes and args.model is None
    if menus:
        ui.say("Model (Enter = recommended):")
        for i, m in enumerate(MODELS, 1):
            size = table["checkpoints"][m].get("bytes")
            ui.say(f"  {i}) {m:<18} {TITLE[m]:<70} {gib(size) if size else '':>10}")
    ui.say(f"  recommended: {rec} ({why})" if rec else f"  no recommendation: {why}")
    if args.model is None:
        before = a.model
        a.model = ui.ask("Model", a.model or rec or "27b", MODELS)
        if a.model != before:  # another model: its own saved block, or its defaults (never the other model's folder)
            block = _valid_saved(saved)["models"].get(a.model, {})
            for k in PER_MODEL:
                if getattr(args, k) is None:  # a flag given on the command line stays
                    setattr(a, k, block.get(k))
    else:
        ui.say(f"Model: {a.model} (--model)")
    if _choose(a.model, inv) is None:
        ui.say(f"  note: no {a.model} layout fits this hardware (merlin doctor --model {FAMILY[a.model]} explains)")
    return a


def _ask_layout(a: Answers, inv: pf.Inventory, ui: Asker, args: argparse.Namespace, repo: Path) -> Answers:
    opts = layouts(a.model, inv, repo)
    if opts and args.tp is None and args.instances is None:
        labels = [o.label for o in opts]
        rec_opt = next(o for o in opts if o.recommended)
        if ui.yes:
            ui.say(f"Layout: {rec_opt.label}: {rec_opt.text} ({rec_opt.why}" + (f"; {len(opts) - 1} other option(s) with --reconfigure)" if len(opts) > 1 else ")"))
        else:
            ui.say("Layout (Enter = recommended):")
            for i, o in enumerate(opts, 1):
                ui.say(f"  {i}) {o.label:<22} {o.text}" + ("  <- recommended" if o.recommended else ""))
            ui.say(f"  recommended: {rec_opt.label} ({rec_opt.why})")
        cur = layout_label(a) if (a.tp or a.instances) else rec_opt.label  # a saved layout stays the default; else the recommendation
        pick = ui.ask("Layout", cur if cur in labels else rec_opt.label, labels)
        chosen = next(o for o in opts if o.label == pick)
        a.tp, a.instances = chosen.tp, chosen.instances
    elif args.tp is not None or args.instances is not None:
        ui.say(f"Layout: {layout_label(a)} (flags)")
    for note in layout_notes(a, inv, repo):
        if note.startswith("P2P is off"):  # the selector's measured note; every note repeats under the serve line
            ui.say(f"  NOTE {note}")
    return a


def _ask_access(a: Answers, ui: Asker, args: argparse.Namespace, key_path: Path) -> Answers:
    if args.host is None:
        if not ui.yes:
            ui.say("Access:")
            ui.say("  1) local    this machine only (127.0.0.1)")
            ui.say("  2) network  every host that can reach this machine (0.0.0.0), an API key is required")
        cur = "local" if a.host == "127.0.0.1" else "network"
        pick = ui.ask("Access", cur, ["local", "network"])
        a.host = "127.0.0.1" if pick == "local" else "0.0.0.0"
    if a.host != "127.0.0.1" and not a.api_key:
        a.api_key = secrets.token_urlsafe(32)
        ui.say(f"  API key generated; it is shown this once and kept in {key_path} (mode 0600).")
        ui.say(f"  Clients send `Authorization: Bearer <key>`; `cat {key_path}` prints it again.")
        ui.say(f"  {a.api_key}")
    if a.host == "127.0.0.1" and not args.api_key:
        a.api_key = None
    return a


REVIEW_ITEMS = ("model", "layout", "folder", "access", "port")


def review_answers(a: Answers, inv: pf.Inventory, ui: Asker, args: argparse.Namespace, repo: Path, table: dict[str, Any],
                   saved: dict[str, Any] | None, key_path: Path) -> Answers:
    """After the questions and before any check or download: list the answers and re-ask any one of them, as often as
    wanted, until Enter. Answers given by flags are shown but not offered. Another model asks its own layout, folder
    and port again (they belong to the model)."""
    flagged = {"model": args.model is not None, "layout": args.tp is not None or args.instances is not None,
               "folder": args.models_dir is not None, "access": args.host is not None, "port": args.port is not None}
    while True:
        values = {"model": a.model, "layout": layout_label(a), "folder": str(a.checkpoint_dir()),
                  "access": "local (127.0.0.1)" if a.host == "127.0.0.1" else f"network ({a.host}), API key",
                  "port": str(a.port or default_port(a.model))}
        open_ = [k for k in REVIEW_ITEMS if not flagged[k]]
        if not open_:
            return a
        ui.say("Your answers (nothing is downloaded or saved yet):")
        for i, k in enumerate(open_, 1):
            ui.say(f"  {i}) {k:<7} {values[k]}")
        for k in REVIEW_ITEMS:
            if flagged[k]:
                ui.say(f"     {k:<7} {values[k]} (flag)")
        pick = ui.ask(f"Change an answer? (Enter = no, or 1-{len(open_)})", "no", open_ + ["no", "n"])
        if pick in ("no", "n"):
            return a
        before = a.model
        a = ask_questions(a, inv, ui, args, repo, table, saved, key_path, only={pick})
        if pick == "model" and a.model != before:
            a = ask_questions(a, inv, ui, args, repo, table, saved, key_path, only={"layout", "folder", "port"})


def layout_notes(a: Answers, inv: pf.Inventory, repo: Path) -> list[str]:
    try:
        plans = serve_plan(a, inv, repo)
    except S.CliError as e:
        return [f"{a.model} cannot be served on this hardware: {e}"]
    return [n for p in plans for n in p.notes if not n.startswith("no checkpoint directory")]


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def host_report(inv: pf.Inventory, ui: Asker) -> None:
    names: dict[str, int] = {}
    for g in inv.gpus:
        names[g.name] = names.get(g.name, 0) + 1
    cards = ", ".join(f"{n} x {k} ({pf._gib(next(g.vram_mib for g in inv.gpus if g.name == k)):.0f} GB)" for k, n in names.items()) or "none"
    ram = f"{pf._gib(inv.host.ram_mib):.0f} GiB" if inv.host.ram_mib else "unknown"
    ui.say(f"Host: GPUs {cards}; P2P {inv.p2p}; driver {inv.host.driver or 'unknown'}; RAM {ram}"
           + (f"; {inv.source}" if inv.source != "probe" else ""))


def report_problems(problems: list[Problem], ui: Asker) -> bool:
    """Print; True when a FAIL is among them."""
    for p in problems:
        ui.say(f"{p.level} {p.code}: {p.text}")
        if p.fix:
            ui.say(f"     fix: {p.fix}")
    return any(p.level == "FAIL" for p in problems)


def main(argv: list[str] | None = None, inp: Any = None, out: Any = None, run: Callable[[list[str]], str | None] = default_runner) -> int:
    try:  # keep our lines in order with the subprocesses' output when stdout is a pipe or a log
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    try:
        return _main(list(sys.argv[1:] if argv is None else argv), inp, out, run)
    except SetupError as e:
        print(f"setup: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print("\nsetup: interrupted; ./setup.sh again continues (downloads resume)", file=sys.stderr)
        return 130


def _main(argv: list[str], inp: Any, out: Any, run: Callable[[list[str]], str | None]) -> int:
    args = parse_args(argv)
    repo = REPO
    merlin_dir = repo / ".merlin"
    key_path = merlin_dir / KEY_FILE
    ui = Asker(args.yes, inp, out)
    table = sizes_table(repo)
    saved = read_json(merlin_dir / SETUP_JSON)
    a = merge_answers(saved, args, args.api_key or read_key(key_path))
    if args.pull_only:
        return pull_main(args, a, repo, table, ui)
    inv = pf.simulate(args.simulate) if args.simulate else pf.probe()
    ui.say(f"Merlin {release_version(repo)} (by Halt95) setup ({repo})")
    host_report(inv, ui)

    # Host checks that cost no download.
    env = dict(os.environ)
    problems = [p for p in (git_problem(repo, run, env), gcc_problem(run, env, a), driver_problem(inv)) if p]
    if a.from_source:
        problems += source_build_problems(run, env)
    if report_problems(problems, ui):
        ui.say("setup: fix the FAIL lines above and run ./setup.sh again; nothing was downloaded.")
        return 1
    if a.cc and not os.environ.get("CC"):
        ui.say(f"Compiler: {a.cc} / {a.cxx} (remembered in .merlin/setup.json)")

    # Questions (a complete saved set for this model is reused unless --reconfigure).
    asked = False
    if not answers_complete(a, saved) or args.reconfigure:
        a = ask_questions(a, inv, ui, args, repo, table, saved, key_path)
        if not ui.yes:
            a = review_answers(a, inv, ui, args, repo, table, saved, key_path)
        asked = True
    else:
        ui.say(f"Saved answers from .merlin/setup.json (--reconfigure asks again): {a.model}, {layout_label(a)}, "
               f"{a.checkpoint_dir()}, {a.host}:{a.port}" + (", API key" if a.api_key else ""))
    if a.port is None:
        a.port = default_port(a.model)

    # Model-specific checks and the plan.
    try:
        plans = serve_plan(a, inv, repo)
    except S.CliError as e:
        ui.say(f"FAIL layout: {a.model} with {layout_label(a)} does not fit this host: {e}")
        ui.say("     fix: ./setup.sh --reconfigure picks a layout for the hardware found now")
        return 3
    cards = len(plans[0].cards)
    more = [p for p in [ram_problem(a.model, inv, cards)] if p]
    for w in dict.fromkeys(w for p in plans for w in p.warnings):  # once each: instances repeat their row's warnings
        more.append(Problem("WARN", *w.split(": ", 1)) if ": " in w else Problem("WARN", "serve", w))
    conflict = folder_conflict(a.checkpoint_dir(), a.model, repo)
    if conflict:
        more.append(Problem("FAIL", "folder", f"{a.checkpoint_dir()} {conflict}; nothing is downloaded over another checkpoint",
                            "choose another folder (--models-dir, or --reconfigure) or remove that one"))
    commit = head_commit(repo, run)
    state = read_json(merlin_dir / STATE_JSON)
    plan = download_plan(a, repo, state, commit, table=table)
    more += disk_problems(plan)
    failed = report_problems(more, ui)
    ui.say(render_plan(plan))
    ui.say("Serve: " + serve_line(a, repo))
    for p in plans:
        for n in p.notes:
            ui.say(f"  NOTE {n}")
    if failed:
        ui.say("setup: fix the FAIL lines above and run ./setup.sh again; nothing was downloaded.")
        return 1
    if a.model == "27b-thinkingcap":
        ui.say(f"Licence: {THINKINGCAP_LICENCE}")
    if args.dry_run:
        ui.say("Dry run: nothing downloaded, nothing installed, nothing written, nothing started.")
        return 0
    # Confirmation: once, after the questions; a saved re-run asks nothing (--yes neither).
    if asked and plan.pending() and ui.ask("Proceed with the downloads above? (y/n)", "y", ["y", "n", "yes", "no"]) in ("n", "no"):
        ui.say("setup: stopped; nothing downloaded and nothing saved.")
        return 1
    if a.api_key and (asked or args.api_key):
        write_private(key_path, a.api_key + "\n", 0o600)
    save_answers(merlin_dir / SETUP_JSON, saved, a)

    # Install, checkpoint, state.
    env["PATH"] = f"{merlin_dir / 'bin'}:{env.get('PATH', '')}"
    if plan.need_install:
        install_engine(repo, a, env)
        write_json(merlin_dir / STATE_JSON, {**install_state(repo, a.from_source, commit), "installed_at": int(time.time())})
    else:
        ui.say("Engine: venv is current (wheel + lock + commit unchanged since the last install); nothing to install.")
    pull_checkpoint(repo, a, env, plan.revision, args.rehash)
    if args.no_start:
        ui.say("Done. Start it with ./setup.sh")
        return 0

    # Doctor, endpoint, start.
    ui.say("")
    ui.say(f"merlin doctor --model {FAMILY[a.model]}:")
    S.main(["doctor", "--model", FAMILY[a.model]], out)
    print_endpoint(a, ui, key_path)
    sys.stdout.flush()
    os.execvpe(serve_argv(a, repo)[0], serve_argv(a, repo), serve_env(a))
    return 0  # pragma: no cover


def pull_main(args: argparse.Namespace, a: Answers, repo: Path, table: dict[str, Any], ui: Asker) -> int:
    """``merlin pull <model> [--models-dir D]``: the checkpoint only. No GPU, driver or compiler check, no serving
    choice; the folder is this model's (saved or default) unless --models-dir names another."""
    if not a.model:
        raise SetupError("merlin pull needs a model name", 2)
    ckpt_dir = a.checkpoint_dir()
    ui.say(f"Merlin {release_version(repo)} pull: {a.model} -> {ckpt_dir}")
    conflict = folder_conflict(ckpt_dir, a.model, repo)
    if conflict:
        ui.say(f"FAIL folder: {ckpt_dir} {conflict}")
        ui.say("     fix: choose another folder (--models-dir) or remove that one")
        return 1
    plan = download_plan(a, repo, None, None, table=table, engine=False)
    failed = report_problems(disk_problems(plan), ui)
    ui.say(render_plan(plan))
    if failed:
        return 1
    if a.model == "27b-thinkingcap":
        ui.say(f"Licence: {THINKINGCAP_LICENCE}")
    if args.dry_run:
        ui.say("Dry run: nothing downloaded.")
        return 0
    if not ui.yes and plan.pending() and ui.ask("Proceed with the download above? (y/n)", "y", ["y", "n", "yes", "no"]) in ("n", "no"):
        ui.say("pull: stopped; nothing downloaded.")
        return 1
    env = dict(os.environ)
    env["PATH"] = f"{repo / '.merlin' / 'bin'}:{env.get('PATH', '')}"
    pull_checkpoint(repo, a, env, plan.revision, args.rehash)
    ui.say("Done.")
    return 0


def print_endpoint(a: Answers, ui: Asker, key_path: Path) -> None:
    host = "127.0.0.1" if a.host in ("0.0.0.0", "::") else a.host
    auth = f' -H "Authorization: Bearer $(cat {shlex.quote(str(key_path))})"' if a.api_key else ""
    n = a.instances if a.instances and a.instances > 1 else 1
    ports = ", ".join(str(a.port + i) for i in range(n))
    body = json.dumps({"model": served_name(a.model), "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 32})
    ui.say("")
    ui.say(f"Endpoint: http://{host}:{a.port}/v1  (OpenAI-compatible; /v1/messages for /v1/messages clients)"
           + (f"; instances on ports {ports}" if n > 1 else ""))
    if a.host == "0.0.0.0":
        ui.say(f"  reachable from the network on every address of this machine, port {a.port}; the API key is in {key_path} (0600)")
    ui.say("Test (once the log says `Application startup complete`):")
    ui.say(f"  curl -s http://{host}:{a.port}/v1/chat/completions -H 'Content-Type: application/json'{auth} -d {shlex.quote(body)}")
    if n > 1:
        ui.say(f"--instances {n}: the servers start in the background (own process groups), logs merlin-<profile>-<port>.log in the "
               "current directory; stop them by PID.")
    else:
        ui.say("Ctrl+C stops the server; ./setup.sh starts it again without questions.")
    ui.say("")


if __name__ == "__main__":
    sys.exit(main())
