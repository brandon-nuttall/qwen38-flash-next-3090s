# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``merlin``: one front end over the profile registry, preflight and the one launcher.

    merlin serve [checkpoint] [path] [--profile ID | --tp N] [--instances N] [--dry-run] [-- extra vllm args]
    merlin list
    merlin doctor [preflight.py args]

The checkpoint name is optional when the directory can be identified (config.json + the recipes' checkpoint.sha256
lists). The profile comes from preflight (the largest TESTED layout that fits this host, else the largest fitting
untested row), from ``--profile`` or from ``--tp``; a host with more cards than the layout uses gets a card subset
(P2P-connected where the profile needs it, the tightest topology group first) and a hint for the free cards;
``--instances N`` partitions the host into N disjoint sets across every card class and starts one server per set.
``GPUS`` / ``CUDA_VISIBLE_DEVICES`` always win. Every profile starts
through ``tools/merlin/launch.py``, so a single launch's env + argv equal what the profile's shim produces (instances add
their own ports and caches, see ``plan_instances``). Standard library only.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch as L  # noqa: E402
import preflight as pf  # noqa: E402
import registry as R  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PLACEHOLDER = "<checkpoint-dir>"
LAUNCHER = "tools/merlin/launch.py"
# name -> registry key (launch.CHECKPOINTS is the source)
CHECKPOINTS: dict[str, str] = {k: v[0] for k, v in L.CHECKPOINTS.items()}
# the launcher files that pin a profile exactly (every other row starts through merlin serve --profile)
SHIMS: dict[tuple[str, str], str] = {
    ("flash-next", R.REFERENCE_ID): "models/flash-next/serve.sh",
    ("27b", "27b-tp1-3090"): "models/qwen38-27b/serve-1card.sh",
    ("27b", "27b-tp2-3090"): "models/qwen38-27b/serve.sh",
    ("27b", "27b-tp4-3090"): "models/qwen38-27b/serve-tp4.sh",
    ("27b-thinkingcap", "27b-tp2-3090"): "models/thinkingcap/serve.sh",
    ("27b-thinkingcap", "27b-tp4-3090"): "models/thinkingcap/serve-tp4.sh",
    ("nemotron-super-vl", "svl-quad-3090"): "models/nemotron-super-vl/serve.sh",
    ("ornith1.5", "ornith-quad-3090"): "models/ornith1.5/serve.sh",
}
# Flash-Next host RAM per instance (preflight.py RAM_FLOOR_GIB / RAM_WARN_GIB): ~47.7 GiB pinned PLE table + workers.
FN_RAM_FLOOR_GIB = pf.RAM_FLOOR_GIB
FN_RAM_WARN_GIB = pf.RAM_WARN_GIB
MIXED_FN_EVIDENCE = (
    "community-validated for mixed RTX 3090 + RTX 3090 Ti quads on the Flash-Next family: one public report "
    "(https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin/discussions/1, 2 x 3090 + 2 x 3090 Ti, PCIe 3.0, the v2.5.1 "
    "engine with Flash-Next / Swift) ran without errors; not measured on this engine"
)


NO_P2P_THROUGHPUT_NOTE = (
    "P2P is off on GPUs {gpus}: TP{tp} is the single-request latency and pool-size layout; for aggregate throughput "
    "without P2P prefer {n} single-card servers (measured on 4 x CMP 170HX 64 GB without P2P, 8 agents x 16K: "
    "TP1 x 4 = 873 tok/s, TP2 x 2 = 523, TP4 = 371; 4 x RTX 4090 showed the same order for prefill): {cmd}"
)


class CliError(Exception):
    def __init__(self, msg: str, code: int = 2):
        super().__init__(msg)
        self.code = code


@dataclass
class Plan:
    checkpoint: str
    profile_id: str
    status: str
    argv: list[str]  # the launcher invocation
    env: dict[str, str]  # what merlin serve adds to the environment
    launcher: str  # repo-relative: always tools/merlin/launch.py
    warnings: list[str] = field(default_factory=list)
    cards: list[int] = field(default_factory=list)
    free: list[int] = field(default_factory=list)  # same-class cards left unused
    skipped: list[int] = field(default_factory=list)  # cards of another class
    notes: list[str] = field(default_factory=list)
    vllm: L.Plan | None = None  # the resolved launcher plan (pure build), for dry-run display


def _profiles_for(name: str, reg: dict[str, Any]) -> list[dict[str, Any]]:
    if name == "flash-next":
        return [p for p in reg["profiles"] if "checkpoint" not in p]
    if name in ("swift1.5", "swift1.5-huihui"):
        return [p for p in reg["profiles"] if p.get("checkpoint") == name]
    return list(reg["profiles"])


def _load(name: str) -> dict[str, Any]:
    if name not in CHECKPOINTS:
        raise CliError(f"unknown checkpoint {name!r}; known: {', '.join(CHECKPOINTS)}")
    return R.load_registry(CHECKPOINTS[name])


def _gpu_selection(environ: dict[str, str]) -> list[int] | None:
    """One ordered GPU selection from CUDA_VISIBLE_DEVICES / GPUS; disagreeing selectors are refused."""
    sel: dict[str, list[int]] = {}
    for k in ("CUDA_VISIBLE_DEVICES", "GPUS"):
        v = environ.get(k, "").strip()
        if v:
            try:
                sel[k] = [int(x) for x in v.split(",")]
            except ValueError:
                raise CliError(f"{k}={v!r}: only comma-separated GPU indices are supported") from None
    if len(sel) == 2 and sel["CUDA_VISIBLE_DEVICES"] != sel["GPUS"]:
        raise CliError(
            f"CUDA_VISIBLE_DEVICES={environ['CUDA_VISIBLE_DEVICES']} and GPUS={environ['GPUS']} disagree; set one"
        )
    for k, v in sel.items():
        if len(set(v)) != len(v):
            raise CliError(f"{k}={environ[k]}: a GPU index is given twice")
    return next(iter(sel.values()), None)  # the caller's order is kept: it is the rank / pipeline-stage order


# --------------------------------------------------------------------------------------------
# Card subsets: classes, P2P cliques, tested-name preference, partitions
# --------------------------------------------------------------------------------------------
def _class_of(g: pf.Gpu) -> tuple[str | None, int | None]:
    return g.arch, R.vram_class(pf._gib(g.vram_mib))


def _groups(inv: pf.Inventory) -> list[list[pf.Gpu]]:
    """Homogeneous card groups (arch + VRAM class), largest first, index order inside a group."""
    by: dict[tuple[str | None, int | None], list[pf.Gpu]] = {}
    for g in inv.gpus:
        by.setdefault(_class_of(g), []).append(g)
    return sorted(by.values(), key=lambda gs: (-len(gs), gs[0].index))


def _tested_name(prof: dict[str, Any], g: pf.Gpu) -> bool:
    import re

    pats = prof["match"].get("tested_gpu_names")
    return not pats or any(re.search(p, g.name) for p in pats)


def _clique(inv: pf.Inventory, cards: list[int]) -> bool:
    return inv.p2p_among(cards) == "on"


def _peer_test(inv: pf.Inventory) -> Callable[[list[int]], bool]:
    """``_clique`` with the driver's pair set built once (the partition search tests thousands of sets). A card with
    BAR1 below 1 GiB has no usable peer (NCCL hangs at warmup on any P2P pair that includes it; preflight treats the set
    as P2P off), so such cards never join a P2P-connected set here either."""
    small = {g.index for g in inv.gpus if g.bar1_mib is not None and g.bar1_mib < pf.BAR1_MIN_MIB}
    if inv.p2p_ok is None:
        return lambda cards: len(cards) < 2 or (inv.p2p == "on" and not small & set(cards))
    ok = {(a, b) for a, b in inv.p2p_ok if a not in small and b not in small}
    ok |= {(b, a) for a, b in ok}
    return lambda cards: all((a, b) in ok for a, b in combinations(cards, 2))


def _idle(pool: list[pf.Gpu]) -> list[pf.Gpu]:
    """Cards that hold no other process (more than 1 GiB in use = busy: the pinned pool would not fit beside it)."""
    return [g for g in pool if (g.mem_used_mib or 0) <= pf.BUSY_MIB]


# --------------------------------------------------------------------------------------------
# Topology: how far apart two cards sit (nvidia-smi topo -m links, else the NUMA node, else unknown)
# --------------------------------------------------------------------------------------------
LINK_RANK = {"NV": 0, "PIX": 1, "PXB": 2, "PHB": 3, "NODE": 4, "SYS": 5}
UNKNOWN_RANK = 3  # no topology read: every pair counts as one host bridge apart (index order then decides)


def _link_rank(inv: pf.Inventory, a: pf.Gpu, b: pf.Gpu) -> int:
    """0 NVLink, 1 same PCIe switch, 2 switches apart, 3 same host bridge, 4 same NUMA node, 5 across sockets."""
    kind = inv.link(a.index, b.index)
    if kind is not None:
        return LINK_RANK.get("NV" if kind.upper().startswith("NV") else kind.upper(), UNKNOWN_RANK)
    if a.numa_node is not None and b.numa_node is not None:
        return LINK_RANK["PHB"] if a.numa_node == b.numa_node else LINK_RANK["SYS"]
    return UNKNOWN_RANK


def _set_cost(inv: pf.Inventory, by_index: dict[int, pf.Gpu], cards: list[int]) -> int:
    """The widest link inside one set: what a tensor-parallel group pays on every all-reduce."""
    return max((_link_rank(inv, by_index[a], by_index[b]) for a, b in combinations(cards, 2)), default=0)


def _topo_keys(inv: pf.Inventory, pool: list[pf.Gpu]) -> dict[int, tuple[int, int, int]]:
    """(NUMA node, host bridge group, PCIe switch group) per card index, each group named by its lowest index, so a
    sort over these keys walks the host switch by switch, bridge by bridge, node by node."""
    def groups(limit: int) -> dict[int, int]:
        label = {g.index: g.index for g in pool}
        for a, b in combinations(pool, 2):
            if _link_rank(inv, a, b) <= limit:
                la, lb = label[a.index], label[b.index]
                lo, hi = min(la, lb), max(la, lb)
                for k, v in label.items():
                    if v == hi:
                        label[k] = lo
        return label

    switch, bridge, node = groups(LINK_RANK["PIX"]), groups(LINK_RANK["PHB"]), groups(LINK_RANK["NODE"])
    return {g.index: (g.numa_node if g.numa_node is not None else node[g.index], bridge[g.index], switch[g.index]) for g in pool}


def _order(inv: pf.Inventory, prof: dict[str, Any], pool: list[pf.Gpu], need: int | None = None,
           topo: dict[int, tuple[int, int, int]] | None = None, counts: dict[str, int] | None = None) -> list[pf.Gpu]:
    """The selection order over a card pool: tested-name cards first, then the same model together (the larger model
    count first), then by topology (NUMA node, host bridge, PCIe switch), then index. When fewer than ``need`` cards
    carry a tested name no set can be labelled tested, so the same-model key comes first (2 x 3090 + 4 x 3090 Ti takes
    the homogeneous Ti quad, not a mixed one). With ``need`` given, the tightest complete topology group (PCIe switch,
    then host bridge, then NUMA node) of the leading card's model that holds a whole set moves to the front, so an
    uneven host (NUMA nodes 0,0,1,1,1,1) takes the same-node quad 2-5, not 0-3 across the sockets."""
    if counts is None:  # a partition passes the whole pool's counts and topology labels so every set is ranked alike
        counts = {}
        for g in pool:
            counts[g.name] = counts.get(g.name, 0) + 1
    topo = topo if topo is not None else _topo_keys(inv, pool)
    tested = sum(1 for g in pool if _tested_name(prof, g))
    model_first = need is not None and 0 < tested < need

    def key(g: pf.Gpu) -> tuple:
        t, m = (0 if _tested_name(prof, g) else 1), (-counts[g.name], g.name)
        return (*m, t, *topo[g.index], g.index) if model_first else (t, *m, *topo[g.index], g.index)

    order = sorted(pool, key=key)
    if need and order:
        lead = [g for g in order if g.name == order[0].name]  # the tested-name / same-model keys depend on the name alone
        for depth in (3, 2, 1):  # switch, host bridge, NUMA node: the tightest group holding a whole set leads
            groups: dict[tuple, list[pf.Gpu]] = {}
            for g in lead:
                groups.setdefault(topo[g.index][:depth], []).append(g)
            whole = next((s for s in groups.values() if len(s) >= need), None)
            if whole is not None:
                return whole + [g for g in order if g not in whole]
    return order


def _pick_subset(inv: pf.Inventory, prof: dict[str, Any], pool: list[pf.Gpu], need: int, want_p2p: str,
                 budget: "_Budget | None" = None) -> list[int] | None:
    """``need`` idle cards out of ``pool`` for ``prof`` in ``_order`` (tested names, same model, topology, index):
    P2P-connected when the profile needs it (``want_p2p`` on), preferred when it may use it (any); None when no such
    set exists."""
    idx = [g.index for g in _order(inv, prof, _idle(pool), need)]
    if need > len(idx):
        return None
    if want_p2p in ("on", "any"):
        clique = _peer_test(inv)
        try:  # the first P2P-connected set in order, pruned to each card's peers and bounded (32 cards, no P2P: 32 steps)
            found = _disjoint_cliques(inv, idx, need, 1, clique, budget)
        except _SearchBudget:
            found = None
        if found:
            return found[0]
        if want_p2p == "on":
            return None
    return sorted(idx[:need])


def _partition(inv: pf.Inventory, prof: dict[str, Any], pool: list[pf.Gpu], need: int, n: int, want_p2p: str,
               budget: "_Budget | None" = None) -> list[list[int]] | None:
    """``n`` disjoint sets of ``need`` idle cards, cut one set at a time over ``_order`` of what is left (tested names
    together, same model together, then NUMA node / host bridge / PCIe switch, then index; the same-model key leads
    when the cards left hold no all-tested set), so every set stays homogeneous and inside the tightest topology
    group the counts allow. The 4-card pair shape keeps the reference host's measured stride cut (0,2 / 1,3) unless
    the topology says the contiguous cut is tighter. When the cut does not satisfy the P2P need, a backtracking search
    over P2P-connected sets of the whole pool follows (None = no such partition exists; CliError when the search runs
    past its budget and the row needs P2P, which is not proof that none exists)."""
    pool = _idle(pool)
    if n * need > len(pool):
        return None
    tested = [g for g in pool if _tested_name(prof, g)]
    mixed = 0 < len(tested) < len(pool)
    left = list(pool)
    topo = _topo_keys(inv, pool)
    counts: dict[str, int] = {}
    for g in pool:
        counts[g.name] = counts.get(g.name, 0) + 1
    contiguous: list[list[int]] = []
    for _ in range(n):  # re-ordered after every complete set: 6 x 3090 + 8 x 3090 Ti -> 3090 quad, Ti quad, Ti quad
        take = _order(inv, prof, left, need, topo, counts)[:need]
        contiguous.append(sorted(g.index for g in take))
        left = [g for g in left if g not in take]
    head = [c for s in contiguous for c in s]
    idx = head + [g.index for g in _order(inv, prof, left, None, topo, counts)]  # the whole pool: the search may need more cards
    cuts = [contiguous]
    if len(pool) == 4 and need == 2 and n == 2 and not mixed:  # the reference shape: stride is the measured best
        stride = [sorted(head[i::n]) for i in range(n)]
        by_index = {g.index: g for g in pool}
        if sum(_set_cost(inv, by_index, s) for s in stride) <= sum(_set_cost(inv, by_index, s) for s in contiguous):
            cuts = [stride, contiguous]
        else:
            cuts = [contiguous, stride]
    if want_p2p not in ("on", "any"):
        return cuts[0]
    clique = _peer_test(inv)
    for cut in cuts:
        if all(clique(s) for s in cut):
            return cut
    try:
        sets = _disjoint_cliques(inv, idx, need, n, clique, budget)
    except _SearchBudget:
        if want_p2p == "on":
            raise CliError(
                f"the search for {n} disjoint P2P-connected sets of {need} cards over {len(idx)} cards ran past its budget of "
                f"{PARTITION_BUDGET} set tests (not proof that none exists); start the instances by hand (GPUS=<set> PORT=<port> merlin serve ...)"
            ) from None
        return cuts[0]
    if sets is not None:
        return sets
    return None if want_p2p == "on" else cuts[0]


PARTITION_BUDGET = 100_000  # clique tests: far above any 8-card host, a bound for larger hosts


class _SearchBudget(Exception):
    pass


class _Budget:
    """One search allowance shared by every clique search of one command (a plain serve's hints, one --instances
    plan), so retries over set counts and rows cannot each spend the whole PARTITION_BUDGET."""

    def __init__(self, limit: int | None = None):
        self.tests, self.limit, self.exhausted = 0, limit or PARTITION_BUDGET, False

    def spend(self) -> None:
        self.tests += 1
        if self.tests > self.limit:
            self.exhausted = True
            raise _SearchBudget()


def _disjoint_cliques(
    inv: pf.Inventory, idx: list[int], need: int, n: int, clique: Callable[[list[int]], bool] | None = None,
    budget: _Budget | None = None,
) -> list[list[int]] | None:
    """The first ``n`` disjoint P2P-connected sets of ``need`` cards in ``idx`` order, by backtracking: the first
    remaining card joins the next set (with cards it has peer access to) or stays unused. Cards with fewer than
    ``need - 1`` peers in ``idx`` can belong to no set and are dropped first; fewer than ``need x n`` cards left is a
    proven None without a search. ``_SearchBudget`` when the search exceeds the budget's clique tests."""
    clique = clique or _peer_test(inv)
    budget = budget or _Budget()
    if budget.exhausted:
        raise _SearchBudget()
    if need > 1:
        peers_of = {c: [d for d in idx if d != c and clique([c, d])] for c in idx}
        kept = [c for c in idx if len(peers_of[c]) >= need - 1]
    else:
        kept = list(idx)
    if len(kept) < need * n:
        return None

    def search(left: list[int], k: int) -> list[list[int]] | None:
        if k == 0:
            return []
        if len(left) < need * k:
            return None
        first, rest = left[0], left[1:]
        peers = [c for c in rest if clique([first, c])]
        if len(peers) >= need - 1:
            for others in combinations(peers, need - 1):
                budget.spend()
                s = [first, *others]
                if clique(s):
                    sub = search([i for i in rest if i not in others], k - 1)
                    if sub is not None:
                        return [sorted(s), *sub]
        return search(rest, k)

    return search(kept, n)


def _no_p2p_twin(p: dict[str, Any]) -> bool:
    """A 27B row for hosts without peer-to-peer (match p2p "off"): the untested no-P2P twin of a tested 3090 row, same
    layout and pin. It is ranked with the tested rows, so the 27B default on a no-P2P host stays the largest layout
    rather than falling back to the tested one-card row. Flash-Next's no-P2P rows keep their own order (unchanged until
    measured on cards without P2P)."""
    return p["status"] != "tested" and p["match"].get("p2p") == "off" and p["id"].startswith("27b-")


def _ordered_candidates(reg: dict[str, Any], name: str, profile: str | None, tp: int | None) -> list[dict[str, Any]]:
    """Selection order: tested multi-card rows (most cards first), then the 27B no-P2P twins (most cards first), then the
    tested single-card rows, then the rest; rank (the registry's own preference) breaks ties."""
    cands = _profiles_for(name, reg)
    if profile:
        cands = [p for p in cands if p["id"] == profile]
    if tp is not None:
        cands = [p for p in cands if (p.get("layout") or {}).get("tp", p["serve"].get("tensor_parallel")) == tp]
    def tier(p: dict[str, Any]) -> int:
        if p["status"] == "tested":
            return 0 if p["match"]["cards"] > 1 else 2
        return 1 if _no_p2p_twin(p) else 3
    return sorted(cands, key=lambda p: (tier(p), -p["match"]["cards"], p.get("rank", 9)))


def _fits_class(p: dict[str, Any], group: list[pf.Gpu]) -> bool:
    arch, vclass = _class_of(group[0])
    return arch in p["match"]["arch"] and p["match"]["vram_gib"] == vclass


def _want_p2p(reg: dict[str, Any], p: dict[str, Any], inv: pf.Inventory, group: list[pf.Gpu]) -> str:
    """What the profile needs from the subset: on (must be connected), any (prefer), off (no requirement)."""
    m = p["match"]["p2p"]
    if reg["model"] in pf.STRICT_P2P_MODELS:
        return m  # on / off / any as the row declares
    return "on" if m == "on" else "any"


@dataclass
class Choice:
    rep: pf.Report
    prof: dict[str, Any]
    cards: list[int]
    free: list[int]
    skipped: list[int]


def choose(
    reg: dict[str, Any], name: str, inv: pf.Inventory, profile: str | None, tp: int | None, want_cards: int | None = None,
    groups: list[list[pf.Gpu]] | None = None,
) -> Choice:
    """Pick a profile and a card set for ``name`` on ``inv``: the largest TESTED layout that fits a homogeneous card
    group, else the largest fitting untested row; smaller layouts take a subset. ``groups`` restricts the card groups
    considered (``--instances`` plans one class at a time). Raises CliError when nothing fits."""
    groups = _groups(inv) if groups is None else groups
    reasons: list[str] = []
    for status_pass in ("tested", "untested"):
        for group in groups:
            for p in _ordered_candidates(reg, name, profile, tp):
                if status_pass == "tested" and p["status"] != "tested" and not _no_p2p_twin(p):
                    continue  # the untested pass retries tested rows too: on other card names they run untested
                if not _fits_class(p, group):
                    continue
                need = p["match"]["cards"]
                if want_cards is not None and need != want_cards:
                    continue
                cls = f"{'/'.join(p['match']['arch'])} {p['match']['vram_gib']} GB"
                if need > len(group):
                    reasons.append(f"{p['id']}: needs {need} {cls} cards; this host has {len(group)} of that class")
                    continue
                if need > len(_idle(group)):
                    busy = [g.index for g in group if g not in _idle(group)]
                    reasons.append(f"{p['id']}: needs {need} idle {cls} cards; GPUs {','.join(map(str, busy))} already hold more than "
                                   f"1 GiB (skipped; GPUS={','.join(str(g.index) for g in group[:need])} forces them)")
                    continue
                cards = _pick_subset(inv, p, group, need, _want_p2p(reg, p, inv, group))
                if cards is None:
                    reasons.append(f"{p['id']}: needs {need} P2P-connected {cls} cards (peer-to-peer is {inv.p2p_among([g.index for g in group])} among them)")
                    continue
                if not profile and _no_p2p_twin(p) and inv.p2p_among(cards) == "on":
                    continue  # automatic selection: cards with peer-to-peer run the P2P row of the same layout
                rep = pf.evaluate(inv, reg, model=reg["model"], want_profile=p["id"], gpus=cards)
                if rep.exit_code != 0 or not rep.profile_id:
                    reasons += [f"{p['id']}: {f.text}" for f in rep.findings if f.level == "FAIL"]
                    continue
                if status_pass == "tested" and rep.status != "tested" and not _no_p2p_twin(p):
                    continue  # a tested row counts as tested only on an all-tested-name set; try it again as untested
                free = [g.index for g in group if g.index not in cards]
                skipped = [g.index for g in inv.gpus if g.index not in cards and g.index not in free]
                return Choice(rep, p, cards, free, skipped)
    if profile and not reasons:
        want = R.get_profile(reg, profile)["match"]
        have = sorted({f"{g.arch} {R.vram_class(pf._gib(g.vram_mib))} GB" for g in inv.gpus})
        reasons.append(f"--profile {profile} does not match this hardware: it needs {want['cards']} x "
                       f"{'/'.join(want['arch'])} {want['vram_gib']} GB; found {', '.join(have)}")
    why = pf.evaluate(inv, reg, model=reg["model"]).no_profile_reason if not profile and tp is None else None
    if why and not why.startswith("mixed GPUs"):  # the whole-host explanation; a pinned row's refusal is its own line above
        reasons.append(why)
    reasons.append(f"run `merlin doctor --model {reg['model']}` for the full report")
    raise CliError(
        f"no usable {name} profile on this hardware.\n  " + "\n  ".join(dict.fromkeys(reasons)), 3
    )


# --------------------------------------------------------------------------------------------
# One plan
# --------------------------------------------------------------------------------------------
def _hint_cmd(name: str, ckpt: str, free: list[int], port: int, tp: int | None = None, profile: str | None = None,
              instances: int | None = None) -> str:
    """A merlin serve command for the hints: ``GPUS=.. PORT=..`` when cards / a port are given, ``--tp`` / ``--profile``
    carried over, ``--instances N`` when asked."""
    pre = (f"GPUS={','.join(map(str, free))} " if free else "") + (f"PORT={port} " if port else "")
    tail = (f" --tp {tp}" if tp else "") + (f" --profile {profile}" if profile else "") + (f" --instances {instances}" if instances else "")
    return f"{pre}merlin serve {name} {shlex.quote(ckpt) if ckpt != PLACEHOLDER else ckpt}{tail}"


def _instance_env_hint(repo: Path, i: int, port: int) -> str:
    """What ``--instances`` would export for instance ``i`` on ``port`` beyond GPUS / PORT: its own cache root (every
    builder maps CACHE_ROOT to VLLM_CACHE_ROOT + TRITON_CACHE_DIR) and the distributed ports derived from PORT, so a
    hand-started second server shares nothing with instance 0."""
    m, d = L.distributed_ports(str(port))
    return f"CACHE_ROOT={repo.as_posix()}/.vllm-cache/instance-{i} MASTER_PORT={m} VLLM_PORT={d}"


def _tp1_fits(reg: dict[str, Any], inv: pf.Inventory, group: list[pf.Gpu], card: int) -> bool:
    """Whether some single-card row of this class passes preflight on ``card`` (A10G / L4 pins fall under the floor)."""
    return any(pf.evaluate(inv, reg, model=reg["model"], want_profile=q["id"], gpus=[card]).exit_code == 0
               for q in _profiles_for_model(reg) if q["match"]["cards"] == 1 and _fits_class(q, group))


def _profiles_for_model(reg: dict[str, Any]) -> list[dict[str, Any]]:
    return list(reg["profiles"])


def _throughput_note(reg: dict[str, Any], name: str, ckpt: str, inv: pf.Inventory, p: dict[str, Any], cards: list[int],
                     n_class: int) -> str:
    """Item 5's NOTE: the measurements always, the ``--tp 1 --instances N`` command only when a single-card row of this
    class passes preflight on these cards."""
    group = [g for g in inv.gpus if _class_of(g) == _class_of(next(x for x in inv.gpus if x.index == cards[0]))]
    if _tp1_fits(reg, inv, group, cards[0]):
        cmd = _hint_cmd(name, ckpt, [], 0, tp=1, instances=n_class)
    else:
        cmd = "no single-card 27B row fits these cards (their VRAM leaves no 262K pool on one card), so this layout stands"
    return NO_P2P_THROUGHPUT_NOTE.format(tp=p["match"]["cards"], gpus=",".join(map(str, cards)), n=n_class, cmd=cmd)


def _free_hint(reg: dict[str, Any], name: str, ckpt: str, inv: pf.Inventory, p: dict[str, Any], group: list[pf.Gpu],
               free: list[int], port: int, tp: int | None, profile: str | None, repo: Path) -> str:
    """What the idle cards of the chosen class left over can run: ``--instances k`` when k sets of the chosen layout fit
    the class, and one ``GPUS=.. PORT=.. CACHE_ROOT=..`` command for the largest row of this checkpoint that fits the
    free cards (the chosen one under --tp / --profile); or why nothing fits."""
    need = p["match"]["cards"]
    idle = len(_idle(group))
    budget = _Budget()
    k = 0
    for k in range(idle // need, 1, -1):  # how many sets of the chosen layout the idle cards really hold
        if not _fits_ram(reg, inv, [need] * k):
            continue
        try:
            if _partition(inv, p, group, need, k, _want_p2p(reg, p, inv, group), budget) is not None:
                break
        except CliError:
            continue
    else:
        k = 0
    parts: list[str] = []
    flags = (f" --tp {tp}" if tp else "") + (f" --profile {profile}" if profile else "")
    if k >= 2:
        parts.append(f"--instances {k}{flags} ({k} x {p['id']} on the {idle} idle cards of this class)")
    found = _hint_set(reg, name, inv, p, group, free, tp, profile, beside=[need], budget=budget)
    if not isinstance(found, str):
        q, pick = found
        same = q is p
        parts.append(f"{_instance_env_hint(repo, 1, port)} {_hint_cmd(name, ckpt, pick, port, tp=tp if same else None, profile=profile if same else None)}"
                     + ("" if same else f" ({q['id']}: the {len(free)} free cards hold no {p['id']})"))
        return f"GPUs {','.join(map(str, free))} free: {', or '.join(parts)}"
    why = found
    if parts:
        return f"GPUs {','.join(map(str, free))} free: {parts[0]} (no {name} layout for {len(free)} x {_class_name(group[0])}: {why})"
    return f"GPUs {','.join(map(str, free))} unused: no {name} layout for {len(free)} x {_class_name(group[0])} ({why})"


def _hint_set(reg: dict[str, Any], name: str, inv: pf.Inventory, p: dict[str, Any], group: list[pf.Gpu], free: list[int],
              tp: int | None, profile: str | None, beside: list[int] = (), budget: "_SearchBudget | None" = None,
              ) -> tuple[dict[str, Any], list[int]] | str:
    """The row and cards a hinted second server would take: the chosen row first (under --tp / --profile), then the
    largest row of this checkpoint that fits the free cards with a set that runs: P2P-connected where the row needs
    it, passing preflight on exactly those cards (the same side-effect-free check merlin serve applies to them), and
    fitting the host RAM beside the servers in ``beside`` (card counts). A string says why nothing fits."""
    ranked = [q for q in _ordered_candidates(reg, name, None, None) if _fits_class(q, group)]
    rows = [p, *[q for q in ranked if q is not p]] if (tp is not None or profile) else ranked
    pool = [g for g in group if g.index in free]
    sized = [q for q in rows if q["match"]["cards"] <= len(free)]
    refused: list[str] = []
    for q in sized:
        if not _fits_ram(reg, inv, [*beside, q["match"]["cards"]]):
            refused.append(f"host RAM {_ram_budget_gib(inv):.0f} GiB holds no {q['match']['cards']}-card Flash-Next server beside the "
                           f"one started (~{pf.ram_floor_gib(q['match']['cards']):.0f} GiB more)")
            continue
        pick = _pick_subset(inv, q, pool, q["match"]["cards"], _want_p2p(reg, q, inv, group), budget)
        if not pick:
            continue
        rep = pf.evaluate(inv, reg, model=reg["model"], want_profile=q["id"], gpus=pick)
        if rep.exit_code == 0 and rep.profile_id:
            return q, pick
        refused += [f"{q['id']} on GPUs {','.join(map(str, pick))}: {f.text}" for f in rep.findings if f.level == "FAIL"]
    if refused:
        return "; ".join(dict.fromkeys(refused))
    if sized:
        return f"no P2P-connected set for a {sized[0]['match']['cards']}-card row among them (peer-to-peer is {inv.p2p_among(free)})"
    smallest = min((q["match"]["cards"] for q in _profiles_for(name, reg) if _fits_class(q, group)), default=None)
    return f"the smallest {name} row of this class takes {smallest} card(s)" if smallest else f"no {name} row of this class"


def _ram_budget_gib(inv: pf.Inventory) -> float | None:
    """The host RAM new Flash-Next servers can share: what is available now on a probed host (running servers are
    already out of it), else MemTotal / the cgroup limit; None when unknown."""
    ram = inv.host.ram_mib
    if inv.source == "probe":
        avail = _mem_available_mib()
        if avail is not None and (ram is None or avail < ram):
            ram = avail
    return None if ram is None else pf._gib(ram)


def _fits_ram(reg: dict[str, Any], inv: pf.Inventory, sizes: list[int]) -> bool:
    """Whether Flash-Next servers of these card counts fit the host RAM together (other families pin no host table)."""
    if reg["model"] != "flash-next":
        return True
    budget = _ram_budget_gib(inv)
    return budget is None or sum(pf.ram_floor_gib(s) for s in sizes) <= budget


def _default_port(reg: dict[str, Any]) -> int:
    return int(reg.get("launch", {}).get("port") or reg["base"].get("port", 8000))


def resolve(
    name: str,
    path: str | None,
    profile: str | None,
    inv: pf.Inventory,
    extra: list[str],
    repo: Path = REPO,
    environ: dict[str, str] | None = None,
    tp: int | None = None,
) -> Plan:
    reg = _load(name)
    env_in = os.environ if environ is None else environ
    ckpt = path or env_in.get("MERLIN_CHECKPOINT_DIR") or PLACEHOLDER
    if not (repo / LAUNCHER).is_file():
        raise CliError(f"{LAUNCHER} is not in this tree ({repo}); nothing started.")
    sel = _gpu_selection(env_in)
    ids = [p["id"] for p in _profiles_for(name, reg)]
    if profile and profile not in ids:
        raise CliError(f"--profile {profile!r} does not belong to {name}; its profiles: {', '.join(ids)}")
    notes: list[str] = []
    if sel is not None:
        # GPUS / CUDA_VISIBLE_DEVICES: exactly those cards (preflight checks the set it is given)
        picked = [g for g in inv.gpus if g.index in sel]
        cands = [R.get_profile(reg, profile)] if profile else _ordered_candidates(reg, name, None, tp)
        if tp is not None and picked and len({_class_of(g) for g in picked}) == 1:
            shaped = [q for q in cands if _fits_class(q, picked) and q["match"]["cards"] == len(picked)]
            if not shaped:
                sizes = sorted({q["match"]["cards"] for q in cands if _fits_class(q, picked)})
                raise CliError(
                    f"--tp {tp} on {_class_name(picked[0])} takes {' or '.join(map(str, sizes)) or 'no'} card(s); "
                    f"GPUS={','.join(map(str, sel))} names {len(picked)}")
            cands = shaped
        order = [q["id"] for q in cands]
        rep = None
        if profile or tp is not None or reg["model"] == "flash-next" and name != "flash-next":
            for pid in order:
                rep = pf.evaluate(inv, reg, model=reg["model"], want_profile=pid, gpus=sel)
                if rep.exit_code == 0:
                    break
        else:
            rep = pf.evaluate(inv, reg, model=reg["model"], gpus=sel)
            if rep.exit_code == 0 and rep.profile_id not in ids:
                raise CliError(f"auto-selected {rep.profile_id} is not a {name} profile; pass --profile")
        if rep is None or rep.exit_code != 0 or not rep.profile_id:
            fails = [f"{f.code}: {f.text}" for f in (rep.findings if rep else []) if f.level == "FAIL"]
            raise CliError(
                f"no usable {name} profile on this hardware.\n  " + "\n  ".join(fails or ["no profile matched"]),
                rep.exit_code if rep and rep.exit_code else 3,
            )
        p = R.get_profile(reg, rep.profile_id)
        cards, free, skipped = [g.index for g in rep.inventory.gpus], [], []
    else:
        ch = choose(reg, name, inv, profile, tp)
        rep, p, cards, free, skipped = ch.rep, ch.prof, ch.cards, ch.free, ch.skipped
        if skipped:
            notes.append(f"GPUs {','.join(map(str, skipped))} skipped: another GPU class than the chosen set (a layout never spans classes; "
                         f"--instances N fills every class that has a row of the layout)")
        group = [g for g in inv.gpus if g.index in cards + free]
        busy_free = [g.index for g in group if g.index in free and g not in _idle(group)]
        free = [i for i in free if i not in busy_free]
        if busy_free:
            notes.append(f"GPUs {','.join(map(str, busy_free))} busy: they hold more than 1 GiB and are not offered to another server")
        if free:
            notes.append(_free_hint(reg, name, ckpt, inv, p, group, free, int(env_in.get("PORT") or _default_port(reg)) + 1,
                                    tp, profile, repo))
    names = sorted({g.name for g in rep.inventory.gpus})
    status = rep.status or p["status"]
    if reg["model"] == "27b" and len(cards) > 1 and rep.effective_p2p != "on":
        n_class = len(cards) if sel is not None else len(cards) + len(_idle([g for g in inv.gpus if g.index in free]))
        notes.append(_throughput_note(reg, name, ckpt, inv, p, cards, n_class))
    if len(names) > 1:
        notes.append(f"mixed cards in one layout ({' + '.join(names)}): it runs at the slower card's pace; P2P {rep.effective_p2p} across the set")
        if reg["model"] == "flash-next" and status == "untested" and all("3090" in n for n in names):
            status = "untested"
            notes.append(MIXED_FN_EVIDENCE)
    env: dict[str, str] = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, cards))}
    model = reg["model"]
    busy = [g.index for g in rep.inventory.gpus if (g.mem_used_mib or 0) > pf.BUSY_MIB]
    if busy:  # only reachable through GPUS / CUDA_VISIBLE_DEVICES: the selector skips busy cards
        notes.append(f"GPUs {','.join(map(str, busy))} already hold more than 1 GiB (another process?); the pinned pool may not fit")
    if model == "flash-next" and inv.source == "probe":
        avail = _mem_available_mib()
        if avail is not None and pf._gib(avail) < pf.ram_floor_gib(len(cards)):
            raise CliError(pf.ram_available_text(len(cards), avail) + ". Nothing started.")
    if model in ("27b", "svl"):
        if p["match"]["cards"] > 1 and p["match"]["p2p"] == "on" and rep.effective_p2p != "on":
            raise CliError(
                f"{p['id']} was qualified with peer-to-peer and P2P is {rep.effective_p2p} on these cards; refusing: "
                + pf.no_p2p_hint(reg, model, p["match"]["arch"][0], p["match"]["vram_gib"], cards, rep.effective_p2p, inv)
            )
        if p["match"]["cards"] > 1 and p["match"]["p2p"] == "any":
            env["P2P"] = rep.effective_p2p  # the launcher takes P2P=on|off
        elif p["match"]["cards"] > 1 and p["match"]["p2p"] == "off":
            env["P2P"] = "off"  # a no-P2P row runs without peer-to-peer whatever the driver reports
        env["GPUS"] = env["CUDA_VISIBLE_DEVICES"]
    elif model == "ornith":
        env["GPUS"] = env["CUDA_VISIBLE_DEVICES"]
        if p["match"]["cards"] > 1:
            env["P2P"] = p["match"]["p2p"]  # the row was qualified on exactly this interconnect
    else:
        if rep.p2p_mode and len(p["interconnect"]["modes"]) > 1:
            env["P2P"] = rep.p2p_mode
    if rep.adjusted is not None:  # the pin preflight derived for these cards (whole blocks where the row records them)
        env["KV_BYTES"] = str(rep.adjusted["pin_bytes"])
        maxlen = reg.get("launch", {}).get("max_model_len")
        if model == "svl" and maxlen and rep.adjusted["pool_tokens"] < maxlen:
            # Super VL serves its native 1M by default; a reduced pool that cannot hold one 1M session serves 262K
            env["MAXLEN"] = "262144"
            notes.append(f"the reduced KV pool (~{rep.adjusted['pool_tokens']:,} tokens) holds no {maxlen:,}-token session on these "
                         f"cards: MAXLEN=262144")
    for k in ("KV_BYTES", "P2P", "MAXLEN"):  # an explicit caller value always wins over what preflight derived
        if k in env and env_in.get(k) and env_in[k] != env[k]:
            notes.append(f"{k}={env_in[k]} from your environment overrides the derived {k}={env[k]}")
            env[k] = env_in[k]
    argv = [sys.executable, str(repo / LAUNCHER), "--model", model, "--profile", p["id"], "--checkpoint", name, "--status", status,
            "--", ckpt, *extra]
    warns = [f"{f.code}: {f.text}" for f in rep.findings if f.level == "WARN"]
    try:
        vllm = L.build(model, p["id"], ckpt, {**env_in, **env}, extra, None, name, repo, status=status)
    except L.LaunchError as e:
        raise CliError(f"{p['id']}: {e}", e.code) from None
    return Plan(name, p["id"], status, argv, env, LAUNCHER, warns, cards, free, skipped, notes, vllm)


def missing_resources(plan: Plan, environ: dict[str, str]) -> list[str]:
    """Launch inputs the launcher will refuse without (checked here so --dry-run can say so without starting)."""
    out: list[str] = []
    v = plan.vllm
    if v is None:
        return out
    if v.family == "flash-next":
        tree = v.tree or str(v.repo)
        if not (Path(tree) / "vllm").is_dir():
            out.append(f"TREE={tree} (no vllm/ package)")
    for path, msg in v.files:
        if not Path(path).is_file():
            out.append(msg)
    return out


# --------------------------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------------------------
def _running_fn_instances() -> int:
    """Flash-Next servers already running on this host (their environment carries the PLE offload timeout)."""
    n = 0
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/environ", "rb") as f:
                    envb = f.read()
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read()
            except OSError:
                continue
            if b"VLLM_PLE_OFFLOAD_READY_TIMEOUT=" in envb and b"vllm" in cmd and b"serve" in cmd and b"VLLM_PP_LAYER_PARTITION" in envb:
                n += 1
    except OSError:
        return 0
    return n  # every rank of one server counts; divide by the layout size below


def _class_name(g: pf.Gpu) -> str:
    return f"{g.arch} {R.vram_class(pf._gib(g.vram_mib))} GB"


def _gpus_of(group: list[pf.Gpu]) -> str:
    return f"GPUs {','.join(str(g.index) for g in group)} ({_class_name(group[0])})"


def _allocate_instances(
    reg: dict[str, Any], name: str, inv: pf.Inventory, profile: str | None, tp: int | None, n: int,
) -> tuple[list[tuple[dict[str, Any], list[int]]], list[str]]:
    """``n`` (profile, card set) pairs over every card class of the host, each set homogeneous and running its own
    row. Per layout size (the ``--tp`` / ``--profile`` one, else every size): class by class (tested first, then the
    larger VRAM class, then index), ``choose`` picks the row for the idle cards, its disjoint sets are cut, and
    ``choose`` runs again over the cards left so another row of the same size (the no-P2P quad behind a P2P quad) can
    fill them. Flash-Next sets count against the host RAM as they are added. Among the sizes that reach ``n`` the one
    with the most tested instances wins, then the larger size. Returns the sets and the notes (classes left out and
    why). Raises CliError naming the best attempt (sets, unused cards, what stopped it) when fewer than ``n`` fit."""
    groups = _groups(inv)
    sizes: list[int | None] = [None] if profile or tp is not None else sorted(
        {q["match"]["cards"] for q in _profiles_for(name, reg)}, reverse=True)
    budget = _Budget()
    ram_budget = _ram_budget_gib(inv) if reg["model"] == "flash-next" else None
    attempts: list[tuple[list[tuple[dict[str, Any], list[int]]], int, list[str], list[str], list[tuple[list[pf.Gpu], str]], int]] = []
    for size in sizes:
        sets: list[tuple[dict[str, Any], list[int]]] = []
        tested_n = 0
        have: list[str] = []
        short: list[str] = []
        left_out: list[tuple[list[pf.Gpu], str]] = []  # (class group, why it has no row)
        classes: list[tuple[Choice, list[pf.Gpu]]] = []
        for group in groups:
            try:
                classes.append((choose(reg, name, inv, profile, tp, want_cards=size, groups=[group]), group))
            except CliError as e:
                why = str(e).split("\n")
                left_out.append((group, why[1].strip() if len(why) > 1 else why[0]))
        classes.sort(key=lambda t: (0 if t[0].rep.status == "tested" else 1, -(t[0].prof["match"]["vram_gib"]), t[1][0].index))
        remaining = n
        ram_used = 0.0
        for ch, group in classes:
            pool = _idle(group)
            first = True
            while remaining and pool:
                if not first:
                    try:
                        ch = choose(reg, name, inv, profile, tp, want_cards=size, groups=[pool])
                    except CliError:
                        break
                first = False
                p = ch.prof
                need = p["match"]["cards"]
                want = _want_p2p(reg, p, inv, group)
                k_max = min(len(pool) // need, remaining)
                if ram_budget is not None:
                    k_ram = int((ram_budget - ram_used) // pf.ram_floor_gib(need))
                    if k_ram < k_max:
                        short.append(f"host RAM {ram_budget:.0f} GiB holds {int(ram_budget // pf.ram_floor_gib(need))} x {need}-card "
                                     f"Flash-Next server(s) (~{pf.ram_floor_gib(need):.0f} GiB each)")
                        k_max = k_ram
                if k_max < 1:
                    break
                got: list[list[int]] = []
                budget_hit = False
                for k in range(k_max, 0, -1):
                    try:
                        found = _partition(inv, p, pool, need, k, want, budget)
                    except CliError as e:
                        short.append(f"{_gpus_of(pool)}: {e}")
                        budget_hit = True
                        break  # the budget is spent: smaller k would raise at once too
                    if found is not None:
                        got = found
                        break
                if not got:
                    if not budget_hit:
                        short.append(f"{_gpus_of(pool)}: no disjoint P2P-connected set of {need} cards for {p['id']} "
                                     f"(peer-to-peer is {inv.p2p_among([g.index for g in pool])} among them)")
                    break
                if len(got) < k_max:
                    short.append(f"{_gpus_of(pool)}: {len(got)} of {k_max} disjoint P2P-connected sets of {need} cards for {p['id']} "
                                 f"(peer-to-peer is {inv.p2p_among([g.index for g in pool])} among them)")
                sets += [(p, cards) for cards in got]
                tested_n += len(got) if (ch.rep.status == "tested" or _no_p2p_twin(p)) else 0
                have.append(f"{len(got)} x {p['id']} on GPUs " + " / ".join(",".join(map(str, c)) for c in got))
                remaining -= len(got)
                ram_used += len(got) * pf.ram_floor_gib(need)
                used = {c for cards in got for c in cards}
                pool = [g for g in pool if g.index not in used]
        attempts.append((sets, tested_n, have, short, left_out, size or 0))
    best = max(attempts, key=lambda a: (len(a[0]) >= n, a[1], len(a[0]), a[5]))
    sets, tested_n, have, short, left_out, _size = best
    if len(sets) >= n:
        return sets[:n], [f"{_gpus_of(group)} not used: {why}" for group, why in left_out]
    used = {c for _, cards in sets for c in cards}
    unused = [g.index for group in groups for g in _idle(group) if g.index not in used]
    lines = [f"{_gpus_of(group)}: {why}" for group, why in left_out] + short
    raise CliError(f"--instances {n}: this host holds {len(sets)} instance(s): {'; '.join(have) or 'none'}"
                   + (f"; unused GPUs {','.join(map(str, unused))}" if unused else "")
                   + "".join(f"\n  {ln}" for ln in dict.fromkeys(lines)))


def plan_instances(
    name: str, path: str | None, profile: str | None, inv: pf.Inventory, extra: list[str], n: int,
    repo: Path = REPO, environ: dict[str, str] | None = None, tp: int | None = None, served_suffix: bool = False,
) -> list[Plan]:
    """``n`` disjoint card sets across the host's card classes (``_allocate_instances``: each set homogeneous, each class
    on its own row), each resolved as ``GPUS=<set> PORT=<base+i> merlin serve`` plus what an instance needs of its own:
    ``MASTER_PORT`` / ``VLLM_PORT`` from ``launch.distributed_ports(PORT)`` (exported on every family; a single
    Flash-Next or 27B launch exports none) and, after the first instance, ``CACHE_ROOT``, ``VLLM_CACHE_ROOT`` and
    ``TRITON_CACHE_DIR`` under ``<cache root>/instance-<i>``."""
    reg = _load(name)
    env_in = dict(os.environ if environ is None else environ)
    if _gpu_selection(env_in) is not None:
        raise CliError("--instances partitions the host itself; unset GPUS / CUDA_VISIBLE_DEVICES (or start the instances by hand)")
    if n < 1:
        raise CliError("--instances needs a whole number >= 1")
    sets, class_notes = _allocate_instances(reg, name, inv, profile, tp, n)
    base_port = int(env_in.get("PORT") or _default_port(reg))
    if any(x.startswith(("--port", "--host")) for x in extra):
        raise CliError("--instances assigns each server its own port: set PORT=<base> instead of -- --port / --host")
    plans: list[Plan] = []
    used_ports: list[list[int]] = []
    for i, (p, cards) in enumerate(sets):
        e = dict(env_in)
        e["GPUS"] = e["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cards))
        port = base_port + i
        e["PORT"] = str(port)
        try:  # never a caller's export: the Super VL builder derives the same pair from PORT itself
            m, d = L.distributed_ports(str(port))
        except L.LaunchError as err:
            raise CliError(f"instance {i}: {err}; lower PORT", err.code) from None
        e["MASTER_PORT"], e["VLLM_PORT"] = str(m), str(d)
        cache_root = e.get("CACHE_ROOT") or f"{repo.as_posix()}/.vllm-cache"
        if i > 0:  # compile / Triton / topology caches of their own (the 27B builder exports no cache variables itself)
            e["CACHE_ROOT"] = e["VLLM_CACHE_ROOT"] = f"{cache_root}/instance-{i}"
            e["TRITON_CACHE_DIR"] = f"{e['CACHE_ROOT']}/triton"
        if served_suffix:
            if reg["model"] == "flash-next":
                ck = reg.get("checkpoints", {}).get(p.get("checkpoint") or "", {})  # Swift / Huihui serve their own names
                names = (e.get("MODEL_NAME") or ck.get("served_names") or reg["base"]["served_names"]).split()
                e["MODEL_NAME"] = " ".join(f"{x}-{i}" for x in names)
            else:
                default_served = reg["launch"]["served"] if reg["model"] in ("svl", "ornith") else reg["launch"]["checkpoints"][name]["served"]
                e["SERVED"] = f"{e.get('SERVED') or default_served}-{i}"
        plan = resolve(name, path, p["id"], inv, extra, repo, e, tp)
        plan.env.update({k: e[k] for k in ("PORT", "MASTER_PORT", "VLLM_PORT", "CACHE_ROOT", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR",
                                           "MODEL_NAME", "SERVED") if k in e})
        http = L.effective_host_port(plan.vllm.args, plan.vllm.extra)[1]  # type: ignore[union-attr]
        ports = [http, m, d, d + 1]
        if len(set(ports)) < len(ports):
            raise CliError(f"instance {i}: HTTP / MASTER_PORT / VLLM_PORT collide ({ports}); choose another PORT")
        for j, prev in enumerate(used_ports):
            shared = sorted(set(ports) & set(prev))
            if shared:
                raise CliError(f"instance {i} and instance {j} would share a port ({shared}); choose another PORT")
        used_ports.append(ports)
        plan.notes = [x for x in plan.notes if (not x.startswith("GPUs ") or "skipped" in x) and (i == 0 or not x.startswith("P2P is off"))]
        if i == 0 and any(x.startswith("P2P is off") for x in plan.notes):  # the throughput note counts the whole class
            first = next(g for g in inv.gpus if g.index == cards[0])
            n_class = len(_idle([g for g in inv.gpus if _class_of(g) == _class_of(first)]))
            plan.notes = [_throughput_note(reg, name, path or PLACEHOLDER, inv, p, cards, n_class)
                          if x.startswith("P2P is off") else x for x in plan.notes]
        plans.append(plan)
    used = {c for _, cards in sets for c in cards}
    port = base_port + n
    for group in _groups(inv):
        left = [g.index for g in _idle(group) if g.index not in used]
        mine = [(q, cards) for q, cards in sets if cards[0] in [g.index for g in group]]
        if left and mine:  # a class that got instances
            found = _hint_set(reg, name, inv, mine[0][0], group, left, tp, profile, beside=[len(c) for _, c in sets])
            if not isinstance(found, str):
                q, pick = found
                same = q is mine[0][0]
                plans[0].notes.append(f"GPUs {','.join(map(str, left))} free: {_instance_env_hint(repo, n, port)} "
                                      f"{_hint_cmd(name, path or PLACEHOLDER, pick, port, tp=tp if same else None, profile=profile if same else None)}")
                port += 1
            else:
                plans[0].notes.append(f"GPUs {','.join(map(str, left))} unused: no {name} layout for {len(left)} x {_class_name(group[0])} ({found})")
    plans[0].notes += class_notes
    if reg["model"] == "flash-next":
        need = max(p["match"]["cards"] for p, _ in sets)
        running = _running_fn_instances() // max(need, 1)
        floors = [pf.ram_floor_gib(len(cards)) for _, cards in sets]  # an octo instance needs more than a quad
        ram, what, count = inv.host.ram_mib, "host RAM", n + running
        need_gib = sum(floors) + running * FN_RAM_FLOOR_GIB
        avail = _mem_available_mib()
        if avail is not None and (ram is None or avail < ram) and inv.source == "probe":
            # what is free now, not MemTotal: running servers' PLE tables are already out of it, so only new ones count
            ram, what, count, need_gib = avail, "host memory available now", n, sum(floors)
        if ram is not None:
            have = pf._gib(ram)
            beside = f" beside {running} already running" if running else ""
            if have < need_gib:
                per = " / ".join(f"~{f:.0f}" for f in sorted(set(floors)))
                raise CliError(
                    f"{what} {have:.0f} GiB cannot hold {n} Flash-Next instance(s){beside}: each pins a ~47.7 GiB PLE table "
                    f"and needs {per} GiB resident ({FN_RAM_FLOOR_GIB:.0f} GiB at four workers plus "
                    f"{pf.RAM_PER_EXTRA_WORKER_GIB:.0f} GiB per worker above four; ~{need_gib:.0f} GiB for {count}; "
                    f"{FN_RAM_WARN_GIB:.0f} GiB per instance is the qualified allocation). Start fewer instances."
                )
            if have < count * FN_RAM_WARN_GIB:
                plans[0].warnings.append(
                    f"ram: {what} {have:.0f} GiB is below {count} x {FN_RAM_WARN_GIB:.0f} GiB qualified "
                    f"(floor {need_gib:.0f} GiB; the boot peak is unmeasured)"
                )
    return plans


def _mem_available_mib() -> int | None:
    """MemAvailable capped by what the cgroup has left (``preflight.mem_available_mib``); None off Linux."""
    return pf.mem_available_mib()


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------
def cmd_list(out: Any, repo: Path = REPO, checkpoint: str | None = None, arch: str | None = None, cards: int | None = None,
             vram: int | None = None) -> int:
    """checkpoint -> profile -> status -> hardware -> launcher file, one row per profile; the filters narrow the rows to
    one checkpoint name, an arch (sm_80 / sm_86 / sm_89), a card count and a VRAM class."""
    if checkpoint and checkpoint not in CHECKPOINTS:
        raise CliError(f"unknown checkpoint {checkpoint!r}; known: {', '.join(CHECKPOINTS)}")
    out.write(f"{'checkpoint':<18} {'profile':<36} {'status':<9} {'hardware':<30} launcher file (shim for {LAUNCHER})\n")
    for name, key in CHECKPOINTS.items():
        if checkpoint and name != checkpoint:
            continue
        reg = R.load_registry(key)
        for p in _profiles_for(name, reg):
            m = p["match"]
            if (arch and arch not in m["arch"]) or (cards is not None and m["cards"] != cards) or (vram is not None and m["vram_gib"] != vram):
                continue
            hw = f"{m['cards']} x {m['vram_gib']} GB {'/'.join(m['arch'])} p2p={m['p2p']}"
            rep_launcher = SHIMS.get((name, p["id"]))
            if rep_launcher is None and key == "flash-next":
                rep_launcher = p.get("recipe_launcher") or (f"tools/merlin/launchers/{R.launcher_name(p['id'])}" if p["id"] in reg.get("launchers", []) else None)
            rep_launcher = rep_launcher or f"(merlin serve {name} --profile {p['id']})"
            where = rep_launcher if rep_launcher.startswith("(") or (repo / rep_launcher).is_file() else f"{rep_launcher}  [MISSING in this tree]"
            out.write(f"{name:<18} {p['id']:<36} {p['status']:<9} {hw:<30} {where}\n")
    return 0


def _print_plan(plan: Plan, out: Any, label: str) -> None:
    out.write(f"# {label}\n")
    for n in plan.notes:
        out.write(f"# NOTE {n}\n")
    for w in plan.warnings:
        out.write(f"# WARN {w}\n")
    for m in missing_resources(plan, os.environ):
        out.write(f"# MISSING {m}\n")
    for k, v in plan.env.items():
        out.write(f"{k}={shlex.quote(v)} \\\n")
    out.write(" ".join(shlex.quote(a) for a in plan.argv) + "\n")
    if plan.vllm is not None:
        out.write("# resolves to:\n# " + " ".join(shlex.quote(a) for a in plan.vllm.argv()) + "\n")


def _identify(args: argparse.Namespace) -> tuple[str, str | None, list[str]]:
    targets = list(args.target)
    notes: list[str] = []
    if targets and targets[0] in CHECKPOINTS:
        name = targets[0]
        path = targets[1] if len(targets) > 1 else None
        if len(targets) > 2:
            raise CliError(f"unexpected argument {targets[2]!r} (put extra vllm args after --)")
    elif targets:
        path = targets[0]
        if len(targets) > 1:
            raise CliError(f"unexpected argument {targets[1]!r}: the first argument must be a checkpoint name ({', '.join(CHECKPOINTS)}) or a directory")
        if not Path(path).is_dir():
            raise CliError(f"{path!r} is neither a checkpoint name ({', '.join(CHECKPOINTS)}) nor a directory")
        try:
            name, notes = L.identify_checkpoint(path)
        except L.LaunchError as e:
            raise CliError(str(e), e.code) from None
        notes.insert(0, f"identified {path} as {name}")
    else:
        path = os.environ.get("MERLIN_CHECKPOINT_DIR")
        if not path:
            raise CliError("give the checkpoint directory (positional, or MERLIN_CHECKPOINT_DIR), or a checkpoint name")
        try:
            name, notes = L.identify_checkpoint(path)
        except L.LaunchError as e:
            raise CliError(str(e), e.code) from None
    return name, path, notes


def cmd_serve(args: argparse.Namespace, extra: list[str], out: Any) -> int:
    if args.simulate and not args.dry_run:
        print("merlin: --simulate only works with --dry-run (it never launches on fabricated hardware)", file=sys.stderr)
        return 2
    if args.profile and args.tp is not None:
        print("merlin: give --profile or --tp, not both", file=sys.stderr)
        return 2
    try:
        name, path, id_notes = _identify(args)
        if path is not None and not Path(path).is_dir():
            raise CliError(f"checkpoint directory {path!r} is not a directory")
        if path is None and not os.environ.get("MERLIN_CHECKPOINT_DIR"):
            id_notes.append("no checkpoint directory given: the command below shows " + PLACEHOLDER + " (dry-run only)")
        inv = pf.simulate(args.simulate) if args.simulate else pf.probe(pf.default_runner)
        if args.instances is not None and args.instances != 1:
            plans = plan_instances(name, path, args.profile, inv, extra, args.instances, repo=REPO, tp=args.tp,
                                   served_suffix=args.served_suffix)
        else:
            plans = [resolve(name, path, args.profile, inv, extra, repo=REPO, tp=args.tp)]
    except CliError as e:
        print(f"merlin: {e}", file=sys.stderr)
        return e.code
    except (ValueError, OSError, KeyError) as e:
        print(f"merlin: {e}", file=sys.stderr)
        return 2
    plans[0].notes[:0] = id_notes
    if args.dry_run:
        for i, plan in enumerate(plans):
            cards = ",".join(map(str, plan.cards))
            label = f"{plan.checkpoint} -> {plan.profile_id} [{plan.status}] on GPUs {cards} -> {plan.launcher}"
            if len(plans) > 1:
                label = f"instance {i}: " + label + f" port {plan.env.get('PORT')}"
            _print_plan(plan, out, label)
        return 0
    for plan in plans:
        if PLACEHOLDER in plan.argv:
            print("merlin: give the checkpoint directory (positional, or MERLIN_CHECKPOINT_DIR)", file=sys.stderr)
            return 2
        gone = missing_resources(plan, os.environ)
        if gone:
            print("merlin: missing launch resources, nothing started:\n  " + "\n  ".join(gone), file=sys.stderr)
            return 2
    sys.stdout.flush()
    if len(plans) == 1:
        plan = plans[0]
        print(f"merlin: {plan.checkpoint} -> {plan.profile_id} [{plan.status}] on GPUs {','.join(map(str, plan.cards))} -> {plan.launcher}", file=sys.stderr)
        for n in plan.notes:
            print(f"merlin: NOTE {n}", file=sys.stderr)
        for w in plan.warnings:
            print(f"merlin: WARN {w}", file=sys.stderr)
        os.execvpe(plan.argv[0], plan.argv, {**os.environ, **plan.env})
        return 0  # pragma: no cover
    log_dir = Path(os.environ.get("MERLIN_LOG_DIR") or os.getcwd())
    started = []
    for i, plan in enumerate(plans):
        env = {**os.environ, **plan.env}
        host, port = L.effective_host_port(plan.vllm.args, plan.vllm.extra)  # type: ignore[union-attr]
        try:
            free = L.port_free(host, port)
        except L.LaunchError as e:
            print(f"merlin: {e}; nothing more started", file=sys.stderr)
            return e.code
        if not free:
            print(f"merlin: port {port} on {host} is busy; nothing more started (PORT={port + len(plans)} moves the whole block)", file=sys.stderr)
            return 2
        log = log_dir / f"merlin-{plan.profile_id}-{port}.log"
        with open(log, "ab") as fh:
            proc = subprocess.Popen(plan.argv, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        started.append((i, plan, port, proc.pid, log))
        print(f"merlin: instance {i}: {plan.profile_id} [{plan.status}] GPUs {','.join(map(str, plan.cards))} -> http://{host}:{port} pid {proc.pid} (own process group) log {log}", file=sys.stderr)
        for n in plan.notes:
            print(f"merlin: NOTE {n}", file=sys.stderr)
    print("merlin: endpoints: " + " ".join(f"http://{L.effective_host_port(p.vllm.args, p.vllm.extra)[0]}:{port}" for _, p, port, _, _ in started), file=sys.stderr)  # type: ignore[union-attr]
    return 0


def main(argv: list[str] | None = None, out: Any = None) -> int:
    out = out or sys.stdout
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["doctor"]:  # transparent forwarding to preflight (options included)
        rest = argv[1:]
        return pf.main(rest[1:] if rest[:1] == ["--"] else rest, stdout=out)
    if argv[:1] == ["pull"]:  # download + verify a checkpoint (setup_cli does it; the engine is installed first when missing)
        import setup_cli

        if len(argv) < 2 or argv[1] not in CHECKPOINTS:
            print(f"merlin: pull needs a checkpoint name ({', '.join(CHECKPOINTS)}) [--models-dir DIR]", file=sys.stderr)
            return 2
        return setup_cli.main(["--pull-only", "--yes", "--model", argv[1], *argv[2:]], out=out)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(prog="merlin", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve a checkpoint through the one launcher")
    s.add_argument("target", nargs="*", help="[checkpoint name] [checkpoint directory]; a directory alone is identified")
    s.add_argument("--profile", help="force a profile id of this checkpoint (checked against the hardware)")
    s.add_argument("--tp", type=int, help="pick the profile with this tensor-parallel size (a subset of a larger host)")
    s.add_argument("--instances", type=int, help="start N servers on N disjoint card sets (consecutive ports from PORT)")
    s.add_argument("--served-suffix", action="store_true", help="with --instances: append -<i> to the served model name(s)")
    s.add_argument("--dry-run", action="store_true", help="print the resolved command and env; start nothing")
    s.add_argument("--simulate", metavar="SPEC", help="preflight's hardware spec, e.g. 4x3090 or 8x4090+8x4090-48g:numa=2 (--dry-run only)")
    ls = sub.add_parser("list", help="checkpoint -> profile -> status -> hardware -> launcher table")
    ls.add_argument("checkpoint", nargs="?", help="only this checkpoint's rows")
    ls.add_argument("--arch", help="only rows for this arch (sm_80, sm_86, sm_89)")
    ls.add_argument("--cards", type=int, help="only rows of this card count")
    ls.add_argument("--vram", type=int, help="only rows of this VRAM class (GB)")
    sub.add_parser("doctor", help="run tools/merlin/preflight.py (all following args pass through)")
    sub.add_parser("pull", help="download and verify a checkpoint: merlin pull <checkpoint> [--models-dir DIR]")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        try:
            return cmd_list(out, REPO, args.checkpoint, args.arch, args.cards, args.vram)
        except CliError as e:
            print(f"merlin: {e}", file=sys.stderr)
            return e.code
    return cmd_serve(args, extra, out)


if __name__ == "__main__":
    sys.exit(main())
