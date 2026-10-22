# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Reduced draft vocabulary for the Qwen4Exp MTP drafter (default OFF).

The drafter head normally shares the target's full ``lm_head`` (the proposer
swaps it in after load), so every draft step re-reads the whole head. With
``VLLM_DRAFT_VOCAB_FILE=<ids.pt>`` the drafter owns an N-row head holding the
checkpoint rows listed in ``ids`` and scatters its logits back into full-vocab
coordinates with ``-inf`` elsewhere. Target verification is untouched, so with
standard/block rejection (and greedy) the accepted output follows the target's
own distribution: a draft vocabulary only moves acceptance. Synthetic
verification accepts by a fixed rate instead of p/q and does not keep that
guarantee, so SpeculativeConfig refuses it (and adaptive verification) while
the knob is set.

The d2t offset map also feeds ``LocalArgmaxMixin.get_top_tokens``.

Pipeline parallelism: the Qwen4Exp drafter (and so its head) is built on the
LAST PP stage only (``Qwen4ExpMTP.__init__``: ``lm_head`` is a ``PPMissingLayer``
elsewhere), so the ids live on that stage's TP group and nowhere else.
"""

from __future__ import annotations

import hashlib
import io
import os
import re

import torch

ENV_DRAFT_VOCAB_FILE = "VLLM_DRAFT_VOCAB_FILE"
ENV_DRAFT_VOCAB_SHA256 = "VLLM_DRAFT_VOCAB_SHA256"
# Head rows must keep the GEMM tile-aligned on every TP shard.
DRAFT_VOCAB_ALIGN = 128

_PINNED_TABLE_FINGERPRINTS: dict[str, str] = {}


def draft_vocab_file() -> str | None:
    """The configured ids file, or None when the lever is off (the default)."""
    return os.environ.get(ENV_DRAFT_VOCAB_FILE) or None


def load_draft_vocab_ids(
    path: str, vocab_size: int, tp_size: int = 1
) -> torch.Tensor:
    """Read and hard-validate a draft vocabulary file.

    A malformed file must fail the boot; it must never degrade to the full head.
    If ``VLLM_DRAFT_VOCAB_SHA256`` is set, the file's sha256 must match it.
    Returns a sorted, unique, contiguous 1-D int64 CPU tensor.
    """
    expected = os.environ.get(ENV_DRAFT_VOCAB_SHA256, "").strip().lower()
    if expected:
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(
                f"{ENV_DRAFT_VOCAB_SHA256} must be 64 hex characters, got {expected!r}"
            )
        # Hash and load the same bytes (no window to swap the file between).
        with open(path, "rb") as fh:
            raw = fh.read()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(
                f"{path}: draft vocab file sha256 mismatch (expected {expected}, "
                f"got {actual}); refusing to load"
            )
        ids = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    else:
        ids = torch.load(path, map_location="cpu", weights_only=True)
    ids = validate_draft_vocab_ids(ids, vocab_size, tp_size, source=path)
    pin_draft_vocab_fingerprint(path, draft_vocab_fingerprint(ids))
    return ids


def pin_draft_vocab_fingerprint(path: str, fingerprint: str) -> str:
    """Pin ``fingerprint`` for ``path``, or refuse if a different table was
    already pinned in this process (the file was replaced mid-boot)."""
    pinned = _PINNED_TABLE_FINGERPRINTS.setdefault(path, fingerprint)
    if pinned != fingerprint:
        raise ValueError(
            f"{path}: draft vocab table changed during this boot (pinned "
            f"{pinned}, now {fingerprint}); restart with one stable table"
        )
    return pinned


def check_draft_vocab_agreement(fingerprints: list[str], source: str) -> str:
    """Every rank that serves the reduced head must hold the same table.
    ``fingerprints`` is indexed by rank in the drafter's group."""
    if len(set(fingerprints)) != 1:
        per_rank = ", ".join(f"rank {r}: {fp}" for r, fp in enumerate(fingerprints))
        raise ValueError(
            f"{source}: draft vocab table differs across ranks ({per_rank}); "
            "every rank must load the same table"
        )
    return fingerprints[0]


def draft_vocab_fingerprint(ids: torch.Tensor) -> str:
    """Table identity for the compile hash: N plus a sha256 of the int64 ids."""
    ids64 = ids.detach().to("cpu", torch.int64).contiguous()
    digest = hashlib.sha256(ids64.numpy().tobytes()).hexdigest()
    return f"n={int(ids64.numel())}:sha256={digest}"


def draft_vocab_table_factor(path: str) -> str:
    """The table fingerprint compile_factors() hashes for ``path``.

    The pinned one (recorded when the table was first read or loaded), else
    read from the file now and pinned (unvalidated: the model load validates
    and fails the boot). Never raises: an unreadable file yields a distinct
    marker, and the load then fails.
    """
    pinned = _PINNED_TABLE_FINGERPRINTS.get(path)
    if pinned is not None:
        return pinned
    try:
        ids = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - hashing must not crash the boot
        return f"unreadable:{type(exc).__name__}"
    if not isinstance(ids, torch.Tensor) or ids.ndim != 1:
        return f"invalid:{type(ids).__name__}"
    # Pinned now: the load that follows must read this same table.
    return pin_draft_vocab_fingerprint(path, draft_vocab_fingerprint(ids))


def validate_draft_vocab_ids(
    ids: object, vocab_size: int, tp_size: int = 1, source: str = "<ids>"
) -> torch.Tensor:
    if (
        not isinstance(ids, torch.Tensor)
        or ids.ndim != 1
        or ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(f"{source}: draft vocab must be a 1-D int32/int64 tensor")
    ids = ids.to(torch.int64).contiguous()
    n = int(ids.numel())
    if not 0 < n < vocab_size:
        raise ValueError(
            f"{source}: draft vocab size {n} must be in (0, {vocab_size})"
        )
    if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
        raise ValueError(f"{source}: draft vocab id outside [0, {vocab_size})")
    if not bool(torch.all(ids[1:] > ids[:-1])):
        raise ValueError(f"{source}: draft vocab ids must be sorted and unique")
    align = DRAFT_VOCAB_ALIGN * max(1, tp_size)
    if n % align:
        raise ValueError(
            f"{source}: draft vocab size {n} must be divisible by "
            f"{DRAFT_VOCAB_ALIGN}*TP ({align})"
        )
    return ids


def d2t_offsets(ids: torch.Tensor) -> torch.Tensor:
    """``target_id = k + d2t[k]`` map consumed by ``LocalArgmaxMixin``."""
    return ids - torch.arange(ids.numel(), dtype=torch.int64, device=ids.device)


def select_head_rows(
    weight: torch.Tensor, ids: torch.Tensor, vocab_size: int, hidden_size: int
) -> torch.Tensor:
    """Keep the draft-vocabulary rows of the checkpoint's full ``lm_head``."""
    if tuple(weight.shape) != (vocab_size, hidden_size):
        raise RuntimeError(
            f"draft vocab expected a full unsharded lm_head.weight "
            f"{(vocab_size, hidden_size)}, got {tuple(weight.shape)}"
        )
    return weight.index_select(0, ids.to(weight.device))


def scatter_to_full(
    logits: torch.Tensor,
    ids: torch.Tensor,
    vocab_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draft logits [T, N] -> full-vocab logits [T, V] (-inf outside the slice).

    q(x)=0 for out-of-slice tokens, which the rejection sampler reads as
    "never proposed"; the target still reaches them through the residual
    distribution.

    ``out`` is a persistent [rows >= T, V] buffer (allocated before capture);
    the result is its first T rows, refilled in full on every call, so the
    values are the same as with a fresh tensor and nothing is allocated inside
    a captured step. The result aliases ``out``: it is valid until the next
    call. Without ``out`` a fresh tensor is returned (eager callers, tests).
    """
    if out is None:
        full = logits.new_full((logits.shape[0], vocab_size), float("-inf"))
    else:
        full = out[: logits.shape[0]]
        full.fill_(float("-inf"))
    full.index_copy_(1, ids, logits)
    return full
