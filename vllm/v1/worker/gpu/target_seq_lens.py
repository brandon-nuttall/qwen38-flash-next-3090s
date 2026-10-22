# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Exact target (verify) seq_lens on the CPU without a blocking copy
(VLLM_TARGET_SEQ_LENS_ASYNC).

Pure numpy/torch, no vllm imports, so the CPU unit tests can load this file alone.

Problem. The FlashInfer verify build needs the per-row target lengths on the CPU (page counts,
last-page lengths and the plan are derived from them). The GPU computes them in
``prepare_pos_seq_lens`` as ``num_computed_tokens[slot] + query_len``; under async scheduling the
CPU mirror of ``num_computed_tokens`` is optimistic (it does not know how many drafts the previous
verify rejected), so the build calls ``seq_lens.cpu()`` -- a blocking device->host copy that waits
for EVERYTHING queued before it, i.e. the previous step's verify forward AND all K draft steps.

Fix. The only unknown, ``num_computed_tokens`` on the GPU, is final as soon as the previous step's
``post_update`` (rejection bookkeeping) has run -- before any draft step. So right after
``post_update`` we enqueue a pinned, non-blocking D2H copy of the whole ``num_computed_tokens``
array and record an event (``TargetSeqLensSnapshot.capture``). The next verify build waits on that
event only (the drafts queued behind it keep the GPU busy) and computes

    seq_len[i] = snapshot[idx_mapping[i]] + (query_start_loc[i + 1] - query_start_loc[i])
    seq_len[i] = 0 for padded rows

which is exactly what ``_prepare_pos_seq_lens_kernel`` writes from the same inputs (same
``num_computed_tokens`` values, same CPU ``query_start_loc`` the GPU copy was made from). The
FlashInfer build then runs on identical CPU inputs => identical page table, identical plan bytes,
identical kernel schedule and reduction order. No planner assumption is involved.

Validity. The snapshot is valid only while nothing has written ``num_computed_tokens`` on the GPU
since the capture. The writers are exactly: ``_post_update_kernel`` (postprocess_sampled),
``_post_update_num_computed_tokens_kernel`` (postprocess_num_computed_tokens) and the staged
writes of ``RequestState.apply_staged_writes`` (new requests). Each bumps ``RequestState.
nct_epoch``; a snapshot carries the epoch it was taken at, and a handle is only issued when the
epoch is unchanged. A handle additionally pins its slot's capture serial, so a later capture
into the same pinned slot invalidates it instead of being read.
"""

from __future__ import annotations

import numpy as np
import torch


def exact_target_seq_lens(
    nct_snapshot: np.ndarray,
    idx_mapping_np: np.ndarray,
    query_start_loc_np: np.ndarray,
    num_reqs: int,
    num_rows: int,
) -> np.ndarray:
    """CPU replica of _prepare_pos_seq_lens_kernel's seq_lens (input_batch.py): int32
    [num_rows], real rows nct[slot] + query_len, padded rows 0."""
    assert num_rows >= num_reqs
    out = np.zeros(num_rows, dtype=np.int32)
    if num_reqs:
        q = query_start_loc_np[1 : num_reqs + 1].astype(np.int64) - query_start_loc_np[
            :num_reqs
        ].astype(np.int64)
        s = nct_snapshot[idx_mapping_np[:num_reqs]].astype(np.int64) + q
        out[:num_reqs] = s.astype(np.int32)
    return out


class ExactSeqLensHandle:
    """Lazy exact target seq_lens for one step. ``get`` waits on the snapshot event (the
    previous step's post_update + copy), never on the draft steps queued after it."""

    __slots__ = (
        "_owner", "_slot", "_serial", "_idx", "_qsl", "_num_reqs", "_result",
        "_failed",
    )

    def __init__(self, owner, slot, serial, idx_mapping_np, query_start_loc_np, num_reqs):
        self._owner = owner
        self._slot = slot
        self._serial = serial
        self._idx = np.array(idx_mapping_np[:num_reqs], dtype=np.int64, copy=True)
        self._qsl = np.array(query_start_loc_np[: num_reqs + 1], dtype=np.int64, copy=True)
        self._num_reqs = num_reqs
        self._result: np.ndarray | None = None
        self._failed = False

    @property
    def num_reqs(self) -> int:
        return self._num_reqs

    def get(self, num_rows: int) -> torch.Tensor | None:
        """int32 CPU tensor [num_rows] (a fresh tensor per call), or None if the snapshot
        is no longer readable (caller syncs instead)."""
        if self._failed or num_rows < self._num_reqs:
            return None
        if self._result is None:
            o = self._owner
            if o.serials[self._slot] != self._serial:
                self._failed = True
                return None
            o.wait(self._slot)
            # Re-check after the wait: the serial only changes on the engine thread, but keep
            # the read strictly behind the check.
            if o.serials[self._slot] != self._serial:
                self._failed = True
                return None
            self._result = exact_target_seq_lens(
                o.host_view(self._slot), self._idx, self._qsl, self._num_reqs,
                self._num_reqs,
            )
        out = np.zeros(num_rows, dtype=np.int32)
        out[: self._num_reqs] = self._result
        return torch.from_numpy(out)


class TargetSeqLensSnapshot:
    """Double-buffered pinned snapshots of RequestState.num_computed_tokens.gpu."""

    def __init__(self, max_num_reqs: int, num_slots: int = 2, device=None,
                 pin_memory: bool = True):
        self.max_num_reqs = max_num_reqs
        self.bufs = [
            torch.zeros(max_num_reqs, dtype=torch.int32, pin_memory=pin_memory)
            for _ in range(num_slots)
        ]
        self.events = [None] * num_slots  # torch.cuda.Event per slot, created lazily
        self.serials = [0] * num_slots
        self._next = 0
        self._serial = 0
        self.latest: tuple[int, int] | None = None  # (slot, epoch)
        # counters
        self.captures = 0
        self.issued = 0
        self.stale_epoch = 0

    def capture(self, nct_gpu: torch.Tensor, epoch: int) -> None:
        """Enqueue the D2H copy on the current stream (right after post_update)."""
        slot = self._next
        self._next = (slot + 1) % len(self.bufs)
        self._serial += 1
        self.serials[slot] = self._serial
        n = min(self.max_num_reqs, nct_gpu.shape[0])
        self.bufs[slot][:n].copy_(nct_gpu[:n], non_blocking=True)
        ev = self.events[slot]
        if ev is None and nct_gpu.is_cuda:
            ev = self.events[slot] = torch.cuda.Event()
        if ev is not None:
            ev.record()
        self.latest = (slot, epoch)
        self.captures += 1

    def invalidate(self) -> None:
        self.latest = None

    def handle(self, epoch_now: int, idx_mapping_np: np.ndarray,
               query_start_loc_np: np.ndarray, num_reqs: int) -> ExactSeqLensHandle | None:
        if self.latest is None:
            return None
        slot, epoch = self.latest
        if epoch != epoch_now:
            self.stale_epoch += 1
            return None
        if num_reqs <= 0 or int(np.max(idx_mapping_np[:num_reqs])) >= self.max_num_reqs:
            return None
        self.issued += 1
        return ExactSeqLensHandle(
            self, slot, self.serials[slot], idx_mapping_np, query_start_loc_np, num_reqs
        )

    def wait(self, slot: int) -> None:
        ev = self.events[slot]
        if ev is not None:
            ev.synchronize()

    def host_view(self, slot: int) -> np.ndarray:
        return self.bufs[slot].numpy()
