# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Truncate the MTP draft distribution to its own top-k/top-p set.

Rejection sampling is exact for ANY proposal q as long as the verifier divides
by the q the draft was actually drawn from. The target applies the request's
top-k/top-p before verification, so a draft token outside the target's nucleus
is always rejected. The target's kept set is unknown at draft time, so the
draft is drawn from q' = q renormalised on the draft's own top-k/top-p set
(computed on the temperature-scaled logits, exactly as the target orders it).

The truncated logits are written to the draft-logits cache too (the caller
hands them to `gumbel_sample(logits_cache=...)` WITHOUT a separate
`logits_cache_source`), so the verifier's softmax(cache / T) is q' and the
output distribution stays the target's.

Lifecycle (why this file also stages the per-request values)
------------------------------------------------------------
The per-request top-k / top-p live in `SamplingStates` as `UvaBackedTensor`s
whose `.gpu` handle is re-pointed at a different pool buffer by EVERY
`apply_staged_writes()` (every step). A handle cached once is therefore stale on
the next step, and under FULL CUDA graphs `sample_draft` is captured, so a
Python re-read at replay time is useless too. The draft sampler reads
speculator-owned, persistent, fixed-address `[max_num_reqs]` buffers, and the
model runner refreshes them eagerly, outside any captured region, immediately
before every `propose()` through `stage_draft_top_k_top_p` (reached via
`DraftModelSpeculator.prepare_draft_sampling`), using only the PUBLIC
`SamplingStates.get_top_k_top_p`. Nothing here retains a `.gpu` handle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    apply_top_k_top_p_pytorch,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.sample.states import SamplingStates

FLAG_NAME = "VLLM_SPEC_DRAFT_UNDER_TARGET_TOPK"


def check_draft_under_topk_support(
    *,
    speculator_name: str,
    supported: bool,
    has_watermarker: bool,
    rejection_sample_method: str,
) -> None:
    """Refuse (raise) an unsupported combination while the flag is ON.

    Called at speculator init only when the flag is requested. Raising rather
    than warning is deliberate: a lever that silently does nothing makes every
    measurement of it meaningless.
    """
    if not supported:
        raise ValueError(
            f"{FLAG_NAME}=1 is not supported by {speculator_name}: only "
            "speculators whose draft sampling reads the runner-prepared "
            "top-k/top-p buffers (supports_draft_under_topk) are audited."
        )
    if has_watermarker:
        raise ValueError(f"{FLAG_NAME}=1 is not compatible with draft watermarking.")
    if rejection_sample_method == "synthetic":
        raise ValueError(
            f"{FLAG_NAME}=1 is not compatible with synthetic verification: it "
            "never reads the draft distribution, so truncating it only costs "
            "time."
        )


def check_draft_under_topk_sampler(*, custom_sampler: bool, watermark: bool) -> None:
    """Refuse (raise) a custom or watermarking target sampler while ON."""
    if custom_sampler:
        raise ValueError(
            f"{FLAG_NAME}=1 is only audited with the stock sampler and "
            "RejectionSampler, not a model-state custom sampler."
        )
    if watermark:
        raise ValueError(f"{FLAG_NAME}=1 is not compatible with watermarking.")


def stage_draft_top_k_top_p(
    draft_top_k: torch.Tensor,  # [max_num_reqs] int32, persistent
    draft_top_p: torch.Tensor,  # [max_num_reqs] float32, persistent
    sampling_states: SamplingStates,
    idx_mapping: torch.Tensor,  # [>= num_reqs] request-state slots
    idx_mapping_np: np.ndarray,  # [>= num_reqs] same, host mirror
    num_reqs: int,
    *,
    dummy_run: bool = False,
) -> None:
    """Refresh the persistent draft buffers with THIS step's per-request values.

    Eager and graph-external: it must run on the proposal stream before the
    draft graphs replay. The buffers are indexed by request-state slot (the
    same convention as the speculator's `temperature` / `seeds` and the draft
    logits cache); the accessor returns batch order, so it is scattered back by
    slot. Slots not in the batch keep the no-truncation sentinels. Writes go
    into the SAME tensors every call, so their `data_ptr` never changes. No
    host sync: the accessor's decisions use the host mirrors, everything else
    is device fills and an `index_copy_`.
    """
    max_num_reqs = draft_top_k.shape[0]
    if draft_top_p.shape != draft_top_k.shape:
        raise ValueError(
            f"draft top-k {tuple(draft_top_k.shape)} / top-p "
            f"{tuple(draft_top_p.shape)} buffers differ in shape"
        )
    if sampling_states.max_num_reqs != max_num_reqs:
        raise ValueError(
            f"draft buffers are sized for {max_num_reqs} requests but the "
            f"sampler holds {sampling_states.max_num_reqs}"
        )

    draft_top_k.fill_(sampling_states.vocab_size)
    draft_top_p.fill_(1.0)
    if dummy_run or num_reqs == 0:
        return

    slots = idx_mapping[:num_reqs]
    top_k, top_p = sampling_states.get_top_k_top_p(slots, idx_mapping_np[:num_reqs])
    dst = slots.long()
    if top_k is not None:
        if top_k.shape != (num_reqs,):
            raise ValueError(f"top-k {tuple(top_k.shape)} != ({num_reqs},)")
        draft_top_k.index_copy_(0, dst, top_k)
    if top_p is not None:
        if top_p.shape != (num_reqs,):
            raise ValueError(f"top-p {tuple(top_p.shape)} != ({num_reqs},)")
        draft_top_p.index_copy_(0, dst, top_p)


def truncate_draft_logits_top_k_top_p(
    logits: torch.Tensor,  # [num_reqs, vocab] pre-temperature draft logits
    idx_mapping: torch.Tensor,  # [num_reqs] request-state index (-1 = padding)
    temperature: torch.Tensor,  # [max_num_reqs]
    top_k: torch.Tensor,  # [max_num_reqs]; >= vocab = disabled
    top_p: torch.Tensor,  # [max_num_reqs]; 1.0 = disabled
) -> torch.Tensor:
    """Return `logits` with tokens outside each row's top-k/top-p set at -inf.

    * Pre-temperature in, pre-temperature out (the cache is pre-temperature);
      the set is selected on logits / T, matching the target's order.
    * ONLY rows with `valid & ~greedy & (k < V | p < 1)` may change. The
      pytorch top-p path masks fp32-underflow tail tokens even at p == 1.0, so
      "a no-op at the sentinels" is not something the backend guarantees; the
      row guard makes untouched rows bit-identical by construction (this also
      covers padding rows, idx_mapping == -1, and greedy rows).
    * `top_k` may carry the sampler's vocab size, which can exceed the draft's
      logits width: it is clamped to the width.
    * Static shapes, no host sync (safe inside a captured graph). The backend
      is the target sampler's `apply_top_k_top_p` dispatch on CUDA (Triton);
      CPU tensors use the pytorch reference (the dispatch would otherwise try
      Triton on a CPU tensor wherever Triton is installed).
    """
    vocab = logits.shape[-1]
    valid = idx_mapping >= 0
    idx = idx_mapping.long().clamp_min(0)  # padded rows are ignored downstream
    temp = temperature[idx].to(torch.float32)
    k = top_k[idx].clamp_max(vocab)
    p = top_p[idx]
    greedy = temp == 0.0
    active = valid & ~greedy & ((k < vocab) | (p < 1.0))
    scale = torch.where(greedy | (temp == 1.0), torch.ones_like(temp), temp)
    scaled = logits.to(torch.float32) / scale.unsqueeze(-1)  # fresh tensor
    if logits.is_cuda:
        truncated = apply_top_k_top_p(scaled, k, p)
    else:
        truncated = apply_top_k_top_p_pytorch(scaled, k, p, allow_cpu_sync=False)
    drop = torch.isneginf(truncated) & active.unsqueeze(-1)
    return logits.masked_fill(drop, float("-inf"))
