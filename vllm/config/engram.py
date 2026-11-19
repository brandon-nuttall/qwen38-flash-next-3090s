# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.

from typing import TYPE_CHECKING, Self

from pydantic import Field, model_validator

from vllm import flashnext_family
from vllm.config.utils import config, get_hash_factors, hash_factors

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_FLASH_ENGRAM_ARCHITECTURES = {
    "HYV4FlashForCausalLM",
    "HYV4FlashVLForConditionalGeneration",
}

_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
    **dict.fromkeys(flashnext_family.target_archs(), "ple_layer_ids"),
    "HYV4FlashForCausalLM": "engram_layer_ids",
    "HYV4FlashVLForConditionalGeneration": "engram_layer_ids",
}


def model_has_engram_layers(model_config: "ModelConfig | None") -> bool:
    """Whether the model carries n-gram embedding layers."""
    if model_config is None:
        return False
    field = _NGRAM_LAYER_FIELDS.get(model_config.architecture)
    if field is None:
        return False
    return bool(getattr(model_config.hf_text_config, field, None))


def uses_ple_pull_transport(model_config: "ModelConfig | None") -> bool:
    """Whether this model's n-gram table is served by the PLE pull transport.

    That is the flash-next family exactly: the field naming its layers is
    ``ple_layer_ids``. Only these models tie ``cpu_offload`` to the pull
    transport; other n-gram models (HY4 Flash) keep upstream's pinned-host
    semantics."""
    return (
        model_config is not None
        and _NGRAM_LAYER_FIELDS.get(model_config.architecture) == "ple_layer_ids"
    )


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool | None = None
    """Store embedding weights in pinned CPU memory for UVA lookup.
    ``None`` (unset) resolves per family in `verify_model_config`: False on
    the flash-next family, where the PLE pull transport owns the table (an
    explicit True is refused there — see
    `vllm.v1.e2_guards.verify_engram_against_pull`); True everywhere else,
    matching upstream's host-RAM defaults. Explicit booleans are honored."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    dp_shared_memory: bool | None = None
    """Share CPU-offloaded embedding weights between co-located
    DP replicas. Each node stores one copy of every TP shard, reducing host
    memory without per-step Engram DP collectives. Requires sufficient
    /dev/shm capacity and a shared IPC namespace. Defaults to enabled when the
    model and other settings allow it; Flash uses private replicas. Falls back
    to per-replica tables when DP replicas are not co-located on one node or
    /dev/shm cannot hold them."""

    use_thp: bool = False
    """Back private CPU-offloaded tables with transparent huge pages (best
    effort, falls back to ordinary pinned pages). Prefaulting the tables at
    startup takes longer. Requires cpu_offload without dp_shared_memory.
    Always on for private tables on ROCm."""

    @model_validator(mode="after")
    def _validate_shared_memory(self) -> Self:
        # cpu_offload may still be None here (it resolves per family in
        # verify_model_config, once the model is known); only an explicit
        # False contradicts a host-table option at construction time.
        if self.dp_shared_memory and self.cpu_offload is False:
            raise ValueError("dp_shared_memory requires cpu_offload=True")
        if self.use_thp and (self.cpu_offload is False or self.dp_shared_memory):
            raise ValueError(
                "use_thp requires cpu_offload=True and dp_shared_memory=False"
            )
        return self

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings and non-empty n-gram layer ids."
            )
        if model_config.architecture in _FLASH_ENGRAM_ARCHITECTURES:
            for option in ("dp_shared_memory", "embedding_across_dp"):
                if getattr(self, option):
                    raise ValueError(f"Flash Engram does not support {option}")
        # cpu_offload resolves per family: None takes the family default
        # (False on the flash-next family, whose PLE pull transport owns the
        # table; True everywhere else, upstream's host-RAM tables), an explicit
        # True beside the pull transport is refused, and an explicit False
        # (GPU tables) is honored.
        if self.cpu_offload is None:
            self.cpu_offload = not uses_ple_pull_transport(model_config)
        elif self.cpu_offload and uses_ple_pull_transport(model_config):
            raise ValueError(
                "cpu_offload=True beside the PLE pull transport: both "
                "transports would own the table. The flash-next family "
                "serves cpu_offload=False only."
            )
        # Re-check the host-table options against the resolved value (the
        # construction-time pass cannot see through a None default).
        if self.dp_shared_memory and not self.cpu_offload:
            raise ValueError("dp_shared_memory requires cpu_offload=True")
        if self.use_thp and (not self.cpu_offload or self.dp_shared_memory):
            raise ValueError(
                "use_thp requires cpu_offload=True and dp_shared_memory=False"
            )

    def resolve_dp_shared_memory(
        self,
        parallel_config: "ParallelConfig",
        model_config: "ModelConfig | None" = None,
    ) -> None:
        """Share host tables by default wherever the model supports them."""
        if self.dp_shared_memory is None:
            self.dp_shared_memory = (
                self.cpu_offload
                and not self.use_thp
                and parallel_config.data_parallel_size > 1
                and not parallel_config.enable_elastic_ep
                and (
                    model_config is None
                    or model_config.architecture not in _FLASH_ENGRAM_ARCHITECTURES
                )
            )

    def verify_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject unsupported embedding parallel topologies."""
        if self.dp_shared_memory:
            if parallel_config.data_parallel_size <= 1:
                raise ValueError("dp_shared_memory requires data_parallel_size > 1.")
            if parallel_config.enable_elastic_ep:
                raise ValueError("dp_shared_memory is not supported with elastic EP.")
        if (
            self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
            and parallel_config.enable_elastic_ep
        ):
            raise ValueError(
                "Engram embedding_across_dp is not supported with elastic EP yet."
            )

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
