# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logical text checkpoint mappings for the QwenAir reference model.

These mappings are exact identity mappings for a single unsharded text model.
They are deliberately not a converter for the unpublished physical checkpoint:
its PLE table shards, expert packing and MTP ownership need observed headers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from megatron.core.models.qwenair import QwenAirForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import DirectMapping
from megatron.bridge.models.qwenair.qwenair_provider import QwenAirModelProvider


_HC_WEIGHTS = (
    "hc_norm.weight",
    "input_mix_weight_down.weight",
    "input_mix_weight_up.weight",
    "block_inject_weight.weight",
)
_GDN_WEIGHTS = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "conv1d.weight",
    "A_log",
    "dt_bias",
    "norm.weight",
    "out_proj.weight",
)
_QSA_WEIGHTS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "q_norm.weight",
    "k_norm.weight",
    "indexer.index_qk_proj.weight",
    "indexer.q_layernorm.weight",
    "indexer.k_layernorm.weight",
)
_MOE_WEIGHTS = (
    "gate.weight",
    "experts.gate_up_proj",
    "experts.down_proj",
    "shared_expert.gate_proj.weight",
    "shared_expert.up_proj.weight",
    "shared_expert.down_proj.weight",
    "shared_expert_gate.weight",
)
_PLE_STATE = (
    "ple_embedding.ngram_embedding.weight",
    "ple_embedding.layer_multipliers",
    "ple_embedding.ngram_heads_vocab_sizes",
    "ple_embedding.ngram_heads_offsets",
    "key_proj.weight",
    "value_proj.weight",
    "norm_key.weight",
    "norm_query.weight",
    "norm_conv.weight",
    "conv1d.weight",
)


def qwenair_logical_state_names() -> tuple[str, ...]:
    """Return the supported single-rank HF/MCore logical state patterns."""
    names = ["model.embed_tokens.weight", "lm_head.weight"]
    names.extend(f"model.hyper_connection_mixer.{name}" for name in _HC_WEIGHTS[:-1])
    for block in ("attn_hyper_connection", "mlp_hyper_connection"):
        names.extend(f"model.layers.*.{block}.{name}" for name in _HC_WEIGHTS)
    names.extend(f"model.layers.*.linear_attn.{name}" for name in _GDN_WEIGHTS)
    names.extend(f"model.layers.*.self_attn.{name}" for name in _QSA_WEIGHTS)
    names.extend(f"model.layers.*.mlp.{name}" for name in _MOE_WEIGHTS)
    names.extend(f"model.layers.*.ple.{name}" for name in _PLE_STATE)
    return tuple(names)


@MegatronModelBridge.register_bridge(
    source="Qwen4ExpForCausalLM",
    target=QwenAirForCausalLM,
    provider=QwenAirModelProvider,
    model_type="qwen4_exp_text",
)
class QwenAirTextBridge(MegatronModelBridge):
    """Map Qwen4-Exp text reference tensors to identically named MCore tensors.

    The frozen HF implementation needs Transformers 5.16.dev, while the pinned
    Bridge runtime requires <=5.15. Use config-only construction and logical
    tensor fixtures until the two isolated runtimes and physical checkpoint
    manifest are available. This class does not register top-level multimodal
    ``qwen4_exp`` as text-only.
    """

    MODEL_CONFIG_CLASS = None
    SUPPORTS_HF_PRETRAINED_EXPORT = False

    def provider_bridge(self, hf_pretrained: object) -> QwenAirModelProvider:
        """Build the bounded QwenAir text provider from an HF config wrapper."""
        config = getattr(hf_pretrained, "config", None)
        if config is None:
            raise TypeError("QwenAir text bridge requires a source with a config")
        return QwenAirModelProvider.from_hf_config(config)

    @classmethod
    def megatron_to_hf_config(cls, provider: QwenAirModelProvider) -> dict[str, Any]:
        """Preserve QwenAir-only geometry in a config-only roundtrip."""
        if not provider.qwenair_text_config:
            raise ValueError("QwenAir provider has no source text configuration")
        config = deepcopy(provider.qwenair_text_config)
        config.pop("qsa_backend", None)
        config["model_type"] = "qwen4_exp_text"
        config["architectures"] = ["Qwen4ExpForCausalLM"]
        return config

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map the unsharded logical state tensors in both directions."""
        return MegatronMappingRegistry(*(DirectMapping(name, name) for name in qwenair_logical_state_names()))
