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

"""Provider for the Qwen4-Exp text model implemented in Megatron-Core."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, fields, is_dataclass
from inspect import signature
from typing import Any, Literal, Mapping

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig as MCoreTransformerConfig


try:
    from megatron.core.models.qwenair import QwenAirForCausalLM, QwenAirTextConfig
except ModuleNotFoundError as exc:
    if exc.name == "megatron.core.models.qwenair":
        raise ImportError(
            "QwenAir Bridge requires the matching Megatron-Core QwenAir text API; "
            "update 3rdparty/Megatron-LM to the QwenAir implementation commit"
        ) from exc
    raise

from megatron.bridge.models.gpt_provider import GPTModelProvider


def _require_qwenair_mcore_api() -> None:
    """Fail clearly if Bridge is paired with an older or partial MCore checkout."""
    config_fields = {item.name for item in fields(QwenAirTextConfig)} if is_dataclass(QwenAirTextConfig) else set()
    required_fields = {
        "layer_types",
        "ple_layer_ids",
        "hc_count",
        "indexer_budget",
        "num_experts",
        "mtp_num_hidden_layers",
        "qsa_backend",
    }
    forward = getattr(QwenAirForCausalLM, "forward", None)
    forward_fields = set(signature(forward).parameters) if callable(forward) else set()
    init_fields = set(signature(QwenAirForCausalLM.__init__).parameters)
    required_forward = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "labels",
        "labels_are_shifted",
        "output_router_logits",
        "enable_mtp",
    }
    if (
        not callable(getattr(QwenAirTextConfig, "from_hf_dict", None))
        or not required_fields <= config_fields
        or not isinstance(QwenAirForCausalLM, type)
        or not issubclass(QwenAirForCausalLM, MegatronModule)
        or not required_forward <= forward_fields
        or "pg_collection" not in init_fields
    ):
        raise RuntimeError(
            "QwenAir Bridge requires the matching Megatron-Core QwenAir text API; "
            "update 3rdparty/Megatron-LM to the QwenAir implementation commit"
        )


def _require_te_qsa_api(backend: str) -> None:
    """Check the QSA callable added by the QwenAir Transformer Engine fork."""
    requirements = {
        "te_reference": (
            "qsa_block_sparse_attention",
            "shangxiaokang/TransformerEngine@c4f14012b02b9162b585794a4abb5e6946b8835f",
            set(),
        ),
        "te_indexed_sdpa": (
            "qsa_indexed_sdpa_attention",
            "shangxiaokang/TransformerEngine@c4f14012b02b9162b585794a4abb5e6946b8835f",
            set(),
        ),
        "te_triton": (
            "qsa_triton_attention",
            "shangxiaokang/TransformerEngine@3250741db1e06acd638da6124bc193405565f771",
            {"validate_indices"},
        ),
    }
    function_name, required_commit, backend_parameters = requirements[backend]
    try:
        te = importlib.import_module("transformer_engine.pytorch")
    except (ImportError, OSError) as exc:
        raise ImportError(f"qsa_backend={backend!r} requires {required_commit}") from exc
    qsa = getattr(te, function_name, None)
    expected = {"query", "key", "value", "selected_key_blocks", "scale", *backend_parameters}
    try:
        available = set(signature(qsa).parameters) if callable(qsa) else set()
    except (TypeError, ValueError):
        available = set()
    if not expected <= available:
        raise ImportError(f"qsa_backend={backend!r} requires the QSA API from {required_commit}")


def _config_dict(config: Any) -> dict[str, Any]:
    """Convert a Hugging Face config or mapping to a plain dictionary."""
    if isinstance(config, Mapping):
        return dict(config)
    if callable(getattr(config, "to_dict", None)):
        return dict(config.to_dict())
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError("QwenAir configuration must be a mapping or Hugging Face config")


def _text_config_dict(config: Any) -> dict[str, Any]:
    config_dict = _config_dict(config)
    if config_dict.get("model_type") == "qwen4_exp" or any(
        name in config_dict for name in ("vision_config", "image_token_id", "text_config")
    ):
        raise ValueError("qwen4_exp includes vision; use an approved multimodal bridge for that model")
    if config_dict.get("model_type") not in (None, "qwen4_exp_text"):
        raise ValueError("QwenAir text provider requires model_type=qwen4_exp_text")
    return config_dict


def _parameter_dtype(text: Mapping[str, Any]) -> torch.dtype:
    dtype = text.get("dtype", text.get("torch_dtype", "bfloat16"))
    if isinstance(dtype, torch.dtype):
        return dtype
    supported = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype not in supported:
        raise ValueError(f"Unsupported QwenAir text dtype: {dtype!r}")
    return supported[dtype]


@dataclass
class QwenAirModelProvider(GPTModelProvider):
    """Construct the text-only model from a Qwen4-Exp text config.

    Expert parallelism and expert-data-parallel replicas use the explicit
    process groups installed by :class:`ModelProviderMixin`. TP, PP, CP and
    expert TP remain fail-closed. The checkpoint's MTP training objective and
    the QSA indexer's separate objective are also unavailable, so a successful
    LM forward/backward run is not a complete pretraining acceptance result.
    """

    qwenair_text_config: dict[str, Any] = field(default_factory=dict)
    qsa_backend: Literal["dense", "te_reference", "te_indexed_sdpa", "te_triton"] = "dense"
    mtp_num_layers: int | None = 0

    @classmethod
    def from_hf_config(
        cls,
        hf_config: Any,
        *,
        qsa_backend: Literal["dense", "te_reference", "te_indexed_sdpa", "te_triton"] = "dense",
    ) -> QwenAirModelProvider:
        """Build a provider from a standalone ``qwen4_exp_text`` config."""
        _require_qwenair_mcore_api()
        text = _text_config_dict(hf_config)
        config = QwenAirTextConfig.from_hf_dict({**text, "qsa_backend": qsa_backend})
        dtype = _parameter_dtype(text)
        return cls(
            qwenair_text_config=text,
            qsa_backend=qsa_backend,
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            kv_channels=config.head_dim,
            ffn_hidden_size=config.moe_intermediate_size,
            moe_ffn_hidden_size=config.moe_intermediate_size,
            num_moe_experts=config.num_experts,
            moe_router_topk=config.num_experts_per_tok,
            vocab_size=config.vocab_size,
            seq_length=config.max_position_embeddings,
            layernorm_epsilon=config.rms_norm_eps,
            init_method_std=config.initializer_range,
            share_embeddings_and_output_weights=config.tie_word_embeddings,
            normalization="RMSNorm",
            layernorm_zero_centered_gamma=True,
            position_embedding_type="rope",
            rotary_base=int(config.rope_theta),
            rotary_percent=config.partial_rotary_factor,
            add_bias_linear=False,
            add_qkv_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            hidden_dropout=0.0,
            params_dtype=dtype,
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
        )

    def provide(
        self,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> QwenAirForCausalLM:
        """Instantiate the text model with explicit EP and expert-DP groups."""
        _require_qwenair_mcore_api()
        if not self.qwenair_text_config:
            raise ValueError("qwenair_text_config must be supplied")
        unsupported_fields = (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
        )
        unsupported = {name: getattr(self, name) for name in unsupported_fields if getattr(self, name) != 1}
        # MCore resolves an unset ETP size to TP size, which is one above.
        if self.expert_tensor_parallel_size not in (None, 1):
            unsupported["expert_tensor_parallel_size"] = self.expert_tensor_parallel_size
        if self.sequence_parallel:
            unsupported["sequence_parallel"] = True
        if self.virtual_pipeline_model_parallel_size is not None:
            unsupported["virtual_pipeline_model_parallel_size"] = self.virtual_pipeline_model_parallel_size
        if vp_stage is not None:
            unsupported["vp_stage"] = vp_stage
        if unsupported:
            raise NotImplementedError(f"QwenAir does not yet support this parallel layout: {unsupported}")
        if pre_process is False or post_process is False:
            raise NotImplementedError("QwenAir text reference does not support pipeline stage splits")
        if self.mtp_enabled or self.mtp_num_layers not in (None, 0):
            raise NotImplementedError("QwenAir MTP training contract is unavailable")
        if self.qsa_backend in ("te_reference", "te_indexed_sdpa", "te_triton"):
            _require_te_qsa_api(self.qsa_backend)
        config = QwenAirTextConfig.from_hf_dict({**self.qwenair_text_config, "qsa_backend": self.qsa_backend})

        # QwenAirTextConfig owns the model-specific geometry. MCore DDP and the
        # pipeline schedule also read TransformerConfig runtime fields from
        # ``model.config``. Copy those fields from the finalized Bridge provider
        # so there is one duck-typed config object rather than two divergent
        # sources of parallel and precision policy.
        for config_field in fields(MCoreTransformerConfig):
            setattr(config, config_field.name, getattr(self, config_field.name))
        config.expert_model_parallel_size = self.expert_model_parallel_size
        config.expert_tensor_parallel_size = self.expert_tensor_parallel_size or 1
        config.validate()

        pg_collection = self._pg_collection
        if config.expert_model_parallel_size > 1 and pg_collection is None:
            raise RuntimeError(
                "QwenAir expert parallelism requires provide_distributed_model() "
                "or an explicit provider._pg_collection"
            )
        return QwenAirForCausalLM(config, pg_collection=pg_collection)
