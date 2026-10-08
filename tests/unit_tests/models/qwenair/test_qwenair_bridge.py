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

"""QwenAir text provider and logical checkpoint conversion contracts."""

from fnmatch import fnmatchcase
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.qwenair import QwenAirModelProvider, QwenAirTextBridge, qwenair_provider
from megatron.bridge.models.qwenair.qwenair_bridge import qwenair_logical_state_names
from megatron.bridge.models.qwenair.qwenair_step import qwenair_loss


def _tiny_text_config() -> dict:
    return {
        "model_type": "qwen4_exp_text",
        "dtype": "float32",
        "vocab_size": 32,
        "eos_token_id": 4,
        "hidden_size": 16,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 32,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [2],
        "ple_embed_dim": 16,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 31,
        "make_ngram_vocab_size_divisible_by": 8,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 8,
        "indexer_budget": 8,
        "indexer_compress_ratio": 2,
        "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
    }


def test_canonical_text_provider_preserves_48_layer_schedule() -> None:
    config = _tiny_text_config()
    config.update(
        num_hidden_layers=48,
        hidden_size=2560,
        vocab_size=248320,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        num_experts=512,
        num_experts_per_tok=10,
        hc_count=4,
        hc_lowrank=320,
        ple_embed_dim=2560,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        indexer_n_heads=4,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        max_position_embeddings=262144,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
        layer_types=["full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(48)],
    )
    provider = QwenAirModelProvider.from_hf_config(config)
    assert provider.num_layers == 48
    assert provider.num_attention_heads == 24
    assert provider.kv_channels == 256
    assert provider.share_embeddings_and_output_weights is False
    assert provider.mtp_num_layers == 0
    assert provider.qwenair_text_config["layer_types"].count("full_attention") == 12
    assert provider.qwenair_text_config["ple_layer_ids"] == [2]
    with pytest.raises(ValueError, match="distributed table"):
        provider.provide()


def test_text_provider_rejects_multimodal_config() -> None:
    text = _tiny_text_config()
    with pytest.raises(ValueError, match="vision"):
        QwenAirModelProvider.from_hf_config({"model_type": "qwen4_exp", "text_config": text})


def test_auto_bridge_dispatch_preserves_text_geometry() -> None:
    class TextConfig(PretrainedConfig):
        model_type = "qwen4_exp_text"

    config = TextConfig(architectures=["Qwen4ExpForCausalLM"], **_tiny_text_config())
    bridge = AutoBridge.from_hf_config(config)
    provider = bridge.to_megatron_provider(load_weights=False)
    assert isinstance(provider, QwenAirModelProvider)
    exported = QwenAirTextBridge.megatron_to_hf_config(provider)
    assert exported["model_type"] == "qwen4_exp_text"
    assert exported["architectures"] == ["Qwen4ExpForCausalLM"]
    for field in ("layer_types", "ple_layer_ids", "hc_count", "indexer_budget", "num_experts_per_tok"):
        assert exported[field] == _tiny_text_config()[field]


def test_small_model_has_complete_identity_mapping_and_roundtrips() -> None:
    config = _tiny_text_config()
    provider = QwenAirModelProvider.from_hf_config(config)
    model = provider.provide()
    bridge = QwenAirTextBridge()
    registry = bridge.mapping_registry()
    state = model.state_dict()

    assert len(model.model.layers) == 4
    assert provider.qsa_backend == "dense"
    assert model.config.num_layers == provider.num_layers
    assert model.config.params_dtype == provider.params_dtype
    assert model.config.calculate_per_token_loss == provider.calculate_per_token_loss
    assert model.model.layers[3].self_attn.backend == "dense"
    assert [layer.layer_type for layer in model.model.layers] == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "qwen_sparse_attention",
    ]
    assert any(".ple." in name for name in state)
    assert any(".indexer." in name for name in state)
    assert any(".experts." in name for name in state)

    hf_state = {}
    for name, tensor in state.items():
        to_hf = registry.megatron_to_hf_lookup(name)
        from_hf = registry.hf_to_megatron_lookup(name)
        assert to_hf is not None, name
        assert from_hf is not None, name
        assert to_hf.hf_param == name
        assert from_hf.megatron_param == name
        loaded = from_hf.hf_to_megatron(tensor.clone(), SimpleNamespace(weight=tensor))
        exported = to_hf.megatron_to_hf(loaded, None)
        assert torch.equal(exported[name], tensor), name
        hf_state.update(exported)

    assert set(hf_state) == set(state)
    assert all(any(fnmatchcase(name, pattern) for name in state) for pattern in qwenair_logical_state_names())
    serialized = BytesIO()
    torch.save(hf_state, serialized)
    serialized.seek(0)
    loaded_hf_state = torch.load(serialized, weights_only=True)
    restored = provider.provide()
    restored_state = {
        name: registry.hf_to_megatron_lookup(name).hf_to_megatron(tensor, restored)
        for name, tensor in loaded_hf_state.items()
    }
    restored.load_state_dict(restored_state, strict=True)
    model.eval()
    restored.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(restored(input_ids).logits, model(input_ids).logits)


def test_provider_rejects_unimplemented_parallel_and_mtp() -> None:
    provider = QwenAirModelProvider.from_hf_config(_tiny_text_config())
    provider.tensor_model_parallel_size = 2
    with pytest.raises(NotImplementedError, match="parallel layout"):
        provider.provide()
    provider.tensor_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 2
    with pytest.raises(NotImplementedError, match="expert_tensor_parallel_size"):
        provider.provide()
    provider.expert_tensor_parallel_size = None
    provider.expert_model_parallel_size = 2
    with pytest.raises(RuntimeError, match="provide_distributed_model"):
        provider.provide()
    provider.expert_model_parallel_size = 1
    provider.mtp_num_layers = 1
    with pytest.raises(NotImplementedError, match="MTP"):
        provider.provide()
    provider.mtp_num_layers = 0
    provider.mtp_enabled = True
    with pytest.raises(NotImplementedError, match="MTP"):
        provider.provide()


def test_provider_rejects_partial_mcore_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwenair_provider, "QwenAirForCausalLM", object)
    with pytest.raises(RuntimeError, match="matching Megatron-Core QwenAir text API"):
        qwenair_provider._require_qwenair_mcore_api()


def test_te_backend_requires_fork_api_and_selects_mcore_path(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _tiny_text_config()
    config["indexer_compress_ratio"] = 4
    provider = QwenAirModelProvider.from_hf_config(config, qsa_backend="te_reference")
    assert provider.qsa_backend == "te_reference"
    monkeypatch.setattr(qwenair_provider.importlib, "import_module", lambda _name: SimpleNamespace())
    with pytest.raises(ImportError, match="TransformerEngine@c4f14012"):
        provider.provide()

    def qsa_reference(query, key, value, selected_key_blocks, *, scale=None):
        return query

    monkeypatch.setattr(
        qwenair_provider.importlib,
        "import_module",
        lambda _name: SimpleNamespace(qsa_block_sparse_attention=qsa_reference),
    )
    model = provider.provide()
    assert model.model.layers[3].self_attn.backend == "te_reference"


def test_indexed_te_backend_checks_its_own_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = QwenAirModelProvider.from_hf_config(_tiny_text_config(), qsa_backend="te_indexed_sdpa")

    def qsa_reference(query, key, value, selected_key_blocks, *, scale=None):
        return query

    monkeypatch.setattr(
        qwenair_provider.importlib,
        "import_module",
        lambda _name: SimpleNamespace(qsa_block_sparse_attention=qsa_reference),
    )
    with pytest.raises(ImportError, match="te_indexed_sdpa"):
        provider.provide()

    monkeypatch.setattr(
        qwenair_provider.importlib,
        "import_module",
        lambda _name: SimpleNamespace(qsa_indexed_sdpa_attention=qsa_reference),
    )
    model = provider.provide()
    assert model.model.layers[3].self_attn.backend == "te_indexed_sdpa"


def test_qwenair_loss_preserves_scaled_backward_and_reports_components() -> None:
    scaled = torch.tensor(3.5, requires_grad=True)
    loss, num_tokens, metrics = qwenair_loss(
        scaled,
        num_tokens=torch.tensor(3, dtype=torch.int),
        reporting_loss_sum=torch.tensor(6.0),
        router_aux_loss_sum=torch.tensor(0.75),
    )
    loss.backward()

    assert scaled.grad.item() == 1.0
    assert num_tokens.item() == 3
    assert torch.equal(metrics["lm loss"], torch.tensor([6.0, 3.0]))
    assert torch.equal(metrics["router aux loss"], torch.tensor([0.75, 3.0]))
