# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Megatron Bridge Recipe Configurations

This module exposes all recipe configurations from all model families.
"""

from importlib import import_module as _import_module

from megatron.bridge.recipes.bagel import *
from megatron.bridge.recipes.bagel.h100 import *
from megatron.bridge.recipes.deepseek import *
from megatron.bridge.recipes.deepseek.h100 import *
from megatron.bridge.recipes.exaone import *
from megatron.bridge.recipes.exaone.h100 import *
from megatron.bridge.recipes.gemma import *
from megatron.bridge.recipes.gemma.h100 import *
from megatron.bridge.recipes.gemma3_vl import *
from megatron.bridge.recipes.gemma3_vl.h100 import *
from megatron.bridge.recipes.gemma4_vl import *
from megatron.bridge.recipes.gemma4_vl.h100 import *
from megatron.bridge.recipes.glm import *
from megatron.bridge.recipes.glm.h100 import *
from megatron.bridge.recipes.glm_vl import *
from megatron.bridge.recipes.glm_vl.h100 import *
from megatron.bridge.recipes.gpt import *
from megatron.bridge.recipes.gpt.h100 import *
from megatron.bridge.recipes.gpt_oss import *
from megatron.bridge.recipes.gpt_oss.h100 import *
from megatron.bridge.recipes.kimi import *
from megatron.bridge.recipes.kimi.h100 import *
from megatron.bridge.recipes.kimi_vl import *
from megatron.bridge.recipes.kimi_vl.h100 import *
from megatron.bridge.recipes.llama import *
from megatron.bridge.recipes.llama.h100 import *
from megatron.bridge.recipes.minimax import *
from megatron.bridge.recipes.minimax.h100 import *
from megatron.bridge.recipes.ministral3 import *
from megatron.bridge.recipes.ministral3.h100 import *
from megatron.bridge.recipes.moonlight import *
from megatron.bridge.recipes.moonlight.h100 import *
from megatron.bridge.recipes.muse_glimmer import *
from megatron.bridge.recipes.muse_glimmer.h100 import *
from megatron.bridge.recipes.nemotron_vl import *
from megatron.bridge.recipes.nemotron_vl.h100 import *
from megatron.bridge.recipes.nemotronh import *
from megatron.bridge.recipes.nemotronh.h100 import *
from megatron.bridge.recipes.nemotronh_multimodal import *
from megatron.bridge.recipes.nemotronh_multimodal.h100 import *
from megatron.bridge.recipes.olmoe import *
from megatron.bridge.recipes.olmoe.h100 import *
from megatron.bridge.recipes.qwen import *
from megatron.bridge.recipes.qwen.h100 import *
from megatron.bridge.recipes.qwen2_audio import *
from megatron.bridge.recipes.qwen2_audio.h100 import *
from megatron.bridge.recipes.qwen_omni import *
from megatron.bridge.recipes.qwen_omni.h100 import *
from megatron.bridge.recipes.qwen_vl import *
from megatron.bridge.recipes.qwen_vl.h100 import *
from megatron.bridge.recipes.qwenair import *
from megatron.bridge.recipes.stepfun import *
from megatron.bridge.recipes.stepfun.h100 import *


_LAZY_RECIPE_MODULES = {
    "flux": "megatron.bridge.recipes.flux",
    "wan": "megatron.bridge.recipes.wan",
}

_LAZY_RECIPE_EXPORTS = {
    "flux_12b_pretrain_config": "megatron.bridge.recipes.flux",
    "flux_12b_sft_config": "megatron.bridge.recipes.flux",
    "flux_12b_pretrain_2gpu_h100_bf16_config": "megatron.bridge.recipes.flux.h100",
    "flux_12b_sft_2gpu_h100_bf16_config": "megatron.bridge.recipes.flux.h100",
    "wan_14b_pretrain_config": "megatron.bridge.recipes.wan",
    "wan_14b_sft_config": "megatron.bridge.recipes.wan",
    "wan_1_3b_pretrain_config": "megatron.bridge.recipes.wan",
    "wan_1_3b_sft_config": "megatron.bridge.recipes.wan",
    "wan_1_3b_text2image_pretrain_config": "megatron.bridge.recipes.wan",
    "wan_1_3b_text2video_pretrain_config": "megatron.bridge.recipes.wan",
    "wan_14b_pretrain_8gpu_h100_bf16_config": "megatron.bridge.recipes.wan.h100",
    "wan_14b_sft_8gpu_h100_bf16_config": "megatron.bridge.recipes.wan.h100",
    "wan_1_3b_pretrain_8gpu_h100_bf16_config": "megatron.bridge.recipes.wan.h100",
    "wan_1_3b_sft_8gpu_h100_bf16_config": "megatron.bridge.recipes.wan.h100",
    "wan_1_3b_text2image_pretrain_1gpu_h100_bf16_config": "megatron.bridge.recipes.wan.h100",
    "wan_1_3b_text2video_pretrain_4gpu_h100_bf16_config": "megatron.bridge.recipes.wan.h100",
}


def __getattr__(name: str) -> object:
    """Load diffusion recipe exports only when callers request them."""
    module_name = _LAZY_RECIPE_EXPORTS.get(name) or _LAZY_RECIPE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = _import_module(module_name)
    value = module if name in _LAZY_RECIPE_MODULES else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include deferred diffusion recipes in module introspection."""
    return sorted({*globals(), *_LAZY_RECIPE_MODULES, *_LAZY_RECIPE_EXPORTS})


__all__ = [
    *(name for name in globals() if not name.startswith("_")),
    *_LAZY_RECIPE_MODULES,
    *_LAZY_RECIPE_EXPORTS,
]
