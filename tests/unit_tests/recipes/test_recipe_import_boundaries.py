# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Import-boundary tests for public recipe exports."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

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


def test_qwenair_recipe_import_does_not_load_diffusion_recipes(tmp_path: Path) -> None:
    """A QwenAir deep import must not require Diffusers or TorchAO."""
    blocker_dir = tmp_path / "without_diffusion_dependencies"
    blocker_dir.mkdir()
    (blocker_dir / "sitecustomize.py").write_text(
        """
import builtins
import importlib.util

_original_find_spec = importlib.util.find_spec
_original_import = builtins.__import__

def _is_blocked(name):
    return name == "diffusers" or name.startswith("diffusers.") or name == "torchao" or name.startswith("torchao.")

def _find_spec(fullname, package=None):
    if _is_blocked(fullname):
        return None
    return _original_find_spec(fullname, package)

def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if _is_blocked(name):
        root_name = name.partition(".")[0]
        raise ModuleNotFoundError(f"No module named {root_name!r}", name=root_name)
    return _original_import(name, globals, locals, fromlist, level)

importlib.util.find_spec = _find_spec
builtins.__import__ = _import
""",
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(blocker_dir),
            str(repo_root / "src"),
            str(repo_root / "3rdparty" / "Megatron-LM"),
            environment.get("PYTHONPATH"),
        )
        if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from megatron.bridge.recipes.qwenair.b300.qwenair import _tiny_text_config; "
                "assert _tiny_text_config()['model_type'] == 'qwen4_exp_text'; "
                "assert not any(name == 'megatron.bridge.recipes.flux' or "
                "name.startswith('megatron.bridge.recipes.flux.') for name in sys.modules); "
                "assert not any(name == 'megatron.bridge.recipes.wan' or "
                "name.startswith('megatron.bridge.recipes.wan.') for name in sys.modules)"
            ),
        ],
        capture_output=True,
        env=environment,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_recipe_package_tolerates_mcore_without_qwenair(tmp_path: Path) -> None:
    """External MCore checkouts without QwenAir keep the generic recipes importable."""
    blocker_dir = tmp_path / "without_qwenair_mcore"
    blocker_dir.mkdir()
    (blocker_dir / "sitecustomize.py").write_text(
        """
import importlib.util

_original_find_spec = importlib.util.find_spec

def _find_spec(fullname, package=None):
    if fullname == "megatron.core.models.qwenair":
        return None
    return _original_find_spec(fullname, package)

importlib.util.find_spec = _find_spec
""",
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(blocker_dir),
            str(repo_root / "src"),
            str(repo_root / "3rdparty" / "Megatron-LM"),
            environment.get("PYTHONPATH"),
        )
        if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import megatron.bridge.recipes as recipes; "
                "import megatron.bridge.recipes.qwenair as qwenair; "
                "assert qwenair.__all__ == []; "
                "assert not hasattr(recipes, 'qwenair_text_pretrain_32gpu_b300_bf16_config')"
            ),
        ],
        capture_output=True,
        env=environment,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_lazy_diffusion_exports_preserve_wildcard_and_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """All legacy diffusion names resolve to the original recipe callables."""
    recipes = importlib.import_module("megatron.bridge.recipes")
    expected = {
        name: getattr(importlib.import_module(module_name), name) for name, module_name in _LAZY_RECIPE_EXPORTS.items()
    }
    for name in _LAZY_RECIPE_EXPORTS:
        monkeypatch.delattr(recipes, name, raising=False)

    namespace: dict[str, object] = {}
    exec("from megatron.bridge.recipes import *", namespace)

    assert _LAZY_RECIPE_EXPORTS.keys() <= set(recipes.__all__)
    assert _LAZY_RECIPE_EXPORTS.keys() <= set(dir(recipes))
    for name in ("flux", "wan"):
        assert namespace[name] is importlib.import_module(f"megatron.bridge.recipes.{name}")
        assert getattr(recipes, name) is namespace[name]
    for name, expected_value in expected.items():
        assert namespace[name] is expected_value
        assert getattr(recipes, name) is expected_value
