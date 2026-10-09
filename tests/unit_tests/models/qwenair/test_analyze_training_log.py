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

"""QwenAir bounded-training log analysis contracts."""

from examples.models.qwenair.analyze_training_log import _parse_metrics, _summarize


def _metrics(steps: int, global_batch_size: int) -> list[dict[str, int | float]]:
    return [
        {
            "step": step,
            "consumed_samples": step * global_batch_size,
            "learning_rate": 1.0e-3,
            "lm_loss": 10.0 - 0.05 * step,
            "router_aux_loss": 0.1,
            "grad_norm": 1.0,
            "skipped_iterations": 0,
            "nan_iterations": 0,
        }
        for step in range(1, steps + 1)
    ]


def test_summary_uses_explicit_warmup_and_global_batch_size() -> None:
    metrics = _metrics(20, 128)

    summary = _summarize(metrics, 20, warmup_steps=2, global_batch_size=128)

    assert summary["early_window"] == [3, 12]
    assert summary["late_window"] == [11, 20]
    assert summary["consumed_sample_mismatch_steps"] == []
    assert summary["final_consumed_samples"] == 20 * 128
    assert summary["post_warmup_ols_slope_per_step"] < 0


def test_summary_fails_a_per_step_sample_count_mismatch() -> None:
    metrics = _metrics(20, 128)
    metrics[6]["consumed_samples"] = 6 * 128

    summary = _summarize(metrics, 20, warmup_steps=2, global_batch_size=128)

    assert summary["consumed_sample_mismatch_steps"] == [7]
    assert summary["verdict"] == "FAIL"


def test_log_parser_discards_metrics_before_the_latest_step_reset(tmp_path) -> None:
    def _line(step: int, loss: float) -> str:
        return (
            f"iteration {step}/ 3 | consumed samples: {step * 128} | learning rate: 1.0E-3 | "
            f"lm loss: {loss:.4f} | router aux loss: 2.0 | grad norm: 1.0 | "
            "number of skipped iterations: 0 | number of nan iterations: 0 |\n"
        )

    log_path = tmp_path / "reused.log"
    log_path.write_text(
        _line(1, 9.0) + _line(2, 8.0) + _line(3, 7.0) + _line(1, 6.0) + _line(2, 5.0),
        encoding="utf-8",
    )

    metrics = _parse_metrics(log_path)

    assert [point["step"] for point in metrics] == [1, 2]
    assert [point["lm_loss"] for point in metrics] == [6.0, 5.0]
