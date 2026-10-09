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

"""Extract QwenAir training metrics and assess a bounded loss curve."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import statistics
from pathlib import Path


_LOGGER = logging.getLogger(__name__)
_NUMBER = r"[-+]?(?:nan|inf|(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
_ITERATION_PATTERN = re.compile(
    rf"iteration\s+(?P<step>\d+)/\s*(?P<total>\d+).*?"
    rf"consumed samples:\s*(?P<samples>\d+).*?"
    rf"learning rate:\s*(?P<learning_rate>{_NUMBER}).*?"
    rf"lm loss:\s*(?P<lm_loss>{_NUMBER}).*?"
    rf"router aux loss:\s*(?P<router_aux_loss>{_NUMBER}).*?"
    rf"grad norm:\s*(?P<grad_norm>{_NUMBER}).*?"
    rf"number of skipped iterations:\s*(?P<skipped>\d+).*?"
    rf"number of nan iterations:\s*(?P<nan>\d+)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    """Parse log-analysis options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=100)
    return parser.parse_args()


def _parse_metrics(log_path: Path) -> list[dict[str, int | float]]:
    by_step: dict[int, dict[str, int | float]] = {}
    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = _ITERATION_PATTERN.search(line)
            if match is None:
                continue
            step = int(match.group("step"))
            by_step[step] = {
                "step": step,
                "consumed_samples": int(match.group("samples")),
                "learning_rate": float(match.group("learning_rate")),
                "lm_loss": float(match.group("lm_loss")),
                "router_aux_loss": float(match.group("router_aux_loss")),
                "grad_norm": float(match.group("grad_norm")),
                "skipped_iterations": int(match.group("skipped")),
                "nan_iterations": int(match.group("nan")),
            }
    return [by_step[step] for step in sorted(by_step)]


def _linear_slope(points: list[dict[str, int | float]]) -> float:
    steps = [float(point["step"]) for point in points]
    losses = [float(point["lm_loss"]) for point in points]
    mean_step = statistics.fmean(steps)
    mean_loss = statistics.fmean(losses)
    denominator = sum((step - mean_step) ** 2 for step in steps)
    if denominator == 0:
        return 0.0
    return sum((step - mean_step) * (loss - mean_loss) for step, loss in zip(steps, losses)) / denominator


def _summarize(metrics: list[dict[str, int | float]], expected_steps: int) -> dict[str, object]:
    expected = set(range(1, expected_steps + 1))
    observed = {int(point["step"]) for point in metrics}
    missing_steps = sorted(expected - observed)
    nonfinite_steps = [
        int(point["step"])
        for point in metrics
        if not all(
            math.isfinite(float(point[field]))
            for field in ("learning_rate", "lm_loss", "router_aux_loss", "grad_norm")
        )
    ]
    skipped = max((int(point["skipped_iterations"]) for point in metrics), default=0)
    nan_iterations = max((int(point["nan_iterations"]) for point in metrics), default=0)

    early = [point for point in metrics if 11 <= int(point["step"]) <= 20]
    late_start = max(11, expected_steps - 9)
    late = [point for point in metrics if late_start <= int(point["step"]) <= expected_steps]
    regression = [point for point in metrics if 11 <= int(point["step"]) <= expected_steps]
    early_losses = [float(point["lm_loss"]) for point in early]
    late_losses = [float(point["lm_loss"]) for point in late]
    early_mean = statistics.fmean(early_losses) if early_losses else math.nan
    late_mean = statistics.fmean(late_losses) if late_losses else math.nan
    relative_drop = (early_mean - late_mean) / early_mean if early_mean else math.nan
    slope = _linear_slope(regression) if regression else math.nan
    early_median = statistics.median(early_losses) if early_losses else math.nan
    late_below_early_median = sum(loss < early_median for loss in late_losses)
    complete = not missing_steps and len(metrics) == expected_steps
    healthy = complete and not nonfinite_steps and skipped == 0 and nan_iterations == 0

    if not healthy or not math.isfinite(slope) or late_mean >= early_mean:
        verdict = "FAIL"
    elif relative_drop >= 0.02 and slope < 0 and late_below_early_median >= 7:
        verdict = "PASS"
    elif slope < 0:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    losses = [float(point["lm_loss"]) for point in metrics]
    return {
        "verdict": verdict,
        "expected_steps": expected_steps,
        "observed_steps": len(metrics),
        "missing_steps": missing_steps,
        "nonfinite_steps": nonfinite_steps,
        "skipped_iterations": skipped,
        "nan_iterations": nan_iterations,
        "final_consumed_samples": int(metrics[-1]["consumed_samples"]) if metrics else 0,
        "first_loss": losses[0] if losses else math.nan,
        "last_loss": losses[-1] if losses else math.nan,
        "minimum_loss": min(losses) if losses else math.nan,
        "early_window": [11, 20],
        "late_window": [late_start, expected_steps],
        "early_mean": early_mean,
        "early_median": early_median,
        "late_mean": late_mean,
        "relative_mean_drop": relative_drop,
        "post_warmup_ols_slope_per_step": slope,
        "late_points_below_early_median": late_below_early_median,
    }


def _write_plot(metrics: list[dict[str, int | float]], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _LOGGER.warning("matplotlib is unavailable; skipping %s", output_path)
        return False

    steps = [int(point["step"]) for point in metrics]
    losses = [float(point["lm_loss"]) for point in metrics]
    learning_rates = [float(point["learning_rate"]) for point in metrics]
    moving_average = [statistics.fmean(losses[max(0, index - 9) : index + 1]) for index in range(len(losses))]
    figure, loss_axis = plt.subplots(figsize=(10, 5.5))
    loss_axis.plot(steps, losses, color="#76B900", alpha=0.38, linewidth=1.0, label="LM loss")
    loss_axis.plot(steps, moving_average, color="#366A00", linewidth=2.2, label="10-step moving average")
    loss_axis.set_xlabel("Training step")
    loss_axis.set_ylabel("LM loss")
    loss_axis.grid(alpha=0.2)
    rate_axis = loss_axis.twinx()
    rate_axis.plot(steps, learning_rates, color="#1F77B4", linewidth=1.2, label="Learning rate")
    rate_axis.set_ylabel("Learning rate")
    lines = loss_axis.lines + rate_axis.lines
    loss_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    """Write CSV, JSON summary, and a loss plot when matplotlib is available."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.expected_steps < 1:
        raise ValueError("expected-steps must be positive")
    if not args.log.is_file():
        raise FileNotFoundError(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _parse_metrics(args.log)
    fields = [
        "step",
        "consumed_samples",
        "learning_rate",
        "lm_loss",
        "router_aux_loss",
        "grad_norm",
        "skipped_iterations",
        "nan_iterations",
    ]
    with (args.output_dir / "loss_curve.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)

    summary = _summarize(metrics, args.expected_steps)
    summary["plot_written"] = _write_plot(metrics, args.output_dir / "loss_curve.png") if metrics else False
    (args.output_dir / "loss_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _LOGGER.info("Convergence verdict: %s", summary["verdict"])


if __name__ == "__main__":
    main()
