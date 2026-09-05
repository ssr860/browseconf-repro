from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True, slots=True)
class ThresholdCandidate:
    threshold: int
    count: int
    coverage: float
    accuracy: float
    relative_improvement: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: int | None
    k: float
    overall_accuracy: float
    candidates: list[ThresholdCandidate]


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z * sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def select_threshold(
    observations: Iterable[tuple[int, bool]],
    *,
    k: float = 10,
    min_count: int = 1,
) -> ThresholdSelection:
    rows = list(observations)
    if not rows:
        raise ValueError("At least one calibration observation is required")
    if min_count < 1:
        raise ValueError("min_count must be positive")
    for confidence, _ in rows:
        if not 0 <= confidence <= 100:
            raise ValueError(f"Calibration confidence out of range: {confidence}")
    overall_accuracy = sum(correct for _, correct in rows) / len(rows)
    candidates: list[ThresholdCandidate] = []
    selected: int | None = None
    for threshold in range(101):
        subset = [correct for confidence, correct in rows if confidence >= threshold]
        if not subset:
            continue
        accuracy = sum(subset) / len(subset)
        if overall_accuracy == 0:
            relative = float("inf") if accuracy > 0 else 0.0
        else:
            relative = (accuracy - overall_accuracy) / overall_accuracy
        low, high = wilson_interval(sum(subset), len(subset))
        candidates.append(
            ThresholdCandidate(
                threshold=threshold,
                count=len(subset),
                coverage=len(subset) / len(rows),
                accuracy=accuracy,
                relative_improvement=relative,
                ci_low=low,
                ci_high=high,
            )
        )
        if selected is None and len(subset) >= min_count and relative >= k / 100:
            selected = threshold
    return ThresholdSelection(
        threshold=selected,
        k=k,
        overall_accuracy=overall_accuracy,
        candidates=candidates,
    )
