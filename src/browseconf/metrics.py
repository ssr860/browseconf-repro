from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from .calibration import wilson_interval


def confidence_bin(confidence: int) -> str:
    if confidence == 100:
        return "100"
    lower = max(0, confidence // 5 * 5)
    return f"{lower}-{lower + 4}"


def binary_auroc(scores: list[float], labels: list[bool]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def brier_score(scores: list[float], labels: list[bool]) -> float | None:
    if not scores:
        return None
    return sum((score / 100 - int(label)) ** 2 for score, label in zip(scores, labels)) / len(
        scores
    )


def expected_calibration_error(
    scores: list[int], labels: list[bool], bin_width: int = 5
) -> float | None:
    if not scores:
        return None
    total = len(scores)
    error = 0.0
    for lower in range(0, 100, bin_width):
        upper = lower + bin_width
        indices = [
            index
            for index, score in enumerate(scores)
            if lower <= score < upper or (upper == 100 and score == 100)
        ]
        if not indices:
            continue
        confidence = sum(scores[index] / 100 for index in indices) / len(indices)
        accuracy = sum(labels[index] for index in indices) / len(indices)
        error += len(indices) / total * abs(confidence - accuracy)
    return error


def result_report(
    predictions: Iterable[dict[str, Any]],
    judgements: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    prediction_rows = list(predictions)
    judgement_map = {
        str(row["question_id"]): row.get("correct")
        for row in judgements
        if row.get("correct") is not None
    }
    paired = [row for row in prediction_rows if str(row.get("question_id")) in judgement_map]
    labels = [bool(judgement_map[str(row["question_id"])]) for row in paired]
    successes = sum(labels)
    low, high = wilson_interval(successes, len(labels))
    attempts = [int(row.get("attempt_count", len(row.get("attempts", [])) or 1)) for row in paired]
    confidences = [int(row.get("final_confidence", row.get("confidence", -1))) for row in paired]
    valid_confidences = [
        (confidence, label)
        for confidence, label in zip(confidences, labels)
        if 0 <= confidence <= 100
    ]
    bins: dict[str, dict[str, Any]] = {}
    for label in sorted({confidence_bin(score) for score, _ in valid_confidences}):
        bin_rows = [pair for pair in valid_confidences if confidence_bin(pair[0]) == label]
        bins[label] = {
            "count": len(bin_rows),
            "fraction": len(bin_rows) / len(valid_confidences),
            "mean_confidence": sum(score for score, _ in bin_rows) / len(bin_rows),
            "accuracy": sum(correct for _, correct in bin_rows) / len(bin_rows),
        }
    stop_reasons = Counter(row.get("stop_reason", "unknown") for row in paired)
    scores = [score for score, _ in valid_confidences]
    confidence_labels = [label for _, label in valid_confidences]
    return {
        "n_predictions": len(prediction_rows),
        "n_judged": len(labels),
        "accuracy": successes / len(labels) if labels else None,
        "accuracy_ci95": [low, high] if labels else None,
        "avg_attempts": sum(attempts) / len(attempts) if attempts else None,
        "stop_reasons": dict(stop_reasons),
        "confidence": {
            "coverage": len(valid_confidences) / len(paired) if paired else None,
            "auroc": binary_auroc([float(score) for score in scores], confidence_labels),
            "brier": brier_score([float(score) for score in scores], confidence_labels),
            "ece_5point": expected_calibration_error(scores, confidence_labels),
            "bins": bins,
        },
    }
