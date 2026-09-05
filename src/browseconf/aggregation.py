from __future__ import annotations

import math
import re
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass


def normalize_answer(answer: str) -> str:
    value = unicodedata.normalize("NFKC", answer).casefold().strip()
    value = re.sub(r"^[\s\-*#]*(?:answer|exact answer)\s*:\s*", "", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\b(?:a|an|the)\b", " ", value)
    return " ".join(value.split())


@dataclass(frozen=True, slots=True)
class CandidateAnswer:
    answer: str
    confidence: int
    index: int


@dataclass(frozen=True, slots=True)
class AggregatedAnswer:
    answer: str
    member_indices: list[int]
    score: float
    cluster_key: str


def _clusters(
    candidates: list[CandidateAnswer],
    equivalence: Callable[[str, str], bool] | None = None,
) -> list[list[CandidateAnswer]]:
    if equivalence is None:
        grouped: OrderedDict[str, list[CandidateAnswer]] = OrderedDict()
        for candidate in candidates:
            grouped.setdefault(normalize_answer(candidate.answer), []).append(candidate)
        return list(grouped.values())
    clusters: list[list[CandidateAnswer]] = []
    for candidate in candidates:
        for cluster in clusters:
            if equivalence(cluster[0].answer, candidate.answer):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    return clusters


def self_consistency(
    candidates: Iterable[CandidateAnswer],
    *,
    equivalence: Callable[[str, str], bool] | None = None,
) -> AggregatedAnswer:
    rows = list(candidates)
    if not rows:
        raise ValueError("Cannot aggregate an empty candidate set")
    clusters = _clusters(rows, equivalence)
    selected = max(clusters, key=lambda cluster: (len(cluster), -min(row.index for row in cluster)))
    representative = min(selected, key=lambda row: row.index)
    return AggregatedAnswer(
        answer=representative.answer,
        member_indices=[row.index for row in selected],
        score=float(len(selected)),
        cluster_key=normalize_answer(representative.answer),
    )


def cisc(
    candidates: Iterable[CandidateAnswer],
    *,
    temperature: float = 10.0,
    equivalence: Callable[[str, str], bool] | None = None,
) -> AggregatedAnswer:
    rows = list(candidates)
    if not rows:
        raise ValueError("Cannot aggregate an empty candidate set")
    if temperature <= 0 and not math.isinf(temperature):
        raise ValueError("CISC temperature must be positive")
    if math.isinf(temperature):
        weights = [1.0 / len(rows)] * len(rows)
    else:
        logits = [row.confidence / temperature for row in rows]
        offset = max(logits)
        exponentials = [math.exp(logit - offset) for logit in logits]
        denominator = sum(exponentials)
        weights = [value / denominator for value in exponentials]
    weight_by_index = {row.index: weight for row, weight in zip(rows, weights)}
    clusters = _clusters(rows, equivalence)
    selected = max(
        clusters,
        key=lambda cluster: (
            sum(weight_by_index[row.index] for row in cluster),
            -min(row.index for row in cluster),
        ),
    )
    representative = min(selected, key=lambda row: row.index)
    score = sum(weight_by_index[row.index] for row in selected)
    return AggregatedAnswer(
        answer=representative.answer,
        member_indices=[row.index for row in selected],
        score=score,
        cluster_key=normalize_answer(representative.answer),
    )


def pass_at_k(correctness: Iterable[bool], k: int) -> bool:
    if k < 1:
        raise ValueError("k must be positive")
    return any(list(correctness)[:k])
