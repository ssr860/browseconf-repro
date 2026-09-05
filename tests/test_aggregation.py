import math

import pytest

from browseconf.aggregation import CandidateAnswer, cisc, normalize_answer, self_consistency


def candidates():
    return [
        CandidateAnswer("The Eiffel Tower", 50, 1),
        CandidateAnswer("Eiffel tower.", 40, 2),
        CandidateAnswer("Louvre", 99, 3),
    ]


def test_normalization_and_self_consistency():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    result = self_consistency(candidates())
    assert result.answer == "The Eiffel Tower"
    assert result.member_indices == [1, 2]


def test_cisc_can_prefer_high_confidence_minority():
    result = cisc(candidates(), temperature=5)
    assert result.answer == "Louvre"


def test_cisc_infinite_temperature_is_sc():
    result = cisc(candidates(), temperature=math.inf)
    assert result.answer == "The Eiffel Tower"


def test_invalid_cisc_temperature():
    with pytest.raises(ValueError):
        cisc(candidates(), temperature=0)
