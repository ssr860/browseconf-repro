import pytest

from browseconf.calibration import select_threshold, wilson_interval


def test_selects_minimum_paper_threshold():
    observations = [(60, False), (70, False), (80, True), (90, True)]
    selection = select_threshold(observations, k=10)
    # overall=.5; at tau=61, 2/3=.667 is the first >=10% relative improvement.
    assert selection.threshold == 61
    assert selection.overall_accuracy == 0.5


def test_min_count_is_optional_audit_constraint():
    observations = [(10, True), (20, False), (100, True)]
    literal = select_threshold(observations, k=10, min_count=1)
    constrained = select_threshold(observations, k=10, min_count=2)
    assert literal.threshold == 21
    assert constrained.threshold is None


def test_rejects_invalid_input():
    with pytest.raises(ValueError):
        select_threshold([])
    with pytest.raises(ValueError):
        select_threshold([(101, True)])


def test_wilson_interval_bounds():
    low, high = wilson_interval(5, 10)
    assert 0 < low < 0.5 < high < 1
