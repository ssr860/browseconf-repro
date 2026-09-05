import pytest

from browseconf.parsing import (
    parse_answer_and_confidence,
    parse_judge_correct,
    parse_tool_call,
)


def test_parses_paper_answer_format():
    parsed = parse_answer_and_confidence(
        "<answer>\n**Answer**: Raffaele Contigiani\n**Confidence**: 97\n</answer>"
    )
    assert parsed.answer == "Raffaele Contigiani"
    assert parsed.confidence == 97
    assert parsed.warnings == []


def test_missing_and_invalid_confidence_are_negative_one():
    assert parse_answer_and_confidence("Answer: x").confidence == -1
    parsed = parse_answer_and_confidence("Answer: x\nConfidence: 101")
    assert parsed.confidence == -1
    assert "confidence_out_of_range" in parsed.warnings


def test_parses_last_tool_call():
    parsed = parse_tool_call(
        '<think>x</think><tool_call>{"name":"search","arguments":{"query":["a"]}}</tool_call>'
    )
    assert parsed == ("search", {"query": ["a"]})


def test_invalid_tool_json_raises():
    with pytest.raises(ValueError):
        parse_tool_call("<tool_call>{bad}</tool_call>")


def test_judge_parser_returns_capture_group_not_full_match():
    assert parse_judge_correct("reasoning: ok\ncorrect: yes\nconfidence: 99") is True
    assert parse_judge_correct("correct: no") is False
    assert parse_judge_correct("unparseable") is None
