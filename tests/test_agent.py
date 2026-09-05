from browseconf.agent import TextReactAgent
from browseconf.models import ScriptedModel
from browseconf.tools import StaticTool, ToolRouter


def test_agent_executes_tool_then_returns_answer():
    model = ScriptedModel(
        [
            '<think>need search</think><tool_call>{"name":"search","arguments":{"query":["q"]}}</tool_call>',
            "<answer>Answer: result\nConfidence: 88</answer>",
        ]
    )
    search = StaticTool("search", "search evidence")
    agent = TextReactAgent(
        model=model,
        tools=ToolRouter.from_tools(search, StaticTool("visit", "page")),
    )
    attempt = agent.run(question_id="q1", question="question", attempt_index=1)
    assert attempt.answer == "result"
    assert attempt.confidence == 88
    assert attempt.interaction_count == 1
    assert search.calls == [{"query": ["q"]}]
    assert "search evidence" in model.calls[1][-1]["content"]


def test_context_overflow_is_failure():
    model = ScriptedModel(["unused"])
    agent = TextReactAgent(
        model=model,
        tools=ToolRouter.from_tools(),
        max_context_tokens=1,
    )
    attempt = agent.run(question_id="q", question="too long", attempt_index=1)
    assert attempt.stop_reason == "context_overflow"
    assert attempt.confidence == -1
    assert attempt.answer is None


def test_official_protocol_uses_direct_question_and_sampling_parameters():
    model = ScriptedModel(["<answer>Answer: result\nConfidence: 88</answer>"])
    agent = TextReactAgent(
        model=model,
        tools=ToolRouter.from_tools(),
        protocol="official-deepresearch",
        max_output_tokens=10_000,
        presence_penalty=1.1,
        logprobs=True,
    )
    agent.run(question_id="q", question="direct question", attempt_index=1)
    assert model.calls[0][1]["content"] == "direct question"
    assert "Current date:" in model.calls[0][0]["content"]
    assert model.call_options[0]["stop"] == ["\n<tool_response>", "<tool_response>"]
    assert model.call_options[0]["presence_penalty"] == 1.1
    assert model.call_options[0]["logprobs"] is True


def test_malformed_tool_call_returns_error_observation_then_recovers():
    model = ScriptedModel(
        [
            '<tool_call>{"name": "search", invalid}</tool_call>',
            "<answer>Answer: recovered\nConfidence: 80</answer>",
        ]
    )
    agent = TextReactAgent(model=model, tools=ToolRouter.from_tools())
    attempt = agent.run(question_id="q", question="Q", attempt_index=1)
    assert attempt.answer == "recovered"
    assert "Invalid tool-call JSON" in model.calls[1][-1]["content"]


def test_official_context_limit_forces_a_final_answer():
    model = ScriptedModel(["<answer>Answer: forced\nConfidence: 55</answer>"])
    agent = TextReactAgent(
        model=model,
        tools=ToolRouter.from_tools(),
        protocol="official-deepresearch",
        max_context_tokens=1,
    )
    attempt = agent.run(question_id="q", question="Q", attempt_index=1)
    assert attempt.answer == "forced"
    assert attempt.stop_reason == "answer_context_limit"
    assert "maximum context length" in model.calls[0][-1]["content"]


def test_paper_protocol_uses_published_prompt_and_overflow_is_failure():
    model = ScriptedModel(["unused"])
    agent = TextReactAgent(
        model=model,
        tools=ToolRouter.from_tools(),
        protocol="paper-browseconf",
        max_context_tokens=1,
    )
    attempt = agent.run(question_id="q", question="Q", attempt_index=1)
    assert attempt.stop_reason == "context_overflow"
    assert attempt.answer is None
    assert attempt.confidence == -1
    assert model.calls == []


def test_paper_protocol_accepts_published_markdown_answer_format():
    model = ScriptedModel(["**Answer**: result\n**Confidence**: 88"])
    agent = TextReactAgent(
        model=model,
        tools=ToolRouter.from_tools(),
        protocol="paper-browseconf",
    )
    attempt = agent.run(question_id="q", question="direct question", attempt_index=1)
    assert attempt.answer == "result"
    assert attempt.confidence == 88
    assert attempt.stop_reason == "answer_without_xml_tag"
    assert model.calls[0][1]["content"] == "direct question"
    assert "Web Information Seeking Master" in model.calls[0][0]["content"]
    assert '"name": "search"' in model.calls[0][0]["content"]


def test_attempt_time_limit_stops_before_model_call():
    model = ScriptedModel(["unused"])
    agent = TextReactAgent(
        model=model,
        tools=ToolRouter.from_tools(),
        max_duration_seconds=0,
    )
    attempt = agent.run(question_id="q", question="Q", attempt_index=1)
    assert attempt.stop_reason == "time_limit"
    assert attempt.model_call_count == 0
