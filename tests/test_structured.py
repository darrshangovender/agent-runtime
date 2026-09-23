import pytest
from pydantic import BaseModel

from agent_runtime import DegradedResult, MockModel, enforce_schema
from agent_runtime.structured import JSONExtractionError, extract_json


class Answer(BaseModel):
    title: str
    score: int


GOOD = '{"title": "ok", "score": 7}'


async def test_valid_first_try():
    m = MockModel([GOOD])
    out = await enforce_schema(m, "q", Answer)
    assert isinstance(out, Answer) and out.score == 7
    assert m.call_count == 1
    assert "JSON schema" in m.calls[0]["prompt"]


async def test_self_correction_succeeds_on_retry_two():
    m = MockModel(["not json at all", '{"title": "ok", "score": "seven"}', GOOD])
    out = await enforce_schema(m, "q", Answer, max_retries=2)
    assert isinstance(out, Answer)
    assert m.call_count == 3
    # each retry carries the previous raw output and the validation error
    assert "not json at all" in m.calls[1]["prompt"]
    assert "Validation error" in m.calls[1]["prompt"]
    assert "score" in m.calls[2]["prompt"] and "seven" in m.calls[2]["prompt"]


async def test_degraded_result_after_retries_exhausted():
    m = MockModel(['{"title": "x"}', '{"title": "y"}', '{"title": "z", "score": null}'])
    out = await enforce_schema(m, "q", Answer, max_retries=2)
    assert isinstance(out, DegradedResult)
    assert out.schema_name == "Answer"
    assert out.attempt_count == 3 and len(out.errors) == 3
    assert out.partial == {"title": "z", "score": None}
    assert out.degraded is True
    assert m.call_count == 3


async def test_zero_retries_means_single_call():
    m = MockModel(["garbage"])
    out = await enforce_schema(m, "q", Answer, max_retries=0)
    assert isinstance(out, DegradedResult) and m.call_count == 1
    with pytest.raises(ValueError):
        await enforce_schema(m, "q", Answer, max_retries=-1)


async def test_json_in_code_fence_and_prose_is_accepted():
    m = MockModel(["Sure! Here you go:\n```json\n" + GOOD + "\n```\nHope that helps."])
    out = await enforce_schema(m, "q", Answer)
    assert isinstance(out, Answer)


async def test_model_exceptions_propagate():
    m = MockModel([RuntimeError("provider down")])
    with pytest.raises(RuntimeError, match="provider down"):
        await enforce_schema(m, "q", Answer)


async def test_tier_and_cost_recorded_in_degraded_attempts():
    m = MockModel(["bad"], cost_per_call_usd=0.01)
    out = await enforce_schema(m, "q", Answer, max_retries=1)
    assert isinstance(out, DegradedResult)
    assert out.total_cost_usd == pytest.approx(0.02)
    assert out.attempts[0].model == "mock"


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json("prefix {\"a\": [1, 2]} suffix") == {"a": [1, 2]}
    assert extract_json("[1, 2]") == [1, 2]
    with pytest.raises(JSONExtractionError):
        extract_json("")
    with pytest.raises(JSONExtractionError):
        extract_json("no braces here")
