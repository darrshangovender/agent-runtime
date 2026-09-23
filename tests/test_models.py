import pytest

from agent_runtime import (
    AllTiersFailedError,
    DeterministicFallback,
    FallbackChain,
    MockModel,
    ModelClient,
    ModelError,
    ModelTimeoutError,
)


async def test_mock_model_script_and_calls():
    m = MockModel(["a", "b"])
    assert (await m.complete("p1")).text == "a"
    assert (await m.complete("p2")).text == "b"
    assert (await m.complete("p3")).text == "b"  # repeats last
    assert m.call_count == 3 and m.calls[0]["prompt"] == "p1"


async def test_mock_model_exhaustion_and_exceptions():
    m = MockModel(["a", ValueError("bad")], repeat_last=False)
    await m.complete("x")
    with pytest.raises(ValueError):
        await m.complete("x")
    with pytest.raises(ModelError, match="exhausted"):
        await m.complete("x")


async def test_mock_model_callable_script():
    m = MockModel([lambda p: p.upper()])
    assert (await m.complete("hi")).text == "HI"


def test_clients_satisfy_protocol():
    assert isinstance(MockModel(["x"]), ModelClient)
    assert isinstance(DeterministicFallback(), ModelClient)
    assert isinstance(FallbackChain([DeterministicFallback()]), ModelClient)


async def test_fallback_records_tier_zero_when_primary_answers():
    chain = FallbackChain([MockModel(["primary"]), DeterministicFallback()])
    resp = await chain.complete("q")
    assert resp.text == "primary" and resp.tier == 0
    assert [a.ok for a in resp.attempts] == [True]


async def test_fallback_on_exception_moves_to_next_tier():
    chain = FallbackChain(
        [MockModel([ModelError("503")], name="p"), MockModel(["second"], name="s"),
         DeterministicFallback()]
    )
    resp = await chain.complete("q")
    assert resp.text == "second" and resp.tier == 1 and resp.model == "s"
    assert [a.ok for a in resp.attempts] == [False, True]
    assert "ModelError" in resp.attempts[0].error


async def test_fallback_on_timeout_moves_to_next_tier():
    chain = FallbackChain(
        [MockModel(["slow"], latency_s=1.0), MockModel(["fast"])], timeout_s=0.05
    )
    resp = await chain.complete("q")
    assert resp.text == "fast" and resp.tier == 1
    assert "ModelTimeoutError" in resp.attempts[0].error


async def test_fallback_reaches_deterministic_tier():
    chain = FallbackChain(
        [MockModel([ModelError("a")]), MockModel([ModelError("b")]),
         DeterministicFallback([(r"weather", "It is sunny.")])]
    )
    resp = await chain.complete("what is the weather?")
    assert resp.tier == 2 and resp.text == "It is sunny."
    assert chain.has_deterministic_tail


async def test_all_tiers_fail_without_deterministic_tail():
    chain = FallbackChain([MockModel([ModelError("a")]), MockModel([ModelError("b")])])
    with pytest.raises(AllTiersFailedError) as ei:
        await chain.complete("q")
    assert len(ei.value.attempts) == 2
    # the last tier's exception is chained, not swallowed
    assert isinstance(ei.value.__cause__, ModelError) and str(ei.value.__cause__) == "b"


async def test_mock_model_accepts_exception_classes_in_script():
    m = MockModel([ModelError, "ok"], name="m")
    with pytest.raises(ModelError, match="scripted failure"):
        await m.complete("q")
    assert (await m.complete("q")).text == "ok"


async def test_per_tier_timeouts_list():
    chain = FallbackChain(
        [MockModel(["slow"], latency_s=0.2), MockModel(["ok"])], timeout_s=[0.02, None]
    )
    assert (await chain.complete("q")).tier == 1
    with pytest.raises(ValueError):
        FallbackChain([MockModel(["x"])], timeout_s=[1.0, 2.0])
    with pytest.raises(ValueError):
        FallbackChain([])


async def test_deterministic_never_raises():
    d = DeterministicFallback([("(unclosed", "x"), (r"hello", "hi")], default="dflt")
    assert (await d.complete("hello there")).text == "hi"
    assert (await d.complete("nothing matches")).text == "dflt"
    assert (await d.complete(None)).text == "dflt"  # type: ignore[arg-type]
    assert (await d.complete(12345)).text == "dflt"  # type: ignore[arg-type]
    resp = await d.complete("")
    assert resp.model == "deterministic" and resp.cost_usd == 0.0


async def test_timeout_error_is_both_model_and_timeout_error():
    with pytest.raises(AllTiersFailedError) as ei:
        await FallbackChain([MockModel(["x"], latency_s=1)], timeout_s=0.01).complete("q")
    assert "ModelTimeoutError" in ei.value.attempts[0].error
    assert issubclass(ModelTimeoutError, TimeoutError) and issubclass(ModelTimeoutError, ModelError)
