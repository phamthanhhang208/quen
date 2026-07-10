"""alibaba_client usage counters — incl. DashScope context-cache telemetry."""

from types import SimpleNamespace

import pytest

from quen import alibaba_client as ac


@pytest.fixture(autouse=True)
def _clean_counters():
    ac.reset_usage()
    yield
    ac.reset_usage()


def _resp(prompt=100, completion=10, cached=None):
    details = None if cached is None else SimpleNamespace(cached_tokens=cached)
    return SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        total_tokens=prompt + completion, prompt_tokens_details=details,
    ))


def test_tracks_cached_tokens_when_present():
    ac._track("qwen-flash", _resp(prompt=2000, cached=1500))
    got = ac.usage_summary()["qwen-flash"]
    assert got["prompt_tokens"] == 2000
    assert got["cached_tokens"] == 1500


def test_defaults_to_zero_without_details():
    ac._track("qwen-flash", _resp(cached=None))
    resp_no_attr = SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=5, completion_tokens=1, total_tokens=6))
    ac._track("qwen-flash", resp_no_attr)
    got = ac.usage_summary()["qwen-flash"]
    assert got["calls"] == 2
    assert got["cached_tokens"] == 0


def test_accumulates_per_model():
    ac._track("qwen-flash", _resp(cached=100))
    ac._track("qwen-flash", _resp(cached=250))
    ac._track("qwen3.5-plus", _resp(cached=None))
    s = ac.usage_summary()
    assert s["qwen-flash"]["cached_tokens"] == 350
    assert s["qwen3.5-plus"]["cached_tokens"] == 0


def test_transient_server_faults_in_400_clothing_are_retried():
    """DashScope wraps InternalError.Algo in HTTP 400 — must retry; a
    genuine 400 must re-raise immediately."""
    import httpx
    from openai import BadRequestError

    def bad_request(msg):
        resp = httpx.Response(400, request=httpx.Request("POST", "http://x"),
                              json={"error": {"message": msg}})
        return BadRequestError(msg, response=resp, body=None)

    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise bad_request("<5000203> InternalError.Algo.Embedding_pipeline_Error: x")
        return "ok"
    assert ac._with_transient_retry(flaky) == "ok"
    assert calls["n"] == 3

    def genuine():
        raise bad_request("invalid parameter: dimensions")
    import pytest as _pytest
    with _pytest.raises(BadRequestError):
        ac._with_transient_retry(genuine)
