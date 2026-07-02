"""Offline end-to-end: ingest → dream → ask → trace; abstention; the trust
gate with verify-before-answer; judge_answer calibration; vitals."""

import pytest

from quen.engine import QuenEngine
from quen.llm import ScriptedLLM
from quen.verifiers import RepoGrepVerifier


@pytest.fixture
def engine(cfg, store, embedder, clock):
    return QuenEngine(
        cfg,
        store=store,
        llm=ScriptedLLM.with_offline_defaults(),
        embedder=embedder,
        clock=clock.now,
    )


def test_e2e_supersession_narrative(engine, store, clock):
    r1 = engine.ingest("The team fetches data via useApi.", source_kind="chat")
    assert len(r1.stored) == 1
    old_id = r1.stored[0].id
    assert r1.stored[0].triple is not None

    clock.advance(days=10)
    r2 = engine.ingest(
        "The team fetches data via useQuery.", source_ref="PR#42", source_kind="pr"
    )
    new_id = r2.stored[0].id
    assert r2.stored[0].confidence == pytest.approx(0.9)  # authority: pr

    clock.advance(days=1)
    report = engine.dream()
    assert any(
        ev.old_id == old_id and ev.new_id == new_id and ev.rule == "deterministic"
        for ev in report.supersessions
    )

    clock.advance(days=1)
    res = engine.ask("Which way does the team fetch data, useQuery or what?")
    used_ids = [sm.memory.id for sm in res.used]
    assert new_id in used_ids
    assert old_id not in used_ids
    assert not res.abstained
    assert 0.0 < res.answer_confidence <= 1.0

    trace = store.get_trace(res.trace_id)
    assert trace is not None
    assert any(e["memory_id"] == old_id for e in trace["excluded"])
    assert "superseded by" in trace["excluded"][0]["reason"]
    # counterfactual: what append-only RAG would have injected
    assert any(c["memory_id"] == old_id for c in trace["counterfactual"])
    assert trace["tokens_used"] <= trace["token_budget"]
    assert store.latest_trace()["trace_id"] == res.trace_id


def test_abstention_on_empty_memory(engine):
    res = engine.ask("what is the capital of the moon?")
    assert res.abstained
    assert res.used == []
    assert res.answer_confidence == pytest.approx(0.25)
    assert "don't" in res.answer.casefold()


def test_judge_answer_reviews_and_calibration(engine, store, clock):
    engine.ingest("The team deploy target is Alibaba Cloud ECS.", source_kind="doc")
    clock.advance(days=2)
    res = engine.ask("What is the team deploy target platform?")
    assert res.used
    used_id = res.used[0].memory.id

    clock.advance(days=1)
    engine.judge_answer(res.trace_id, correct=True)
    reviews = store.reviews_for(used_id)
    assert reviews and reviews[-1]["kind"] == "use_judged" and reviews[-1]["grade"] == 3
    events = store.calibration_events("confidence")
    assert len(events) == 1
    assert events[0]["predicted"] == pytest.approx(res.answer_confidence, abs=1e-3)
    assert events[0]["outcome"] == 1
    assert events[0]["freshness_days"] is not None


def test_verify_hint_closes_the_loop(engine, store, clock):
    r = engine.ingest("CI runner pool is ci-pool-7 for integration tests.")
    mem_id = r.stored[0].id
    conf_before = store.get(mem_id).confidence
    s_before = store.get(mem_id).stability

    clock.advance(days=3)
    event = engine.verify_hint(mem_id, "confirmed", evidence="harness grepped ci-pool-7")
    assert event.outcome == "confirmed"
    got = store.get(mem_id)
    assert got.confidence > conf_before
    assert got.last_verified_at == clock.now()
    assert got.stability > s_before
    reviews = store.reviews_for(mem_id)
    assert reviews[-1]["kind"] == "verification" and reviews[-1]["grade"] == 3


def test_verify_hint_refuted_tombstones(engine, store, clock):
    r = engine.ingest("Uploads use the legacy uploadV1 endpoint.")
    mem_id = r.stored[0].id
    clock.advance(days=3)
    event = engine.verify_hint(mem_id, "refuted", evidence="uploadV1 gone from repo")
    assert event.outcome == "refuted"
    got = store.get(mem_id)
    assert got.status in ("deprecated", "superseded")
    assert got.valid_to == clock.now()
    reviews = store.reviews_for(mem_id)
    assert reviews[-1]["grade"] == 1


def test_verify_hint_rejects_bad_outcome(engine, store):
    r = engine.ingest("Some fact to verify.")
    with pytest.raises(ValueError):
        engine.verify_hint(r.stored[0].id, "maybe")


def test_trust_gate_verify_before_answer_confirmed(
    cfg, store, embedder, clock, tmp_path
):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "config.ts").write_text(
        "export const MAX_UPLOAD_MB = 10;\nexport const REGION = 'ap-southeast-1';\n"
    )
    engine = QuenEngine(
        cfg,
        store=store,
        llm=ScriptedLLM.with_offline_defaults(),
        embedder=embedder,
        clock=clock.now,
        verifiers=[RepoGrepVerifier(str(repo))],
    )
    r = engine.ingest("Remember that MAX_UPLOAD_MB stays at 10 in src/config.ts.")
    mem_id = r.stored[0].id

    # age it past the trust threshold: 0.6 * 2^(-40/30) ≈ 0.24 < 0.55
    clock.advance(days=40)
    res = engine.ask("What is the MAX_UPLOAD_MB upload limit in the config?")

    assert res.verifications, "trust gate should have fired verify-before-answer"
    event = res.verifications[0]
    assert event.memory_id == mem_id
    assert event.outcome == "confirmed"
    assert event.evidence and "config.ts" in event.evidence

    got = store.get(mem_id)
    assert got.confidence > 0.6            # bumped
    assert got.last_verified_at == clock.now()
    assert store.reviews_for(mem_id)[-1]["kind"] == "verification"
    trace = store.get_trace(res.trace_id)
    assert trace["verifications"] and trace["verifications"][0]["outcome"] == "confirmed"


def test_pin_and_inspect(engine, store):
    r = engine.ingest("Never rotate the signing key without security review.")
    mem_id = r.stored[0].id
    engine.pin(mem_id)
    assert store.get(mem_id).pinned
    rows = engine.inspect(status="active")
    assert any(row["id"] == mem_id and row["pinned"] for row in rows)
    detail = engine.memory_detail(mem_id)
    assert detail["content_verbatim"]
    assert isinstance(detail["reviews"], list)


def test_vitals_shape(engine, store, clock):
    engine.ingest("The team fetches data via useApi.")
    clock.advance(days=2)
    engine.dream()
    v = engine.vitals()
    assert set(v["counts_by_status"]) == {"active", "deprecated", "superseded"}
    assert len(v["r_histogram"]) == 10
    assert sum(b["count"] for b in v["r_histogram"]) == v["counts_by_status"]["active"]
    assert v["kpis"]["fama"] is None  # no eval run — never fabricate numbers
    assert isinstance(v["retention_calibration"], list)
    assert [s["freshness_bucket"] for s in v["confidence_by_freshness_bucket"]] == [
        "<7d", "7-30d", ">30d",
    ]
