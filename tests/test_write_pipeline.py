"""Write pipeline tests (spec §4.3 — store the delta).

Everything is offline and deterministic: HashingEmbedder + ScriptedLLM +
the injectable clock. FSRS reviews at elapsed_days=0 do not grow S, so
every reinforcement test advances the clock between ingests.
"""

from __future__ import annotations

import json

import pytest

from quen import fsrs, llm
from quen.write_pipeline import SkippedFact, WriteResult, ingest_observation


def _fact(content: str, triple: list[str] | None = None, mtype: str = "episodic") -> dict:
    return {"content": content, "triple": triple, "mtype": mtype}


def _extract_json(*facts: dict) -> str:
    return json.dumps(list(facts))


def _salience_json(*pairs: tuple[float, float]) -> str:
    return json.dumps([{"salience": s, "importance": i} for s, i in pairs])


@pytest.fixture
def ingest(store, scripted, embedder, cfg, clock):
    """Bind the shared fixtures; `now` always reads the mutable clock."""

    def _run(text: str, **kw) -> WriteResult:
        return ingest_observation(
            text,
            store=store,
            llm=scripted,
            embedder=embedder,
            cfg=cfg,
            now=clock.now(),
            **kw,
        )

    return _run


# ------------------------------------------------------------- empty input

def test_empty_text_stores_nothing(ingest, store, scripted):
    result = ingest("   \n\t  ")
    assert result.stored == []
    assert result.skipped == []
    assert result.reinforced == []
    assert store.list() == []
    assert scripted.calls == []  # no LLM call was made


# ----------------------------------------------------------- salience gate

def test_salience_gate_skips_low_salience_facts(ingest, store, scripted, cfg):
    scripted.script(llm.EXTRACT, _extract_json(
        _fact("quen persists memories in sqlite wal mode"),
        _fact("python lists are mutable"),
    ))
    scripted.script(llm.SALIENCE, _salience_json((0.9, 7.0), (0.05, 2.0)))

    result = ingest("some observation")

    assert len(result.stored) == 1
    assert result.stored[0].content == "quen persists memories in sqlite wal mode"
    assert result.stored[0].salience == 0.9
    assert result.stored[0].importance == 7.0
    assert result.reinforced == []
    assert result.skipped == [
        SkippedFact(content="python lists are mutable", reason="salience")
    ]
    assert len(store.list()) == 1  # the skipped fact created no row


def test_salience_parse_failure_falls_back_to_defaults(ingest, scripted):
    scripted.script(llm.EXTRACT, _extract_json(
        _fact("the api gateway lives in the infra repo"),
    ))
    scripted.script(llm.SALIENCE, "not parseable %%%")

    result = ingest("obs")

    # defaults (0.5, 5.0) pass the 0.3 gate — rating failure never drops facts
    assert len(result.stored) == 1
    assert result.stored[0].salience == 0.5
    assert result.stored[0].importance == 5.0


def test_salience_length_mismatch_defaults_missing_and_clamps(ingest, scripted):
    scripted.script(llm.EXTRACT, _extract_json(
        _fact("deploy target is region ap southeast one"),
        _fact("release cadence is every second friday"),
    ))
    # only one entry, and out of range: clamp salience to 1.0, importance to 10
    scripted.script(llm.SALIENCE, _salience_json((1.7, 42.0)))

    result = ingest("obs")

    assert len(result.stored) == 2
    assert result.stored[0].salience == 1.0
    assert result.stored[0].importance == 10.0
    assert result.stored[1].salience == 0.5   # missing entry -> defaults
    assert result.stored[1].importance == 5.0


# ------------------------------------------------------------ triple dedup

def test_triple_dedup_reinforces_existing_memory(ingest, store, scripted, clock):
    scripted.script(llm.EXTRACT, [
        _extract_json(_fact(
            "team fetches data via useApi",
            ["team", "fetches data via", "useApi"],
        )),
        # same slot triple, different casing/phrasing: must dedup, not re-store
        _extract_json(_fact(
            "the team still fetches data via useApi",
            ["Team", "Fetches Data Via", "USEAPI"],
        )),
    ])
    scripted.script(llm.SALIENCE, _salience_json((0.9, 7.0)))

    first = ingest("session one")
    assert len(first.stored) == 1
    original = first.stored[0]
    s0 = original.stability
    assert original.review_count == 0

    clock.advance(days=3)
    second = ingest("session two")

    assert second.stored == []
    assert second.reinforced == [original.id]
    assert second.skipped == [
        SkippedFact(
            content="the team still fetches data via useApi",
            reason="dup_triple",
            existing_id=original.id,
        )
    ]
    assert len(store.list()) == 1  # no second row

    mem = store.get(original.id)
    assert mem.review_count == 1
    assert mem.stability > s0  # Good review after 3 elapsed days grows S
    assert mem.last_review_at == clock.now()
    assert mem.last_accessed_at == clock.now()

    reviews = store.reviews_for(original.id)
    assert len(reviews) == 1
    assert reviews[0]["kind"] == "manual"
    assert reviews[0]["grade"] == int(fsrs.Grade.GOOD)
    assert reviews[0]["elapsed_days"] == pytest.approx(3.0)
    assert 0.0 < reviews[0]["r_before"] < 1.0
    assert reviews[0]["s_after"] > reviews[0]["s_before"]

    tail = store.audit_tail(action="dedup_reinforce")
    assert len(tail) == 1
    assert tail[0]["memory_id"] == original.id


def test_in_batch_triple_dedup_keeps_first(ingest, store, scripted):
    scripted.script(llm.EXTRACT, _extract_json(
        _fact("team uses pytest for tests", ["team", "uses", "pytest"]),
        _fact("tests are written with pytest", ["Team ", "USES", " Pytest"]),
    ))
    scripted.script(llm.SALIENCE, _salience_json((0.9, 6.0), (0.9, 6.0)))

    result = ingest("obs")

    assert len(result.stored) == 1
    assert result.stored[0].content == "team uses pytest for tests"
    assert result.reinforced == []  # nothing pre-existing was touched
    assert result.skipped == [
        SkippedFact(
            content="tests are written with pytest",
            reason="dup_triple",
            existing_id=None,
        )
    ]
    assert len(store.list()) == 1


# ------------------------------------------------------------ cosine dedup

def test_cosine_dedup_reinforces_on_rephrasing(ingest, store, scripted, clock):
    scripted.script(llm.EXTRACT, [
        _extract_json(_fact(
            "the deploy pipeline uses docker buildx for arm builds"
        )),
        # same token bag, reordered: cosine 1.0 on the hashing embedder
        _extract_json(_fact(
            "for arm builds the deploy pipeline uses docker buildx"
        )),
    ])
    scripted.script(llm.SALIENCE, _salience_json((0.8, 6.0)))

    first = ingest("obs one")
    assert len(first.stored) == 1
    original = first.stored[0]
    s0 = original.stability

    clock.advance(days=2)
    second = ingest("obs two")

    assert second.stored == []
    assert second.reinforced == [original.id]
    assert second.skipped == [
        SkippedFact(
            content="for arm builds the deploy pipeline uses docker buildx",
            reason="dup_cosine",
            existing_id=original.id,
        )
    ]
    assert len(store.list()) == 1

    mem = store.get(original.id)
    assert mem.review_count == 1
    assert mem.stability > s0

    reviews = store.reviews_for(original.id)
    assert len(reviews) == 1
    assert reviews[0]["kind"] == "manual"
    assert reviews[0]["grade"] == int(fsrs.Grade.GOOD)


def test_dissimilar_facts_are_not_cosine_deduped(ingest, store, scripted):
    scripted.script(llm.EXTRACT, [
        _extract_json(_fact("ci runs on github actions")),
        _extract_json(_fact("release cadence is monthly")),
    ])
    scripted.script(llm.SALIENCE, _salience_json((0.8, 6.0)))

    ingest("obs one")
    second = ingest("obs two")

    assert len(second.stored) == 1
    assert second.reinforced == []
    assert len(store.list()) == 2


# -------------------------------------------------------- source authority

def test_authority_confidence_mapping(ingest, store, scripted, cfg):
    scripted.script(llm.EXTRACT, [
        _extract_json(_fact("ci runs on github actions")),
        _extract_json(_fact("release cadence is monthly")),
    ])
    scripted.script(llm.SALIENCE, _salience_json((0.8, 6.0)))

    from_pr = ingest("obs one", source_kind="pr", source_ref="PR#42")
    assert from_pr.stored[0].confidence == cfg.authority_confidence["pr"] == 0.9
    assert from_pr.stored[0].source_ref == "PR#42"

    from_unknown = ingest("obs two", source_kind="carrier_pigeon")
    assert (
        from_unknown.stored[0].confidence
        == cfg.authority_confidence["default"]
        == 0.7
    )

    creates = store.audit_tail(action="create")
    assert len(creates) == 2
    assert creates[1]["detail"]["source_kind"] == "pr"       # newest first
    assert creates[1]["detail"]["source_ref"] == "PR#42"
    assert creates[0]["detail"]["source_kind"] == "carrier_pigeon"


def test_stored_memory_gets_fsrs_initial_state(ingest, scripted, clock):
    scripted.script(llm.EXTRACT, _extract_json(
        _fact("staging db name is quen_staging", ["staging db", "is", "quen_staging"]),
    ))
    scripted.script(llm.SALIENCE, _salience_json((0.9, 8.0)))

    result = ingest("obs")

    d0, s0 = fsrs.initial_state(fsrs.Grade.GOOD)
    mem = result.stored[0]
    assert mem.difficulty == d0
    assert mem.stability == s0
    assert mem.review_count == 0
    assert mem.status == "active"
    assert mem.triple == ("staging db", "is", "quen_staging")
    assert mem.valid_from == clock.now()


# --------------------------------------------------------- extract failure

def test_malformed_extract_twice_stores_raw_episodic(ingest, store, scripted, cfg):
    scripted.script(llm.EXTRACT, ["garbage %%% not json", "@@@ still broken"])
    text = "raw observation with the config value FOO=bar"

    result = ingest(text)

    assert len(result.stored) == 1
    assert result.skipped == []
    assert result.reinforced == []
    mem = result.stored[0]
    assert mem.content == text          # the ENTIRE raw text, never dropped
    assert mem.triple is None
    assert mem.mtype == "episodic"
    assert mem.importance == 5.0
    assert mem.salience == 0.5
    assert mem.confidence == cfg.authority_confidence["chat"]  # default kind

    extract_calls = [c for c in scripted.calls if c["marker"] == llm.EXTRACT]
    assert len(extract_calls) == 2      # one retry, then fallback

    errors = store.audit_tail(action="error")
    assert len(errors) == 1
    assert errors[0]["detail"]["stage"] == "extract"
    assert errors[0]["memory_id"] == mem.id
    assert len(store.list()) == 1
    # SALIENCE was never scripted: reaching here proves it was not called


def test_extract_retry_recovers_on_second_attempt(ingest, store, scripted):
    scripted.script(llm.EXTRACT, [
        "garbage %%% not json",
        _extract_json(_fact("the feature flag service is unleash")),
    ])
    scripted.script(llm.SALIENCE, _salience_json((0.8, 6.0)))

    result = ingest("obs")

    assert len(result.stored) == 1
    assert result.stored[0].content == "the feature flag service is unleash"
    assert store.audit_tail(action="error") == []


def test_invalid_extract_entries_are_dropped_and_triples_validated(ingest, scripted):
    raw = json.dumps([
        {"content": "   "},                                    # empty -> dropped
        "not even a dict",                                     # dropped
        {"content": "on call rotation is weekly",
         "triple": ["on call rotation", "is weekly"],          # 2-list -> None
         "mtype": "bogus"},                                    # -> episodic
        {"content": "primary region is eu west",
         "triple": ["primary region", "is", "eu west"],
         "mtype": "semantic"},
    ])
    scripted.script(llm.EXTRACT, raw)
    scripted.script(llm.SALIENCE, _salience_json((0.8, 6.0), (0.8, 6.0)))

    result = ingest("obs")

    assert len(result.stored) == 2
    first, second = result.stored
    assert first.content == "on call rotation is weekly"
    assert first.triple is None
    assert first.mtype == "episodic"
    assert second.triple == ("primary region", "is", "eu west")
    assert second.mtype == "semantic"
