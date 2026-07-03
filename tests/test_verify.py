"""Verification tests (spec §4.7): verification IS an FSRS review
(verification-updates-retention), refutation tombstones on the spot, and
RepoGrepVerifier greps a live mini-repo deterministically."""

from __future__ import annotations

import dataclasses

import pytest

from quen.retrieval import recall
from quen.trust import apply_verification
from quen.verifiers import RepoGrepVerifier, VerificationEvent


# ------------------------------------------- verification-updates-retention


def test_confirmed_is_a_good_review_and_bumps_confidence(
    store, cfg, clock, mem_factory
) -> None:
    mem = mem_factory("the deploy pipeline uses blue_green rollout")
    store.add(mem, actor="test")
    s_before = mem.stability
    clock.advance(days=3)

    event = apply_verification(
        mem,
        "confirmed",
        store=store,
        cfg=cfg,
        now=clock.now(),
        evidence="ci.yaml:12: strategy: blue_green",
        verifier="repo_grep",
    )

    stored = store.get(mem.id)
    assert stored.stability > s_before                      # retention reinforced
    # repo_grep is weak evidence (existence, not truth): bump scaled by 0.7
    assert stored.confidence == pytest.approx(min(1.0, 0.7 + cfg.confidence_bump_on_confirm * 0.7))
    assert stored.last_verified_at == clock.now()
    assert stored.last_review_at == clock.now()
    assert stored.review_count == 1
    assert stored.status == "active"

    rows = store.reviews_for(mem.id)
    assert len(rows) == 1
    assert rows[0]["kind"] == "verification"
    assert rows[0]["grade"] == 3
    assert rows[0]["elapsed_days"] == pytest.approx(3.0)
    assert rows[0]["s_after"] > rows[0]["s_before"]

    tail = store.audit_tail(memory_id=mem.id, action="verify")
    assert len(tail) == 1
    assert tail[0]["detail"]["outcome"] == "confirmed"
    assert tail[0]["detail"]["verifier"] == "repo_grep"

    assert isinstance(event, VerificationEvent)
    assert event.memory_id == mem.id
    assert event.outcome == "confirmed"
    assert event.at == clock.now()


def test_refuted_with_newer_slot_mate_supersedes_and_recall_excludes(
    store, cfg, clock, embedder, mem_factory
) -> None:
    old = mem_factory(
        "the team fetches data with the useApi hook",
        triple=("team", "fetches data via", "useApi"),
    )
    store.add(old, actor="test")
    clock.advance(days=1)
    new = mem_factory(
        "the team fetches data with the useQuery hook",
        triple=("team", "fetches data via", "useQuery"),
    )
    store.add(new, actor="test")

    clock.advance(days=3)
    s_before = old.stability
    apply_verification(old, "refuted", store=store, cfg=cfg, now=clock.now())

    stored = store.get(old.id)
    assert stored.status == "superseded"
    assert stored.superseded_by == new.id                   # points at the slot-mate
    assert stored.valid_to == clock.now()
    assert stored.stability < s_before                      # Again review shrank S
    assert stored.confidence == pytest.approx(0.7 * cfg.confidence_cut_on_refute)

    rows = store.reviews_for(old.id)
    assert len(rows) == 1
    assert rows[0]["kind"] == "verification"
    assert rows[0]["grade"] == 1
    assert rows[0]["s_after"] < rows[0]["s_before"]

    # the refuted memory is out of the answer path, but shown as excluded
    result = recall(
        "how does the team fetch data useApi hook",
        token_budget=500,
        store=store,
        embedder=embedder,
        cfg=cfg,
        now=clock.now(),
    )
    used_ids = [sm.memory.id for sm in result.used]
    assert old.id not in used_ids
    assert new.id in used_ids
    excluded = {e["memory_id"]: e for e in result.excluded_relevant}
    assert old.id in excluded
    assert "superseded by" in excluded[old.id]["reason"]
    assert new.id in excluded[old.id]["reason"]


def test_refuted_without_slot_mate_deprecates(store, cfg, clock, mem_factory) -> None:
    mem = mem_factory(
        "the app caches sessions in redis",
        triple=("app", "caches sessions in", "redis"),
    )
    store.add(mem, actor="test")
    clock.advance(days=2)

    apply_verification(mem, "refuted", store=store, cfg=cfg, now=clock.now())

    stored = store.get(mem.id)
    assert stored.status == "deprecated"
    assert stored.superseded_by is None
    assert stored.valid_to == clock.now()


def test_refuted_tripleless_memory_deprecates(store, cfg, clock, mem_factory) -> None:
    mem = mem_factory("standup happens at ten every morning")
    store.add(mem, actor="test")
    clock.advance(days=2)

    apply_verification(mem, "refuted", store=store, cfg=cfg, now=clock.now())
    assert store.get(mem.id).status == "deprecated"


def test_unverifiable_changes_nothing_but_is_audited(
    store, cfg, clock, mem_factory
) -> None:
    mem = mem_factory("service timeout is thirty seconds")
    store.add(mem, actor="test")
    before = dataclasses.asdict(store.get(mem.id))
    clock.advance(days=2)

    event = apply_verification(
        mem, "unverifiable", store=store, cfg=cfg, now=clock.now(), verifier="repo_grep"
    )

    after = dataclasses.asdict(store.get(mem.id))
    assert after == before                                  # zero field changes
    assert store.reviews_for(mem.id) == []                  # not a review

    tail = store.audit_tail(memory_id=mem.id, action="verify")
    assert len(tail) == 1
    assert tail[0]["detail"]["outcome"] == "unverifiable"
    assert tail[0]["detail"]["verifier"] == "repo_grep"

    assert event.outcome == "unverifiable"
    assert event.at == clock.now()


def test_unknown_outcome_raises(store, cfg, clock, mem_factory) -> None:
    mem = mem_factory("fact")
    store.add(mem, actor="test")
    with pytest.raises(ValueError):
        apply_verification(mem, "maybe", store=store, cfg=cfg, now=clock.now())


# ----------------------------------------------------------- RepoGrepVerifier


@pytest.fixture
def mini_repo(tmp_path):
    """A tiny repo checkout: real source, plus .git and binary noise that
    must be skipped."""
    repo = tmp_path / "demo-repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text(
        "import lib\n\ndata = useQuery(fetch_notifications)\n"
    )
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text("useApi = true\n")  # must NOT be found
    (repo / "blob.bin").write_bytes(b"\x00\x01useApi\x00")    # binary, skipped
    return repo


def test_repo_grep_confirmed_with_file_line_evidence(mini_repo, mem_factory) -> None:
    verifier = RepoGrepVerifier(str(mini_repo))
    mem = mem_factory(
        "the team fetches data via useQuery",
        triple=("team", "fetches data via", "useQuery"),
    )
    assert verifier.can_verify(mem)
    outcome, evidence = verifier.verify(mem)
    assert outcome == "confirmed"
    assert evidence is not None
    assert evidence.startswith("src/app.py:3:")
    assert "useQuery" in evidence


def test_repo_grep_refuted_skips_git_and_binary(mini_repo, mem_factory) -> None:
    verifier = RepoGrepVerifier(str(mini_repo))
    mem = mem_factory(
        "the team fetches data via useApi",
        triple=("team", "fetches data via", "useApi"),
    )
    outcome, evidence = verifier.verify(mem)
    assert outcome == "refuted"                 # only in .git/config and blob.bin
    assert "'useApi'" in evidence
    assert "not found" in evidence
    assert "searched 1 files" in evidence       # app.py only


def test_repo_grep_prefers_code_shaped_identifiers(tmp_path, mem_factory) -> None:
    """A plain English word present in the repo must not fake a confirmation
    when a code-shaped identifier exists and is absent."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "notes.txt").write_text("this project is called quen\n")
    verifier = RepoGrepVerifier(str(repo))
    mem = mem_factory("the project uses useApi everywhere")
    outcome, _ = verifier.verify(mem)
    assert outcome == "refuted"                 # 'project' ignored, 'useApi' absent


def test_repo_grep_unverifiable_without_identifier(mini_repo, mem_factory) -> None:
    verifier = RepoGrepVerifier(str(mini_repo))
    mem = mem_factory("and the with from that it is so")
    assert not verifier.can_verify(mem)
    outcome, evidence = verifier.verify(mem)
    assert outcome == "unverifiable"
    assert evidence is not None


def test_repo_grep_unverifiable_on_empty_or_missing_repo(tmp_path, mem_factory) -> None:
    mem = mem_factory("config lives in useQuery")

    empty = tmp_path / "empty"
    empty.mkdir()
    outcome, _ = RepoGrepVerifier(str(empty)).verify(mem)
    assert outcome == "unverifiable"

    missing = RepoGrepVerifier(str(tmp_path / "nope"))
    assert not missing.can_verify(mem)
    assert missing.verify(mem)[0] == "unverifiable"


def test_repo_grep_skips_files_over_size_limit(tmp_path, mem_factory) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "big.txt").write_text("pad\n" * 400 + "needleIdent\n")  # > 1 KiB
    (repo / "small.txt").write_text("hello world notes\n")
    verifier = RepoGrepVerifier(str(repo), max_file_kb=1)
    mem = mem_factory("config lives in needleIdent")
    outcome, evidence = verifier.verify(mem)
    assert outcome == "refuted"                 # big file skipped, small scanned
    assert "searched 1 files" in evidence


def test_repo_grep_closes_the_loop_with_apply_verification(
    mini_repo, store, cfg, clock, mem_factory
) -> None:
    """End to end: grep the live repo, feed the outcome back as a review."""
    verifier = RepoGrepVerifier(str(mini_repo))
    mem = mem_factory(
        "the team fetches data via useQuery",
        triple=("team", "fetches data via", "useQuery"),
    )
    store.add(mem, actor="test")
    clock.advance(days=3)

    outcome, evidence = verifier.verify(mem)
    event = apply_verification(
        mem,
        outcome,
        store=store,
        cfg=cfg,
        now=clock.now(),
        evidence=evidence,
        verifier=verifier.name,
    )

    assert event.outcome == "confirmed"
    stored = store.get(mem.id)
    assert stored.confidence > 0.7
    assert stored.last_verified_at == clock.now()
    assert store.reviews_for(mem.id)[0]["kind"] == "verification"
    assert store.audit_tail(memory_id=mem.id, action="verify")[0]["detail"][
        "evidence"
    ].startswith("src/app.py:3:")
