"""Eval harness: FAMA scorer semantics + one probe case end-to-end offline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

import fama  # noqa: E402
from run_probe import run_case  # noqa: E402
from run_longmemeval import load_instances, run_instance  # noqa: E402


def test_fama_presence_and_absence():
    s = fama.score_answer(
        "New code should use useQuery for data fetching.",
        valid_facts=["useQuery"],
        invalidated_facts=["useApi"],
    )
    assert s.presence and s.absence and s.fama


def test_fama_fails_on_invalidated_reliance():
    s = fama.score_answer(
        "Use the useApi hook to fetch data.",
        valid_facts=[],
        invalidated_facts=["useApi"],
    )
    assert not s.absence and not s.fama


def test_fama_negation_window_forgives_mentions():
    s = fama.score_answer(
        "We no longer use useApi; new code uses useQuery.",
        valid_facts=["useQuery"],
        invalidated_facts=["useApi"],
    )
    assert s.absence and s.fama


def test_fama_lookahead_passives():
    s = fama.score_answer(
        "The Redis session cluster is decommissioned; sessions live in DynamoDB.",
        valid_facts=["DynamoDB"],
        invalidated_facts=["Redis"],
    )
    assert s.absence and s.fama


def test_fama_word_boundaries():
    assert not fama.word_present("S3", "we use S3000 units")
    assert fama.word_present("S3", "avatars in S3.")


def test_fama_abstention():
    ok = fama.score_answer("I don't know.", valid_facts=[], invalidated_facts=[],
                           expects_abstain=True, abstained=False)
    assert ok.fama and ok.abstain_ok
    bad = fama.score_answer("It is hosted in us-east-1.", valid_facts=[],
                            invalidated_facts=[], expects_abstain=True,
                            abstained=False)
    assert not bad.fama


SUPERSESSION_CASE = {
    "id": "t-superseded",
    "history": [
        {"day": 0, "kind": "chat", "text": "The team fetches data via useApi."},
        {"day": 10, "kind": "pr", "source_ref": "PR#1",
         "text": "The team fetches data via useQuery."},
    ],
    "question": "The team fetches data via which hook?",
    "question_day": 12,
    "valid_facts": ["useQuery"],
    "invalidated_facts": ["useApi"],
}


def test_probe_case_ours_beats_append_only():
    ours = run_case(SUPERSESSION_CASE, "quen", budget=300, live=False)
    baseline = run_case(SUPERSESSION_CASE, "append_only", budget=300, live=False)
    assert ours["fama"] and not baseline["fama"]
    assert not baseline["absence"]  # append-only echoes the stale fact
    assert ours["forgetting"]["invalidated_forgotten"] == 1
    assert ours["forgetting"]["correct_tombstones"] >= 1


def test_longmemeval_dry_run_sample():
    instances = load_instances(live=False, hf_file="", limit=None)
    assert len(instances) == 3
    ku = next(i for i in instances if i["question_type"] == "knowledge-update"
              and not str(i["question_id"]).endswith("_abs"))
    row = run_instance(ku, "quen", budget=300, live=False)
    assert row["correct"], row["answer"]
    abst = next(i for i in instances if str(i["question_id"]).endswith("_abs"))
    row = run_instance(abst, "quen", budget=300, live=False)
    assert row["correct"] and row["abstained"]
