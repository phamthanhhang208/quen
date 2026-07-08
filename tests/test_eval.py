"""Eval harness: FAMA scorer semantics + one probe case end-to-end offline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

import fama  # noqa: E402
from configs import AppendOnlyRAG, split_units  # noqa: E402
from run_probe import run_case  # noqa: E402
from run_longmemeval import load_instances, run_instance  # noqa: E402

from quen.embeddings import HashingEmbedder  # noqa: E402
from quen.llm import ScriptedLLM  # noqa: E402
from quen.retrieval import estimate_tokens  # noqa: E402


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


def test_split_units_leaves_single_facts_alone():
    """The probe shape must pass through untouched — canonical probe
    numbers depend on it."""
    text = "The team fetches data via useApi."
    assert split_units(text) == [text]


def test_split_units_turn_granularity():
    session = "\n".join(
        [
            "user: We migrated the avatar store to S3 yesterday.",
            "assistant: Noted. " + "Filler sentence about nothing. " * 40,
            "user: Also MAX_UPLOAD_MB is now 25.",
        ]
    )
    units = split_units(session)
    assert len(units) > 3  # per turn, long turn further windowed
    assert any("S3" in u for u in units)
    assert any("MAX_UPLOAD_MB" in u for u in units)
    # every window individually fits a small reader budget
    assert all(estimate_tokens(u) <= 180 for u in units)
    # the windowed assistant turn keeps its speaker prefix
    assert sum(u.startswith("assistant:") for u in units) >= 2


def test_baseline_sees_context_despite_huge_sessions():
    """Regression: whole-session units made every LongMemEval session
    bigger than the answer budget, so the baseline answered from an EMPTY
    context (tokens_used=0 on all 458 rows) — a strawman, not a baseline."""
    rag = AppendOnlyRAG(ScriptedLLM.with_offline_defaults(), HashingEmbedder())
    turns = [f"user: Fact number {i} about topic-{i}. " + "pad " * 100
             for i in range(6)]
    rag.ingest("\n".join(turns), day=0, kind="chat", source_ref="s1")
    out = rag.answer("What is fact number 3?", budget=300)
    assert out.tokens_used > 0
    assert out.tokens_used <= 300


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


def test_word_present_separators_colon_and_endash():
    """Scorer bugs found in the KU failure audit: ':' and unicode dashes
    were not join separators, failing literally-correct answers."""
    assert fama.word_present("6:00 pm", "dinner is at 6:00 pm sharp")
    assert fama.word_present("10-12 hours", "it took 10–12 hours")  # en-dash
    assert not fama.word_present("S3", "we use S3000 units")  # boundary intact


def test_abstention_markers_do_not_credit_hedge_then_answer():
    assert fama.looks_like_abstention(
        "There are no relevant memories recorded about that.")
    # a hedge followed by a substantive answer is NOT an abstention
    assert not fama.looks_like_abstention(
        "I cannot state with full assurance, but the records indicate 12.")


def test_lme_judge_template_shapes():
    from quen import llm as llm_mod

    msgs = llm_mod.render_lme_judge("How many?", "15", "Fifteen, I believe.")
    assert msgs[0]["content"].startswith(f"### TASK: {llm_mod.LME_JUDGE}")
    assert "paraphrases" in msgs[0]["content"]
    assert "Gold answer: 15" in msgs[1]["content"]

    abst = llm_mod.render_lme_judge("When did X?", "abstain", "No idea.",
                                    expects_abstain=True)
    assert "DECLINES" in abst[0]["content"]
    assert "hedge followed by a guess" in abst[0]["content"]


def test_dry_run_rows_carry_both_metrics():
    instances = load_instances(live=False, hf_file="", limit=None)
    ku = next(i for i in instances if i["question_type"] == "knowledge-update"
              and not str(i["question_id"]).endswith("_abs"))
    row = run_instance(ku, "quen", budget=300, live=False)
    assert row["correct"] == row["correct_exact"]  # offline: judge == exact


def test_extract_prompt_keeps_passing_personal_facts():
    from quen import llm as llm_mod

    body = llm_mod.render_extract("x", "2026-07-05")[0]["content"]
    assert "personal facts" in body and "VERBATIM" in body
    assert "suggestion" in body  # assistant suggestions are not user facts
    sal = llm_mod.render_salience(["f"])[0]["content"]
    assert "salient by construction" in sal


def test_nli_prompt_teaches_attribute_domains():
    from quen import llm as llm_mod

    body = llm_mod.render_nli("a", "b")[0]["content"]
    assert "different attributes" in body.lower() or "DIFFERENT attributes" in body
    assert "pnpm" in body and "tabs" in body  # the augment few-shot
    assert "updated" in body or "update" in body


def test_answer_prompt_recency_and_commit_rules():
    from quen import llm as llm_mod

    body = llm_mod.render_answer("q", ["- [t=0.9 1d] x"], "h")[0]["content"]
    assert "prefer the NEWER" in body
    assert "do not refuse to answer" in body
