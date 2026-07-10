#!/usr/bin/env python3
"""LongMemEval (arXiv 2410.10813) — Knowledge-Update, Temporal-Reasoning and
Abstention subsets, across the same configs as the probe.

Dry-run (default): a 3-instance committed sample proves the pipeline offline.
Live: downloads `xiaowu0162/longmemeval-cleaned` from HuggingFace
(`--hf-file longmemeval_oracle.json` by default — evidence-only sessions;
use longmemeval_s_cleaned.json for the full haystack) and needs
DASHSCOPE_API_KEY for the Qwen reader.

Scoring, live runs: an LLM judge (the LongMemEval paper's own protocol) is
the primary metric — it accepts paraphrases, digit/word number forms and
hedged-but-correct answers, and requires a clean decline on abstention
items; the judge sees every config's answers symmetrically. The
conservative exact word-boundary match is still computed and reported
alongside as `correct_exact`. Dry-run (offline) scores with the exact
matcher only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common import build_stack, make_parser, write_json  # noqa: E402
from configs import AppendOnlyRAG, FullContext, Quen  # noqa: E402
from fama import looks_like_abstention, word_present  # noqa: E402

SUBSETS = ("knowledge-update", "temporal-reasoning")
DATA_DIR = Path(__file__).parent / "data"
SAMPLE = DATA_DIR / "sample_longmemeval.json"
CONFIGS = ["append_only", "full_context", "quen"]


def load_instances(live: bool, hf_file: str, limit: int | None,
                   offset: int = 0) -> list[dict]:
    if not live:
        instances = json.loads(SAMPLE.read_text())
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise SystemExit(
                "pip install huggingface_hub to download LongMemEval "
                "(or pre-place the file under eval/data/)."
            )
        path = hf_hub_download(
            repo_id="xiaowu0162/longmemeval-cleaned",
            filename=hf_file,
            repo_type="dataset",
            local_dir=DATA_DIR / "longmemeval",
        )
        instances = json.loads(Path(path).read_text())
    wanted = [
        i for i in instances
        if i["question_type"] in SUBSETS or str(i["question_id"]).endswith("_abs")
    ]
    wanted = wanted[offset:]
    return wanted[:limit] if limit else wanted


def make_system(config: str, live: bool):
    llm, embedder = build_stack(live)
    if config == "append_only":
        return AppendOnlyRAG(llm, embedder)
    if config == "full_context":
        return FullContext(llm, embedder)
    return Quen(llm, embedder, verify=False)  # no live source in this bench


def _judge_correct(inst: dict, answer_text: str, *, live: bool,
                   exact: bool) -> bool:
    """LLM-judge verdict (live) with the exact matcher as the offline
    fallback. Applied to every config identically."""
    if not live:
        return exact
    from quen import alibaba_client as ac
    from quen import llm as llm_mod

    msgs = llm_mod.render_lme_judge(
        inst["question"],
        str(inst.get("answer", "")).strip(),
        answer_text,
        expects_abstain=str(inst["question_id"]).endswith("_abs"),
    )
    resp = ac.chat(
        msgs,
        model=os.environ.get("QUEN_LME_JUDGE_MODEL", "qwen3.7-plus"),
        temperature=0.0,
        max_tokens=8,
        extra_body={"enable_thinking": False},
    )
    return (resp.choices[0].message.content or "").strip().upper().startswith("YES")


def run_instance(inst: dict, config: str, *, budget: int, live: bool) -> dict:
    from quen.alibaba_client import is_content_filter

    system = make_system(config, live)
    sessions = inst.get("haystack_sessions", [])
    sessions_skipped = 0
    for day, session in enumerate(sessions):
        if day > 0:
            system.day_boundary(day)
        text = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            for turn in session
        )
        try:
            system.ingest(text, day=day, kind="chat",
                          source_ref=f"session-{day}")
        except Exception as exc:
            # DashScope's input inspection permanently rejects some benchmark
            # session texts — skip that session (recorded on the row, same
            # rule for every config) instead of killing the shard
            if not is_content_filter(exc):
                raise
            sessions_skipped += 1
    # the pre-answer consolidation always runs in full, even under a
    # write-count dream cadence (haystack-scale histories)
    final = getattr(system, "final_boundary", system.day_boundary)
    final(len(sessions))
    try:
        ans = system.answer(inst["question"], budget)
    except Exception as exc:
        if not is_content_filter(exc):
            raise
        # retrieved units tripped the filter at answer time: an empty answer,
        # scored as wrong — degraded, symmetric, and recorded
        from configs import AnswerOut
        ans = AnswerOut("", None, 0, abstained=False)

    is_abstention = str(inst["question_id"]).endswith("_abs")
    gold = str(inst.get("answer", "")).strip()
    if is_abstention:
        # judged from the delivered TEXT for every config — never from a
        # config's self-reported flag (baselines cannot emit one)
        exact = looks_like_abstention(ans.text)
    else:
        # word-boundary match: bare containment lets short golds like
        # "before" match almost any verbose answer
        exact = bool(gold) and word_present(gold, ans.text)
    try:
        correct = _judge_correct(inst, ans.text, live=live, exact=exact)
    except Exception as exc:
        if not is_content_filter(exc):
            raise
        correct = exact  # judge input tripped the filter: exact-match stands
    return {
        "question_id": inst["question_id"],
        "question_type": inst["question_type"],
        "abstention": is_abstention,
        "config": config,
        "correct": correct,
        "correct_exact": exact,
        "abstained": ans.abstained,
        "tokens_used": ans.tokens_used,
        "sessions_skipped": sessions_skipped,
        # stated confidence (quen only; None for baselines) — the raw
        # material for calibration refits at benchmark scale
        "answer_confidence": ans.confidence,
        "answer": ans.text[:300],
    }


def aggregate(rows: list[dict]) -> dict:
    out: dict = {}
    for config in CONFIGS:
        sub = [r for r in rows if r["config"] == config]
        agg = {}
        for subset in SUBSETS:
            chunk = [r for r in sub if r["question_type"] == subset
                     and not r["abstention"]]
            if chunk:
                agg[subset] = round(
                    sum(r["correct"] for r in chunk) / len(chunk), 4
                )
                agg[f"{subset}_exact"] = round(
                    sum(r.get("correct_exact", r["correct"]) for r in chunk)
                    / len(chunk), 4
                )
        abst = [r for r in sub if r["abstention"]]
        if abst:
            agg["abstention"] = round(
                sum(r["correct"] for r in abst) / len(abst), 4
            )
            agg["abstention_exact"] = round(
                sum(r.get("correct_exact", r["correct"]) for r in abst)
                / len(abst), 4
            )
        agg["avg_tokens"] = (
            round(sum(r["tokens_used"] for r in sub) / len(sub), 1) if sub else None
        )
        out[config] = agg
    return out


def main(argv: list[str] | None = None) -> dict:
    parser = make_parser(__doc__)
    parser.add_argument("--hf-file", default="longmemeval_oracle.json")
    args = parser.parse_args(argv)
    instances = load_instances(args.live, args.hf_file, args.limit, args.offset)
    print(f"{len(instances)} instances "
          f"({'live' if args.live else 'dry-run sample'})")

    # crash-safe resume: one JSONL row per (question, config), appended as
    # completed; a re-run skips what's already done
    resume_path = args.out / "longmemeval_rows.jsonl"
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if resume_path.exists():
        for line in resume_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            done.add((str(row["question_id"]), row["config"]))
        if done:
            print(f"resuming: {len(done)} rows already done")

    total = len(instances) * len(CONFIGS)
    with resume_path.open("a") as fh:
        for i, inst in enumerate(instances):
            for config in CONFIGS:
                key = (str(inst["question_id"]), config)
                if key in done:
                    continue
                row = run_instance(inst, config, budget=args.budget,
                                   live=args.live)
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            if (i + 1) % 10 == 0:
                print(f"progress: {(i + 1) * len(CONFIGS)}/{total}")

    summary = aggregate(rows)
    write_json(args.out / "longmemeval_results.json", rows)
    write_json(args.out / "longmemeval_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
