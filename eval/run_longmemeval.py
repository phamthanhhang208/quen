#!/usr/bin/env python3
"""LongMemEval (arXiv 2410.10813) — Knowledge-Update, Temporal-Reasoning and
Abstention subsets, across the same configs as the probe.

Dry-run (default): a 3-instance committed sample proves the pipeline offline.
Live: downloads `xiaowu0162/longmemeval-cleaned` from HuggingFace
(`--hf-file longmemeval_oracle.json` by default — evidence-only sessions;
use longmemeval_s_cleaned.json for the full haystack) and needs
DASHSCOPE_API_KEY for the Qwen reader.

Scoring: exact-match containment of the gold answer (casefold), which is
conservative; abstention items (question_id ends with `_abs`) score correct
when the system abstains. A Qwen judge cross-check is recommended for live
runs — exact-match under-reports paraphrases (stated in the README).
"""

from __future__ import annotations

import json
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


def run_instance(inst: dict, config: str, *, budget: int, live: bool) -> dict:
    system = make_system(config, live)
    sessions = inst.get("haystack_sessions", [])
    for day, session in enumerate(sessions):
        if day > 0:
            system.day_boundary(day)
        text = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            for turn in session
        )
        system.ingest(text, day=day, kind="chat",
                      source_ref=f"session-{day}")
    system.day_boundary(len(sessions))
    ans = system.answer(inst["question"], budget)

    is_abstention = str(inst["question_id"]).endswith("_abs")
    gold = str(inst.get("answer", "")).strip()
    if is_abstention:
        # judged from the delivered TEXT for every config — never from a
        # config's self-reported flag (baselines cannot emit one)
        correct = looks_like_abstention(ans.text)
    else:
        # word-boundary match: bare containment lets short golds like
        # "before" match almost any verbose answer
        correct = bool(gold) and word_present(gold, ans.text)
    return {
        "question_id": inst["question_id"],
        "question_type": inst["question_type"],
        "abstention": is_abstention,
        "config": config,
        "correct": correct,
        "abstained": ans.abstained,
        "tokens_used": ans.tokens_used,
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
        abst = [r for r in sub if r["abstention"]]
        if abst:
            agg["abstention"] = round(
                sum(r["correct"] for r in abst) / len(abst), 4
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
