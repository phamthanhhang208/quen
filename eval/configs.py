"""The three memory systems under identical reader/judge/budget (spec §9):
append-only RAG (no forgetting) · full-context (truncate oldest-first) ·
Quên (ours, with a `verify=False` ablation).

Every system answers through the SAME reader prompt (quen.llm.render_answer)
so differences measure memory management, not prompting.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from quen import llm as llm_mod
from quen.config import QuenConfig
from quen.embeddings import Embedder, cosine
from quen.engine import QuenEngine
from quen.llm import ChatLLM
from quen.retrieval import estimate_tokens
from quen.store import MemoryStore
from quen.verifiers import RepoGrepVerifier

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

_TURN_RE = re.compile(r"^(?:user|assistant|system)\s*:", re.MULTILINE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
CHUNK_TOKENS = 120


def split_units(text: str) -> list[str]:
    """Retrieval granularity for the baselines.

    Multi-turn session transcripts (the LongMemEval shape: "role: content"
    lines) are split per turn — the round-level granularity the LongMemEval
    paper itself recommends over session-level — and turns longer than
    CHUNK_TOKENS are further packed into sentence windows, so a small
    reader budget can always fit SOME evidence. Without this, a whole
    2-5k-token session is the smallest storable unit and a 300-token
    budget fits none of them: the baseline answers from an empty context,
    which measures nothing. Single-fact texts (the probe shape) come back
    unchanged, so probe results are unaffected.
    """
    starts = [m.start() for m in _TURN_RE.finditer(text)]
    if not starts:
        turns = [text]
    else:
        if starts[0] != 0:
            starts = [0] + starts
        turns = [
            text[a:b].strip()
            for a, b in zip(starts, starts[1:] + [len(text)])
        ]
    units: list[str] = []
    for turn in turns:
        if not turn:
            continue
        if estimate_tokens(turn) <= CHUNK_TOKENS:
            units.append(turn)
            continue
        m = _TURN_RE.match(turn)
        prefix = (turn[: m.end()] + " ") if m else ""
        body = turn[m.end():].strip() if m else turn
        window = ""
        for sent in _SENT_RE.split(body):
            if window and estimate_tokens(f"{window} {sent}") > CHUNK_TOKENS:
                units.append(prefix + window)
                window = sent
            else:
                window = f"{window} {sent}".strip()
        if window:
            units.append(prefix + window)
    return units


@dataclass
class AnswerOut:
    text: str
    confidence: Optional[float]
    tokens_used: int
    abstained: bool
    freshness_max: Optional[float] = None  # stalest memory relied upon (days)


class MemorySystem(Protocol):
    name: str

    def ingest(self, text: str, *, day: int, kind: str,
               source_ref: Optional[str]) -> None: ...

    def day_boundary(self, day: int) -> None: ...

    def answer(self, question: str, budget: int) -> AnswerOut: ...


class AppendOnlyRAG:
    """Top-k cosine over everything ever seen (turn-level units), greedy to
    the same budget. No forgetting, no validity, no trust."""

    name = "append_only"

    def __init__(self, llm: ChatLLM, embedder: Embedder):
        self.llm, self.embedder = llm, embedder
        self.entries: list[tuple[str, list[float]]] = []

    def ingest(self, text, *, day, kind, source_ref):
        units = split_units(text)
        self.entries.extend(zip(units, self.embedder.embed(units)))

    def day_boundary(self, day):
        pass

    def answer(self, question, budget) -> AnswerOut:
        q = self.embedder.embed([question])[0]
        ranked = sorted(
            self.entries, key=lambda e: cosine(q, e[1]), reverse=True
        )
        lines, used = [], 0
        for text, _ in ranked:
            t = estimate_tokens(text)
            if used + t > budget:
                continue
            lines.append(f"- {text}")
            used += t
        answer = self.llm.complete(
            llm_mod.render_answer(question, lines, ""), model_hint="chat"
        )
        # report what the reader actually saw (same accounting as Quen's
        # prompt_tokens — trust tags and all)
        prompt_tokens = sum(estimate_tokens(line) for line in lines)
        return AnswerOut(answer, None, prompt_tokens, abstained=not lines)


class FullContext:
    """Everything in the window (turn-level units), oldest truncated first
    when over budget."""

    name = "full_context"

    def __init__(self, llm: ChatLLM, embedder: Embedder):
        self.llm = llm
        self.texts: list[str] = []

    def ingest(self, text, *, day, kind, source_ref):
        self.texts.extend(split_units(text))

    def day_boundary(self, day):
        pass

    def answer(self, question, budget) -> AnswerOut:
        kept: list[str] = []
        used = 0
        for text in reversed(self.texts):  # newest first, truncate oldest
            t = estimate_tokens(text)
            if used + t > budget:
                break
            kept.append(text)
            used += t
        lines = [f"- {t}" for t in reversed(kept)]  # restore chronology
        answer = self.llm.complete(
            llm_mod.render_answer(question, lines, ""), model_hint="chat"
        )
        prompt_tokens = sum(estimate_tokens(line) for line in lines)
        return AnswerOut(answer, None, prompt_tokens, abstained=not lines)


class Quen:
    """Ours: the full engine — salience-gated writes, dream reconciliation,
    relevance-dominant budgeted recall, trust gate (+ verify when a live
    repo snapshot exists for the case)."""

    def __init__(
        self,
        llm: ChatLLM,
        embedder: Embedder,
        *,
        verify: bool = True,
        repo_path: Optional[str] = None,
    ):
        self.name = "quen" if verify else "quen_no_verify"
        self._now = T0
        verifiers = (
            [RepoGrepVerifier(repo_path)] if (verify and repo_path) else []
        )
        # Haystack-scale histories (LongMemEval-S: 40-50 sessions) need the
        # write-count dream trigger — dreaming at EVERY session boundary is
        # neither realistic nor affordable there. 0 (default) = legacy
        # dream-per-boundary, keeping oracle/probe runs bit-identical.
        dream_every = int(os.environ.get("QUEN_EVAL_DREAM_EVERY", "0"))
        self.engine = QuenEngine(
            QuenConfig(db_path=":memory:",
                       dream_every_n_ingests=dream_every),
            store=MemoryStore(":memory:", clock=lambda: self._now),
            llm=llm,
            embedder=embedder,
            verifiers=verifiers,
            clock=lambda: self._now,
        )

    def _set_day(self, day: int) -> None:
        self._now = T0 + timedelta(days=day)

    def ingest(self, text, *, day, kind, source_ref):
        self._set_day(day)
        self.engine.ingest(text, source_kind=kind, source_ref=source_ref)

    def day_boundary(self, day):
        self._set_day(day)
        if self.engine.cfg.dream_every_n_ingests > 0:
            self.engine.maybe_dream()
        else:
            self.engine.dream()

    def final_boundary(self, day):
        """The pre-answer consolidation always runs in full."""
        self._set_day(day)
        self.engine.dream()

    def answer(self, question, budget) -> AnswerOut:
        res = self.engine.ask(question, token_budget=budget)
        # prompt_tokens includes the trust tags/hedges Quen adds — the same
        # what-the-reader-saw accounting the baselines report
        return AnswerOut(
            res.answer,
            res.answer_confidence,
            res.prompt_tokens,
            res.abstained,
            freshness_max=(
                max(sm.freshness_days for sm in res.used) if res.used else None
            ),
        )
