"""The three memory systems under identical reader/judge/budget (spec §9):
append-only RAG (no forgetting) · full-context (truncate oldest-first) ·
Quên (ours, with a `verify=False` ablation).

Every system answers through the SAME reader prompt (quen.llm.render_answer)
so differences measure memory management, not prompting.
"""

from __future__ import annotations

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
    """Top-k cosine over everything ever seen, greedy to the same budget.
    No forgetting, no validity, no trust."""

    name = "append_only"

    def __init__(self, llm: ChatLLM, embedder: Embedder):
        self.llm, self.embedder = llm, embedder
        self.entries: list[tuple[str, list[float]]] = []

    def ingest(self, text, *, day, kind, source_ref):
        self.entries.append((text, self.embedder.embed([text])[0]))

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
    """Everything in the window, oldest truncated first when over budget."""

    name = "full_context"

    def __init__(self, llm: ChatLLM, embedder: Embedder):
        self.llm = llm
        self.texts: list[str] = []

    def ingest(self, text, *, day, kind, source_ref):
        self.texts.append(text)

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
        self.engine = QuenEngine(
            QuenConfig(db_path=":memory:"),
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
