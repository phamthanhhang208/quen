"""QuenEngine — the facade every surface (FastAPI, MCP, eval) codes against.

`ask()` implements the special mechanism (spec §4.7): retrieval → trust gate →
verify-before-answer → hedged answer whose stated confidence tracks the trust
of what it used — and every verification outcome feeds back into retention as
an FSRS review.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

from quen import fsrs, llm as llm_mod
from quen.config import QuenConfig
from quen.dream import DreamReport, SelfTestResult, run_dream, self_test_memory
from quen.embeddings import Embedder, HashingEmbedder
from quen.fsrs import Grade
from quen.llm import ChatLLM, ScriptedLLM
from quen.models import MemoryItem, new_id, utcnow
from quen.retrieval import RecallResult, ScoredMemory, recall
from quen.store import MemoryStore, iso
from quen.trust import (
    answer_confidence,
    apply_verification,
    hedge_phrase,
    hedging_instruction,
    trust_score,
)
from quen.verifiers import RepoGrepVerifier, VerificationEvent, Verifier


@dataclass
class AskResult:
    answer: str
    answer_confidence: float
    trace_id: str
    abstained: bool
    used: list[ScoredMemory] = field(default_factory=list)
    verifications: list[VerificationEvent] = field(default_factory=list)
    tokens_used: int = 0


class QuenEngine:
    def __init__(
        self,
        cfg: Optional[QuenConfig] = None,
        *,
        store: Optional[MemoryStore] = None,
        llm: Optional[ChatLLM] = None,
        embedder: Optional[Embedder] = None,
        verifiers: Optional[Sequence[Verifier]] = None,
        clock: Callable[[], datetime] = utcnow,
    ):
        self.cfg = cfg or QuenConfig()
        self.clock = clock
        self.store = store or MemoryStore(self.cfg.db_path, clock=clock)
        if llm is None or embedder is None:
            raise ValueError(
                "QuenEngine requires explicit llm and embedder — use "
                "get_engine() for env-driven wiring (Qwen or offline)."
            )
        self.llm = llm
        self.embedder = embedder
        self.verifiers: list[Verifier] = list(verifiers or [])

    # ------------------------------------------------------------------ write

    def ingest(
        self,
        text: str,
        *,
        source_ref: Optional[str] = None,
        source_kind: str = "chat",
    ):
        from quen.write_pipeline import ingest_observation

        return ingest_observation(
            text,
            store=self.store,
            llm=self.llm,
            embedder=self.embedder,
            cfg=self.cfg,
            now=self.clock(),
            source_ref=source_ref,
            source_kind=source_kind,
        )

    # ------------------------------------------------------------------- ask

    def ask(self, query: str, *, token_budget: Optional[int] = None) -> AskResult:
        now = self.clock()
        budget = token_budget or self.cfg.default_token_budget
        rr = recall(
            query,
            token_budget=budget,
            store=self.store,
            embedder=self.embedder,
            cfg=self.cfg,
            now=now,
        )

        # ---- trust gate + verify-before-answer (spec §4.7) ----
        verifications: list[VerificationEvent] = []
        if self.verifiers:
            low_trust = sorted(
                (sm for sm in rr.used if sm.trust < self.cfg.trust_threshold),
                key=lambda sm: sm.trust,
            )[: self.cfg.verify_max_per_ask]
            any_refuted = False
            for sm in low_trust:
                event = self._run_verifiers(sm.memory, now)
                if event is None:
                    continue
                verifications.append(event)
                if event.outcome == "refuted":
                    any_refuted = True
            if any_refuted:
                # answer from the corrected state: refuted memories are now
                # tombstoned, so one re-recall backfills the freed budget.
                rr = recall(
                    query,
                    token_budget=budget,
                    store=self.store,
                    embedder=self.embedder,
                    cfg=self.cfg,
                    now=now,
                )

        confirmed_ids = {v.memory_id for v in verifications if v.outcome == "confirmed"}
        context_lines = []
        for sm in rr.used:
            mem = sm.memory
            trust = max(sm.trust, 0.9) if mem.id in confirmed_ids else sm.trust
            tag = (
                f"[trust {trust:.2f} | confidence {sm.confidence:.2f} | "
                f"{sm.freshness_days:.0f}d old"
                + (" | verified against live source just now" if mem.id in confirmed_ids else "")
                + "]"
            )
            hedge = hedge_phrase(trust, mem.source_ref)
            line = f"- {tag} {mem.content}"
            if hedge:
                line += f" (hedge: {hedge})"
            context_lines.append(line)

        answer = self.llm.complete(
            llm_mod.render_answer(query, context_lines, hedging_instruction()),
            model_hint="chat",
        ).strip()
        conf = answer_confidence(rr.used, verifications)
        abstained = (not rr.used) or conf < 0.25
        if not rr.used:
            answer = "I don't have a reliable memory about that."

        trace_id = f"trace-{new_id()}"
        self.store.save_trace(
            {
                "trace_id": trace_id,
                "at": iso(now),
                "query": query,
                "answer": answer,
                "answer_confidence": round(conf, 4),
                "abstained": abstained,
                "token_budget": budget,
                "tokens_used": rr.tokens_used,
                "used": [
                    {
                        "memory_id": sm.memory.id,
                        "snippet": _snippet(sm.memory.content),
                        "relevance": round(sm.relevance, 4),
                        "retrievability": round(sm.retrievability, 4),
                        "importance_norm": round(sm.importance_norm, 4),
                        "score": round(sm.score, 4),
                        "tokens": sm.tokens,
                        "trust": round(sm.trust, 4),
                        "confidence": round(sm.confidence, 4),
                        "freshness_days": round(sm.freshness_days, 2),
                    }
                    for sm in rr.used
                ],
                "excluded": rr.excluded_relevant,
                "counterfactual": rr.counterfactual,
                "verifications": [_event_dict(v) for v in verifications],
            }
        )
        return AskResult(
            answer=answer,
            answer_confidence=round(conf, 4),
            trace_id=trace_id,
            abstained=abstained,
            used=rr.used,
            verifications=verifications,
            tokens_used=rr.tokens_used,
        )

    def _run_verifiers(
        self, mem: MemoryItem, now: datetime
    ) -> Optional[VerificationEvent]:
        for verifier in self.verifiers:
            if verifier.can_verify(mem):
                outcome, evidence = verifier.verify(mem)
                return apply_verification(
                    mem,
                    outcome,
                    store=self.store,
                    cfg=self.cfg,
                    now=now,
                    evidence=evidence,
                    verifier=verifier.name,
                )
        return None

    # ----------------------------------------------------------- maintenance

    def judge_answer(self, trace_id: str, correct: bool) -> None:
        """Use-in-answer review (spec §4.5a) + confidence calibration event."""
        now = self.clock()
        trace = self.store.get_trace(trace_id)
        if trace is None:
            raise KeyError(f"trace not found: {trace_id}")
        grade = Grade.GOOD if correct else Grade.AGAIN
        for used in trace["used"]:
            mem = self.store.get(used["memory_id"])
            if mem is None or mem.status != "active":
                continue
            elapsed = mem.elapsed_days(now)
            r_before = fsrs.retrievability(elapsed, mem.stability)
            before = (mem.difficulty, mem.stability)
            mem.difficulty, mem.stability = fsrs.review(*before, elapsed, grade)
            mem.last_review_at = now
            mem.review_count += 1
            self.store.update(
                mem,
                actor="engine",
                action="use_judged",
                detail={"trace_id": trace_id, "correct": correct},
            )
            self.store.record_review(
                mem.id,
                kind="use_judged",
                grade=int(grade),
                elapsed_days=elapsed,
                r_before=r_before,
                before=before,
                after=(mem.difficulty, mem.stability),
                at=now,
            )
        if trace.get("answer_confidence") is not None:
            freshness = [u.get("freshness_days") for u in trace["used"]]
            freshness = [f for f in freshness if f is not None]
            self.store.record_calibration(
                "confidence",
                predicted=trace["answer_confidence"],
                outcome=correct,
                freshness_days=min(freshness) if freshness else None,
                trace_id=trace_id,
                at=now,
            )

    def dream(self) -> DreamReport:
        return run_dream(
            self.store, self.llm, self.embedder, self.cfg, now=self.clock()
        )

    def verify(self, memory_id: str) -> VerificationEvent:
        """Verify a memory against its live source right now (dashboard button)."""
        now = self.clock()
        mem = self._get_or_raise(memory_id)
        event = self._run_verifiers(mem, now)
        if event is None:
            event = apply_verification(
                mem,
                "unverifiable",
                store=self.store,
                cfg=self.cfg,
                now=now,
                evidence="no verifier available",
                verifier="none",
            )
        return event

    def verify_hint(
        self,
        memory_id: str,
        result: str,
        evidence: Optional[str] = None,
    ) -> VerificationEvent:
        """A harness reports a verification outcome it observed (MCP §5) —
        the engine closes the loop: confidence update + FSRS review."""
        if result not in ("confirmed", "refuted", "unverifiable"):
            raise ValueError(f"invalid verification outcome: {result!r}")
        mem = self._get_or_raise(memory_id)
        return apply_verification(
            mem,
            result,
            store=self.store,
            cfg=self.cfg,
            now=self.clock(),
            evidence=evidence,
            verifier="hint",
        )

    def pin(self, memory_id: str, pinned: bool = True) -> None:
        mem = self._get_or_raise(memory_id)
        mem.pinned = pinned
        self.store.update(
            mem, actor="engine", action="pin" if pinned else "unpin"
        )

    def self_test(self, memory_id: str) -> SelfTestResult:
        mem = self._get_or_raise(memory_id)
        return self_test_memory(
            mem,
            store=self.store,
            llm=self.llm,
            embedder=self.embedder,
            cfg=self.cfg,
            now=self.clock(),
            actor="engine",
        )

    # -------------------------------------------------------------- inspect

    def inspect(
        self,
        *,
        status: Optional[str] = None,
        mtype: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        now = self.clock()
        return [
            self.memory_summary(m, now)
            for m in self.store.list(status=status, mtype=mtype, q=q, limit=limit)
        ]

    def memory_summary(self, mem: MemoryItem, now: Optional[datetime] = None) -> dict:
        now = now or self.clock()
        return {
            "id": mem.id,
            "snippet": _snippet(mem.content),
            "mtype": mem.mtype,
            "status": mem.status,
            "importance": mem.importance,
            "salience": mem.salience,
            "difficulty": round(mem.difficulty, 3),
            "stability": round(mem.stability, 3),
            "last_review_at": iso(mem.last_review_at),
            "review_count": mem.review_count,
            "confidence": round(mem.confidence, 3),
            "freshness_days": round(mem.freshness_days(now), 2),
            "last_verified_at": iso(mem.last_verified_at),
            "pinned": mem.pinned,
            "valid_from": iso(mem.valid_from),
            "valid_to": iso(mem.valid_to),
            "superseded_by": mem.superseded_by,
            "source_ref": mem.source_ref,
            "created_at": iso(mem.created_at),
            "triple": list(mem.triple) if mem.triple else None,
        }

    def memory_detail(self, memory_id: str) -> dict:
        mem = self._get_or_raise(memory_id)
        now = self.clock()
        detail = self.memory_summary(mem, now)
        detail["content"] = mem.content
        detail["content_verbatim"] = mem.content_verbatim
        provenance = []
        for pid in mem.provenance:
            src = self.store.get(pid)
            if src is not None:
                provenance.append({"id": pid, "snippet": _snippet(src.content)})
        detail["provenance"] = provenance
        detail["reviews"] = self.store.reviews_for(memory_id)
        detail["verifications"] = [
            {
                "at": row["at"],
                "verifier": (row["detail"] or {}).get("verifier", "?"),
                "outcome": (row["detail"] or {}).get("outcome", "?"),
                "evidence": (row["detail"] or {}).get("evidence"),
            }
            for row in reversed(self.store.audit_tail(memory_id=memory_id, action="verify"))
        ]
        detail["audit"] = self.store.audit_tail(memory_id=memory_id, limit=30)
        return detail

    # --------------------------------------------------------------- vitals

    def vitals(self) -> dict:
        now = self.clock()
        actives = self.store.active()
        r_values = [
            fsrs.retrievability(m.elapsed_days(now), m.stability) for m in actives
        ]
        s_values = [m.stability for m in actives]
        return {
            "counts_by_status": self.store.counts_by_status(),
            "r_histogram": _histogram_fixed(r_values),
            "s_histogram": _histogram_log(s_values),
            "kpis": _load_eval_kpis(),
            "retention_calibration": _calibration_bins(
                self.store.calibration_events("retention")
            ),
            "confidence_by_freshness_bucket": _confidence_strata(
                self.store.calibration_events("confidence")
            ),
        }

    # --------------------------------------------------------------- helpers

    def _get_or_raise(self, memory_id: str) -> MemoryItem:
        mem = self.store.get(memory_id)
        if mem is None:
            raise KeyError(f"memory not found: {memory_id}")
        return mem


# ----------------------------------------------------------- module helpers

def _snippet(text: str, n: int = 120) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _event_dict(v: VerificationEvent) -> dict:
    return {
        "memory_id": v.memory_id,
        "verifier": v.verifier,
        "outcome": v.outcome,
        "evidence": v.evidence,
        "at": iso(v.at),
    }


def _histogram_fixed(values: list[float]) -> list[dict]:
    """Ten fixed buckets over [0, 1] — for R."""
    buckets = [0] * 10
    for v in values:
        buckets[min(9, int(v * 10))] += 1
    return [
        {"bucket": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "count": buckets[i]}
        for i in range(10)
    ]


_S_EDGES = [0, 1, 3, 7, 14, 30, 90]


def _histogram_log(values: list[float]) -> list[dict]:
    """Roughly-log buckets in days — for S."""
    labels = [
        f"{lo}-{hi}d" for lo, hi in zip(_S_EDGES, _S_EDGES[1:])
    ] + [f">{_S_EDGES[-1]}d"]
    buckets = [0] * len(labels)
    for v in values:
        idx = len(_S_EDGES) - 1
        for i, hi in enumerate(_S_EDGES[1:]):
            if v < hi:
                idx = i
                break
        else:
            idx = len(labels) - 1
        buckets[idx] += 1
    return [{"bucket": b, "count": c} for b, c in zip(labels, buckets)]


def _calibration_bins(events: list[dict], n_bins: int = 10) -> list[dict]:
    bins: list[list] = [[] for _ in range(n_bins)]
    for e in events:
        idx = min(n_bins - 1, int(e["predicted"] * n_bins))
        bins[idx].append(e)
    out = []
    for i, chunk in enumerate(bins):
        if not chunk:
            continue
        out.append(
            {
                "bin": f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}",
                "predicted_mean": round(
                    sum(e["predicted"] for e in chunk) / len(chunk), 4
                ),
                "empirical": round(
                    sum(e["outcome"] for e in chunk) / len(chunk), 4
                ),
                "count": len(chunk),
            }
        )
    return out


_FRESHNESS_BUCKETS = [("<7d", 0.0, 7.0), ("7-30d", 7.0, 30.0), (">30d", 30.0, float("inf"))]


def _confidence_strata(events: list[dict]) -> list[dict]:
    out = []
    for label, lo, hi in _FRESHNESS_BUCKETS:
        chunk = [
            e
            for e in events
            if e.get("freshness_days") is not None and lo <= e["freshness_days"] < hi
        ]
        bins = _calibration_bins(chunk)
        total = sum(b["count"] for b in bins)
        ece = (
            round(
                sum(
                    abs(b["predicted_mean"] - b["empirical"]) * b["count"]
                    for b in bins
                )
                / total,
                4,
            )
            if total
            else None
        )
        out.append(
            {"freshness_bucket": label, "bins": bins, "ece": ece, "count": total}
        )
    return out


def _load_eval_kpis() -> dict:
    """KPI tiles come from a real eval run (eval/out/summary.json) or stay
    null — the dashboard never shows fabricated numbers."""
    path = Path(os.environ.get("QUEN_EVAL_SUMMARY", "eval/out/summary.json"))
    kpis = {
        "fama": None,
        "forgetting_precision": None,
        "forgetting_recall": None,
        "avg_tokens_per_query": None,
    }
    if path.exists():
        try:
            data = json.loads(path.read_text())
            for key in kpis:
                if key in data:
                    kpis[key] = data[key]
        except (json.JSONDecodeError, OSError):
            pass
    return kpis


# ------------------------------------------------------------- env bootstrap

_engine: Optional[QuenEngine] = None


def get_engine() -> QuenEngine:
    """Process singleton wired from env. Offline (QUEN_OFFLINE=1 or no
    DASHSCOPE_API_KEY) → deterministic stack; otherwise Qwen on DashScope."""
    global _engine
    if _engine is not None:
        return _engine
    cfg = QuenConfig.from_env()
    offline = os.environ.get("QUEN_OFFLINE") == "1" or not os.environ.get(
        "DASHSCOPE_API_KEY"
    )
    if offline:
        llm: ChatLLM = ScriptedLLM.with_offline_defaults()
        embedder: Embedder = HashingEmbedder()
    else:
        from quen.embeddings import QwenEmbedder
        from quen.llm import QwenLLM

        llm = QwenLLM(cfg.chat_model, cfg.fast_model)
        embedder = QwenEmbedder(cfg.embed_model)
    verifiers: list[Verifier] = []
    repo = os.environ.get("QUEN_VERIFY_REPO")
    if repo:
        verifiers.append(RepoGrepVerifier(repo))
    _engine = QuenEngine(
        cfg, llm=llm, embedder=embedder, verifiers=verifiers
    )
    return _engine


def reset_engine() -> None:
    global _engine
    _engine = None
