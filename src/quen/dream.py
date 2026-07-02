"""Dream pass (spec §4.6): re-abstract → self-test → supersede → decay/evict
→ compress → journal. Async-friendly, resilient (each phase is isolated), and
fully audited: every action lands in dream_actions and the audit log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from quen import fsrs, llm as llm_mod
from quen.config import QuenConfig
from quen.embeddings import Embedder, cosine
from quen.fsrs import Grade
from quen.llm import ChatLLM, parse_json_block
from quen.models import MemoryItem, make_memory, new_id
from quen.retrieval import recall
from quen.store import MemoryStore, iso
from quen.supersession import SupersessionEvent, deterministic_pass, nli_pass


@dataclass
class SelfTestResult:
    memory_id: str
    probe: str
    expected: str
    answer: str
    passed: bool
    ds: float  # stability delta from the review


@dataclass
class DreamReport:
    run_id: str
    generalization_ids: list[str] = field(default_factory=list)
    self_tests: list[SelfTestResult] = field(default_factory=list)
    supersessions: list[SupersessionEvent] = field(default_factory=list)
    evicted_ids: list[str] = field(default_factory=list)
    compressed_ids: list[str] = field(default_factory=list)
    journal: str = ""
    stats: dict = field(default_factory=dict)


def self_test_memory(
    mem: MemoryItem,
    *,
    store: MemoryStore,
    llm: ChatLLM,
    embedder: Embedder,
    cfg: QuenConfig,
    now: datetime,
    actor: str = "dream",
    run_id: Optional[str] = None,
) -> SelfTestResult:
    """Active recall on one memory: probe → answer from the active set →
    reinforce on pass / re-expand + fail review on failure.

    The tested memory deliberately stays in the recall pool: the test is
    whether the *pipeline* can surface it, not whether the LLM knows it.
    """
    # 1. probe — zero-LLM template for slotted memories, LLM otherwise
    if mem.triple is not None:
        s, r, o = mem.triple
        probe, expected = f"What is the current value: {s} {r} ...?", o
    else:
        raw = llm.complete(llm_mod.render_selftest_probe(mem.content), model_hint="fast")
        try:
            parsed = parse_json_block(raw)
            probe = str(parsed["probe"])
            expected = str(parsed["expected"])
        except (ValueError, KeyError, TypeError):
            probe, expected = f"What do you remember about: {mem.content[:60]}?", mem.content[:60]

    # 2. answer from the active set via the normal retrieval path
    rr = recall(
        probe,
        token_budget=cfg.default_token_budget,
        store=store,
        embedder=embedder,
        cfg=cfg,
        now=now,
    )
    context = "\n".join(sm.memory.content for sm in rr.used)
    answer = llm.complete(
        llm_mod.render_selftest_answer(probe, context), model_hint="fast"
    )

    # 3. grade — word-boundary match first (bare containment lets short
    # objects like "no" match inside "UNKNOWN"), LLM judge fallback
    passed = bool(
        re.search(
            rf"(?<!\w){re.escape(expected.strip())}(?!\w)", answer, re.IGNORECASE
        )
    )
    if not passed and answer.strip().upper() != "UNKNOWN":
        verdict = llm.complete(
            llm_mod.render_judge(probe, expected, answer), model_hint="fast"
        )
        passed = verdict.strip().upper().startswith("YES")

    # 4. review — self-test outcomes are FSRS reviews (spec §4.5b)
    elapsed = mem.elapsed_days(now)
    r_before = fsrs.retrievability(elapsed, mem.stability)
    before = (mem.difficulty, mem.stability)
    grade = Grade.GOOD if passed else Grade.AGAIN
    mem.difficulty, mem.stability = fsrs.review(*before, elapsed, grade)
    mem.last_review_at = now
    mem.review_count += 1
    if not passed:
        # re-expand verbatim + do-not-forget nudge
        if mem.content != mem.content_verbatim:
            mem.content = mem.content_verbatim
            mem.embedding = embedder.embed([mem.content])[0]  # keep ranking honest
        mem.importance = min(10.0, mem.importance + 1.0)
        mem.last_accessed_at = now
    store.update(
        mem,
        actor=actor,
        action="self_test",
        detail={"probe": probe, "passed": passed, "run_id": run_id},
    )
    store.record_review(
        mem.id,
        kind="self_test",
        grade=int(grade),
        elapsed_days=elapsed,
        r_before=r_before,
        before=before,
        after=(mem.difficulty, mem.stability),
        at=now,
    )
    store.record_calibration(
        "retention", predicted=r_before, outcome=passed, memory_id=mem.id, at=now
    )
    return SelfTestResult(
        memory_id=mem.id,
        probe=probe,
        expected=expected,
        answer=answer,
        passed=passed,
        ds=mem.stability - before[1],
    )


def run_dream(
    store: MemoryStore,
    llm: ChatLLM,
    embedder: Embedder,
    cfg: QuenConfig,
    *,
    now: datetime,
    actor: str = "dream",
) -> DreamReport:
    run_id = f"dream-{new_id()}"
    store.start_dream_run(run_id, at=now)
    report = DreamReport(run_id=run_id)

    def log(phase: str, detail: dict) -> None:
        store.log_dream_action(run_id, phase, detail, at=now)

    # ---- 1. re-abstract ---------------------------------------------------
    try:
        report.generalization_ids = _reabstract(
            store, llm, embedder, cfg, now=now, actor=actor, run_id=run_id, log=log
        )
    except Exception as exc:  # dream must be resilient
        log("error", {"phase": "reabstract", "error": repr(exc)})

    # Eviction candidacy is decided NOW, before self-tests run: the recall()
    # calls inside self-testing touch_access every retrieved memory, and a
    # review resets R — either would rescue every doomed memory in a small
    # store and make eviction unreachable. Dream introspection is not usage.
    evictable_ids = {m.id for m in store.active() if _evictable(m, cfg, now)}

    # ---- 2. self-test (lowest R first) ------------------------------------
    # Targets the at-risk-but-not-doomed band: eviction candidates are
    # excluded (see above), and so are fresh memories whose R hasn't decayed
    # below the desired retention yet (nothing to test).
    try:
        actives = [
            m
            for m in store.active()
            if m.id not in evictable_ids
            and fsrs.retrievability(m.elapsed_days(now), m.stability)
            < cfg.desired_retention
        ]
        actives.sort(
            key=lambda m: fsrs.retrievability(m.elapsed_days(now), m.stability)
        )
        for mem in actives[: cfg.selftest_sample_size]:
            result = self_test_memory(
                mem, store=store, llm=llm, embedder=embedder, cfg=cfg,
                now=now, actor=actor, run_id=run_id,
            )
            report.self_tests.append(result)
            log(
                "selftest",
                {
                    "memory_id": result.memory_id,
                    "probe": result.probe,
                    "expected": result.expected,
                    "passed": result.passed,
                    "ds": result.ds,
                },
            )
    except Exception as exc:
        log("error", {"phase": "selftest", "error": repr(exc)})

    # ---- 3. supersession: deterministic first, NLI fallback ---------------
    try:
        events = deterministic_pass(store, now=now, actor=actor, run_id=run_id)
        events += nli_pass(
            store, llm, embedder, cfg, now=now, actor=actor, run_id=run_id
        )
        report.supersessions = events
        for ev in events:
            log(
                "supersede",
                {
                    "old_id": ev.old_id,
                    "new_id": ev.new_id,
                    "rule": ev.rule,
                    "label": ev.label,
                    "nli_confidence": ev.nli_confidence,
                },
            )
    except Exception as exc:
        log("error", {"phase": "supersede", "error": repr(exc)})

    # ---- 4. decay + evict --------------------------------------------------
    # Uses the pre-dream snapshot: accesses caused by this run's own
    # introspection (self-test recalls) do not count as usage.
    try:
        for mem in store.active():
            if mem.id in evictable_ids and not mem.pinned:
                r = fsrs.retrievability(mem.elapsed_days(now), mem.stability)
                mem.status = "deprecated"
                mem.valid_to = now
                store.update(
                    mem,
                    actor=actor,
                    action="evict",
                    detail={"r": r, "run_id": run_id},
                )
                report.evicted_ids.append(mem.id)
                log("evict", {"memory_id": mem.id, "r": round(r, 4)})
    except Exception as exc:
        log("error", {"phase": "evict", "error": repr(exc)})

    # ---- 5. compress (keep verbatim) ---------------------------------------
    # Memories re-expanded by a failed self-test THIS run are exempt —
    # compressing them again would undo the do-not-forget recovery.
    reexpanded = {t.memory_id for t in report.self_tests if not t.passed}
    try:
        for mem in store.active():
            if mem.id in reexpanded:
                continue
            if len(mem.content) > cfg.compress_min_chars:
                compressed = llm.complete(
                    llm_mod.render_compress(mem.content), model_hint="fast"
                ).strip()
                if compressed and len(compressed) < len(mem.content):
                    mem.content = compressed
                    mem.embedding = embedder.embed([compressed])[0]
                    store.update(
                        mem, actor=actor, action="compress", detail={"run_id": run_id}
                    )
                    report.compressed_ids.append(mem.id)
                    log("compress", {"memory_id": mem.id, "chars": len(compressed)})
    except Exception as exc:
        log("error", {"phase": "compress", "error": repr(exc)})

    # ---- 6. journal ---------------------------------------------------------
    report.stats = {
        "reabstracted": len(report.generalization_ids),
        "self_tests": len(report.self_tests),
        "self_test_failures": sum(1 for t in report.self_tests if not t.passed),
        "superseded": len(report.supersessions),
        "evicted": len(report.evicted_ids),
        "compressed": len(report.compressed_ids),
    }
    try:
        report.journal = llm.complete(
            llm_mod.render_journal(report.stats), model_hint="fast"
        ).strip()
    except Exception as exc:
        report.journal = f"(journal unavailable: {exc!r})"
    store.finish_dream_run(run_id, journal=report.journal, stats=report.stats, at=now)
    return report


def _evictable(mem: MemoryItem, cfg: QuenConfig, now: datetime) -> bool:
    """Spec §4.5 eviction predicate: R < θ ∧ unaccessed past TTL ∧ ¬pinned."""
    if mem.pinned:
        return False
    r = fsrs.retrievability(mem.elapsed_days(now), mem.stability)
    unaccessed_days = (now - mem.last_accessed_at).total_seconds() / 86400.0
    return r < cfg.eviction_r_threshold and unaccessed_days > cfg.eviction_ttl_days


def _reabstract(
    store: MemoryStore,
    llm: ChatLLM,
    embedder: Embedder,
    cfg: QuenConfig,
    *,
    now: datetime,
    actor: str,
    run_id: str,
    log,
) -> list[str]:
    """Episodics → reusable generalizations, provenance kept, idempotent
    (episodics already covered by a generalization are not re-consumed)."""
    covered: set[str] = set()
    for sem in store.list(mtype="semantic", limit=100_000):
        covered.update(sem.provenance)
    uncovered = [
        m for m in store.active() if m.mtype == "episodic" and m.id not in covered
    ]
    if len(uncovered) < cfg.reabstract_min_episodics:
        return []

    payload = [
        {"index": i, "content": m.content} for i, m in enumerate(uncovered)
    ]
    raw = llm.complete(llm_mod.render_reabstract(payload), model_hint="chat")
    try:
        items = parse_json_block(raw)
    except ValueError as exc:
        log("error", {"phase": "reabstract", "error": f"bad JSON: {exc}"})
        return []

    created: list[str] = []
    actives = store.active()
    for item in items if isinstance(items, list) else []:
        content = str(item.get("content", "")).strip()
        indices = [
            i for i in item.get("source_indices", [])
            if isinstance(i, int) and 0 <= i < len(uncovered)
        ]
        if not content or not indices:
            continue
        sources = [uncovered[i] for i in indices]
        triple = item.get("triple")
        # same validation bar as the write path — a degenerate triple (empty
        # or non-string part) would mint a slot that supersedes real facts
        triple_t: Optional[tuple[str, str, str]] = None
        if (
            isinstance(triple, (list, tuple))
            and len(triple) == 3
            and all(isinstance(p, str) and p.strip() for p in triple)
        ):
            triple_t = (triple[0], triple[1], triple[2])

        # dedup before storing: triple slot first, cosine second
        if triple_t is not None and store.find_by_triple(triple_t) is not None:
            continue
        emb = embedder.embed([content])[0]
        if any(cosine(emb, m.embedding) >= cfg.dedup_cosine_threshold for m in actives):
            continue

        d0, s0 = fsrs.initial_state(Grade.GOOD)
        gen = make_memory(
            content,
            embedding=emb,
            mtype="semantic",
            triple=triple_t,
            importance=max(s.importance for s in sources),
            salience=max(s.salience for s in sources),
            difficulty=d0,
            stability=s0,
            confidence=min(s.confidence for s in sources),
            source_ref=f"dream:{run_id}",
            provenance=[s.id for s in sources],
            # a generalization is only as current as its newest evidence —
            # dating it `now` would let stale episodics supersede newer facts
            valid_from=max(s.valid_from for s in sources),
            now=now,
        )
        store.add(gen, actor=actor, detail={"run_id": run_id, "phase": "reabstract"})
        created.append(gen.id)
        actives.append(gen)  # later items this run dedup against it too
        log(
            "reabstract",
            {
                "memory_id": gen.id,
                "generalization": content,
                "provenance": [s.id for s in sources],
            },
        )
    return created
