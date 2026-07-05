"""FastAPI surface. DTO shapes are FROZEN together with
dashboard/src/lib/types.ts — change both or neither.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from quen import fsrs
from quen.engine import QuenEngine, _event_dict, _snippet, get_engine


class IngestBody(BaseModel):
    text: str
    source_ref: Optional[str] = None
    source_kind: str = "chat"


class AskBody(BaseModel):
    query: str
    token_budget: Optional[int] = Field(default=None, gt=0)


class RecallBody(BaseModel):
    query: str
    token_budget: int = Field(default=1500, gt=0)


class PinBody(BaseModel):
    pinned: bool = True


class JudgeBody(BaseModel):
    correct: bool


class VerifyHintBody(BaseModel):
    result: str
    evidence: Optional[str] = None


def create_app(engine: Optional[QuenEngine] = None) -> FastAPI:
    app = FastAPI(title="Quên · memory", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine = engine

    def eng() -> QuenEngine:
        if app.state.engine is None:
            app.state.engine = get_engine()
        return app.state.engine

    # ------------------------------------------------------------- config

    @app.get("/config")
    def config():
        cfg = eng().cfg
        return {
            "fsrs_version": fsrs.VERSION,
            "decay": fsrs.DECAY,
            "factor": fsrs.FACTOR,
            "eviction_r_threshold": cfg.eviction_r_threshold,
            "eviction_ttl_days": cfg.eviction_ttl_days,
            "trust_threshold": cfg.trust_threshold,
            "freshness_half_life_days": cfg.freshness_half_life_days,
            "default_token_budget": cfg.default_token_budget,
        }

    # ------------------------------------------------------------- memory

    @app.get("/memory")
    def list_memory(
        status: Optional[str] = None,
        mtype: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 500,
    ):
        return eng().inspect(status=status, mtype=mtype, q=q, limit=limit)

    @app.get("/memory/{memory_id}")
    def get_memory(memory_id: str):
        try:
            return eng().memory_detail(memory_id)
        except KeyError:
            raise HTTPException(404, f"memory not found: {memory_id}")

    @app.post("/memory/{memory_id}/pin")
    def pin_memory(memory_id: str, body: PinBody):
        try:
            eng().pin(memory_id, body.pinned)
        except KeyError:
            raise HTTPException(404, f"memory not found: {memory_id}")
        return {"id": memory_id, "pinned": body.pinned}

    @app.post("/memory/{memory_id}/recall")
    def force_self_test(memory_id: str):
        try:
            result = eng().self_test(memory_id)
        except KeyError:
            raise HTTPException(404, f"memory not found: {memory_id}")
        return {
            "memory_id": result.memory_id,
            "probe": result.probe,
            "expected": result.expected,
            "answer": result.answer,
            "passed": result.passed,
            "ds": round(result.ds, 4),
        }

    @app.post("/memory/{memory_id}/verify")
    def verify_now(memory_id: str):
        try:
            event = eng().verify(memory_id)
        except KeyError:
            raise HTTPException(404, f"memory not found: {memory_id}")
        return _event_dict(event)

    @app.post("/memory/{memory_id}/verify_hint")
    def verify_hint(memory_id: str, body: VerifyHintBody):
        try:
            event = eng().verify_hint(memory_id, body.result, body.evidence)
        except KeyError:
            raise HTTPException(404, f"memory not found: {memory_id}")
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return _event_dict(event)

    # ---------------------------------------------------------- ingest/ask

    @app.post("/ingest")
    def ingest(body: IngestBody):
        result = eng().ingest(
            body.text, source_ref=body.source_ref, source_kind=body.source_kind
        )
        return {
            "stored": [
                {"id": m.id, "snippet": _snippet(m.content)} for m in result.stored
            ],
            "reinforced": result.reinforced,
            "skipped": [
                {"content": s.content, "reason": s.reason} for s in result.skipped
            ],
        }

    @app.post("/recall")
    def recall(body: RecallBody):
        """Budgeted recall with trust fields, no LLM call — same payload as
        the MCP `recall` tool (used by the harness hooks kit)."""
        return eng().recall_dict(body.query, token_budget=body.token_budget)

    @app.post("/ask")
    def ask(body: AskBody):
        result = eng().ask(body.query, token_budget=body.token_budget)
        return {
            "trace_id": result.trace_id,
            "answer": result.answer,
            "answer_confidence": result.answer_confidence,
            "abstained": result.abstained,
        }

    @app.post("/trace/{trace_id}/judge")
    def judge(trace_id: str, body: JudgeBody):
        try:
            eng().judge_answer(trace_id, body.correct)
        except KeyError:
            raise HTTPException(404, f"trace not found: {trace_id}")
        return {"trace_id": trace_id, "judged_correct": body.correct}

    # -------------------------------------------------------------- dream

    @app.post("/dream")
    def dream():
        report = eng().dream()
        return {"run_id": report.run_id, "stats": report.stats}

    @app.get("/dream/log")
    def dream_log(limit: int = 20):
        return eng().store.dream_runs(limit=limit)

    @app.get("/dream/{run_id}")
    def dream_run(run_id: str):
        run = eng().store.dream_run(run_id)
        if run is None:
            raise HTTPException(404, f"dream run not found: {run_id}")
        return run

    # -------------------------------------------------------------- traces

    @app.get("/recall/trace/latest")
    def latest_trace():
        trace = eng().store.latest_trace()
        if trace is None:
            raise HTTPException(404, "no traces yet")
        return trace

    @app.get("/recall/trace/{trace_id}")
    def get_trace(trace_id: str):
        trace = eng().store.get_trace(trace_id)
        if trace is None:
            raise HTTPException(404, f"trace not found: {trace_id}")
        return trace

    # -------------------------------------------------------------- vitals

    @app.get("/vitals")
    def vitals():
        return eng().vitals()

    # ----------------------------------------------------------- dashboard
    # Single-origin deploys: point QUEN_DASHBOARD_DIST at a dashboard build
    # (made with VITE_API_BASE="") and the API serves it too. Mounted last,
    # so every API route above still wins.
    dist = os.environ.get("QUEN_DASHBOARD_DIST", "")
    if dist and Path(dist, "index.html").is_file():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")

    return app


def main() -> None:  # quen-api console script
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("QUEN_HOST", "0.0.0.0"),
        port=int(os.environ.get("QUEN_PORT", "8000")),
    )


app = create_app()
