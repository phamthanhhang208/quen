"""SQLite persistence, audit log, review history, dream runs, recall traces,
calibration events.

Invariants:
- Memories are NEVER hard-deleted. There is no delete path in this module;
  ``status`` transitions (active → deprecated | superseded) are the only way
  a memory leaves circulation. A test greps the source tree for the SQL
  delete statement to keep it that way.
- Every mutation writes an audit row.
- One connection guarded by a lock held per-statement (never across LLM
  calls); WAL mode so the API process tolerates concurrent readers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from quen.models import MemoryItem, _norm, utcnow

SEP = "\x1f"  # slot/triple key joiner — cannot collide with content text

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  content_verbatim TEXT NOT NULL,
  triple_s TEXT, triple_r TEXT, triple_o TEXT,
  slot_key TEXT,
  triple_key TEXT,
  embedding TEXT NOT NULL,
  mtype TEXT NOT NULL CHECK (mtype IN ('episodic','semantic')),
  importance REAL NOT NULL,
  salience REAL NOT NULL,
  difficulty REAL NOT NULL,
  stability REAL NOT NULL,
  last_review_at TEXT NOT NULL,
  review_count INTEGER NOT NULL DEFAULT 0,
  pinned INTEGER NOT NULL DEFAULT 0,
  valid_from TEXT NOT NULL,
  valid_to TEXT,
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active','deprecated','superseded')),
  superseded_by TEXT,
  confidence REAL NOT NULL,
  last_verified_at TEXT,
  source_ref TEXT,
  created_at TEXT NOT NULL,
  last_accessed_at TEXT NOT NULL,
  provenance TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_mem_slot ON memories(slot_key) WHERE slot_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_mem_triple ON memories(triple_key) WHERE triple_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS audit_log (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  at TEXT NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  memory_id TEXT,
  run_id TEXT,
  detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_mem ON audit_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);

CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id TEXT NOT NULL,
  at TEXT NOT NULL,
  kind TEXT NOT NULL,
  grade INTEGER NOT NULL,
  elapsed_days REAL NOT NULL,
  r_before REAL NOT NULL,
  d_before REAL NOT NULL, s_before REAL NOT NULL,
  d_after REAL NOT NULL, s_after REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_mem ON reviews(memory_id);

CREATE TABLE IF NOT EXISTS dream_runs (
  run_id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  journal TEXT,
  stats TEXT
);
CREATE TABLE IF NOT EXISTS dream_actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  at TEXT NOT NULL,
  phase TEXT NOT NULL,
  detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dream_actions_run ON dream_actions(run_id);

CREATE TABLE IF NOT EXISTS recall_traces (
  trace_id TEXT PRIMARY KEY,
  at TEXT NOT NULL,
  query TEXT NOT NULL,
  answer TEXT,
  answer_confidence REAL,
  abstained INTEGER NOT NULL DEFAULT 0,
  token_budget INTEGER,
  tokens_used INTEGER,
  used TEXT,
  excluded TEXT,
  counterfactual TEXT,
  verifications TEXT
);

CREATE TABLE IF NOT EXISTS calibration_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  at TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('retention','confidence')),
  trace_id TEXT,
  memory_id TEXT,
  predicted REAL NOT NULL,
  outcome INTEGER NOT NULL,
  freshness_days REAL
);
"""


def iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def parse_dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def slot_key_str(slot: Optional[tuple[str, str]]) -> Optional[str]:
    return SEP.join(_norm(p) for p in slot) if slot else None


def triple_key_str(triple: Optional[tuple[str, str, str]]) -> Optional[str]:
    return SEP.join(_norm(p) for p in triple) if triple else None


class MemoryStore:
    def __init__(self, db_path: str, *, clock: Callable[[], datetime] = utcnow):
        self._clock = clock
        self._lock = threading.RLock()
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ audit

    def audit(
        self,
        actor: str,
        action: str,
        *,
        memory_id: Optional[str] = None,
        run_id: Optional[str] = None,
        detail: Optional[dict] = None,
        at: Optional[datetime] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_log (at, actor, action, memory_id, run_id, detail)"
                " VALUES (?,?,?,?,?,?)",
                (
                    iso(at or self._clock()),
                    actor,
                    action,
                    memory_id,
                    run_id,
                    json.dumps(detail, default=str) if detail is not None else None,
                ),
            )
            self._conn.commit()

    def audit_tail(
        self,
        *,
        memory_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        q = "SELECT * FROM audit_log"
        conds, args = [], []
        if memory_id:
            conds.append("memory_id = ?")
            args.append(memory_id)
        if action:
            conds.append("action = ?")
            args.append(action)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY seq DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [self._audit_row(r) for r in rows]

    @staticmethod
    def _audit_row(r: sqlite3.Row) -> dict:
        return {
            "seq": r["seq"],
            "at": r["at"],
            "actor": r["actor"],
            "action": r["action"],
            "memory_id": r["memory_id"],
            "run_id": r["run_id"],
            "detail": json.loads(r["detail"]) if r["detail"] else None,
        }

    # --------------------------------------------------------------- memories

    def add(self, mem: MemoryItem, *, actor: str, detail: Optional[dict] = None) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO memories (
                    id, content, content_verbatim, triple_s, triple_r, triple_o,
                    slot_key, triple_key, embedding, mtype, importance, salience,
                    difficulty, stability, last_review_at, review_count, pinned,
                    valid_from, valid_to, status, superseded_by, confidence,
                    last_verified_at, source_ref, created_at, last_accessed_at,
                    provenance
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                self._to_row(mem),
            )
            self._conn.commit()
        self.audit(actor, "create", memory_id=mem.id, detail=detail)

    def update(
        self,
        mem: MemoryItem,
        *,
        actor: str,
        action: str = "update",
        detail: Optional[dict] = None,
    ) -> None:
        row = self._to_row(mem)
        with self._lock:
            cur = self._conn.execute(
                """UPDATE memories SET
                    content=?, content_verbatim=?, triple_s=?, triple_r=?, triple_o=?,
                    slot_key=?, triple_key=?, embedding=?, mtype=?, importance=?,
                    salience=?, difficulty=?, stability=?, last_review_at=?,
                    review_count=?, pinned=?, valid_from=?, valid_to=?, status=?,
                    superseded_by=?, confidence=?, last_verified_at=?, source_ref=?,
                    created_at=?, last_accessed_at=?, provenance=?
                WHERE id=?""",
                row[1:] + (mem.id,),
            )
            if cur.rowcount == 0:
                raise KeyError(f"memory not found: {mem.id}")
            self._conn.commit()
        self.audit(actor, action, memory_id=mem.id, detail=detail)

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        return self._from_row(r) if r else None

    def list(
        self,
        *,
        status: Optional[str] = None,
        mtype: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 500,
    ) -> list[MemoryItem]:
        query = "SELECT * FROM memories"
        conds, args = [], []
        if status:
            conds.append("status = ?")
            args.append(status)
        if mtype:
            conds.append("mtype = ?")
            args.append(mtype)
        if q:
            conds.append("(content LIKE ? OR content_verbatim LIKE ?)")
            args.extend([f"%{q}%", f"%{q}%"])
        if conds:
            query += " WHERE " + " AND ".join(conds)
        query += " ORDER BY created_at DESC, id LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(query, args).fetchall()
        return [self._from_row(r) for r in rows]

    def active(self) -> list[MemoryItem]:
        return self.list(status="active", limit=100_000)

    def find_by_slot(
        self, slot: tuple[str, str], *, status: Optional[str] = "active"
    ) -> list[MemoryItem]:
        q = "SELECT * FROM memories WHERE slot_key = ?"
        args: list[Any] = [slot_key_str(slot)]
        if status:
            q += " AND status = ?"
            args.append(status)
        q += " ORDER BY valid_from DESC, created_at DESC, id"
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [self._from_row(r) for r in rows]

    def find_by_triple(
        self, triple: tuple[str, str, str], *, status: Optional[str] = "active"
    ) -> Optional[MemoryItem]:
        q = "SELECT * FROM memories WHERE triple_key = ?"
        args: list[Any] = [triple_key_str(triple)]
        if status:
            q += " AND status = ?"
            args.append(status)
        q += " ORDER BY created_at DESC, id LIMIT 1"
        with self._lock:
            r = self._conn.execute(q, args).fetchone()
        return self._from_row(r) if r else None

    def touch_access(self, ids: list[str], at: Optional[datetime] = None) -> None:
        if not ids:
            return
        ts = iso(at or self._clock())
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET last_accessed_at=? WHERE id=?",
                [(ts, i) for i in ids],
            )
            self._conn.commit()

    def counts_by_status(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM memories GROUP BY status"
            ).fetchall()
        counts = {"active": 0, "deprecated": 0, "superseded": 0}
        for r in rows:
            counts[r["status"]] = r["n"]
        return counts

    # ---------------------------------------------------------------- reviews

    def record_review(
        self,
        memory_id: str,
        *,
        kind: str,
        grade: int,
        elapsed_days: float,
        r_before: float,
        before: tuple[float, float],
        after: tuple[float, float],
        at: Optional[datetime] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO reviews (memory_id, at, kind, grade, elapsed_days,"
                " r_before, d_before, s_before, d_after, s_after)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    memory_id,
                    iso(at or self._clock()),
                    kind,
                    grade,
                    elapsed_days,
                    r_before,
                    before[0],
                    before[1],
                    after[0],
                    after[1],
                ),
            )
            self._conn.commit()

    def reviews_for(self, memory_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reviews WHERE memory_id=? ORDER BY at, id",
                (memory_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------- dream runs

    def start_dream_run(self, run_id: str, *, at: Optional[datetime] = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO dream_runs (run_id, started_at) VALUES (?,?)",
                (run_id, iso(at or self._clock())),
            )
            self._conn.commit()

    def log_dream_action(
        self,
        run_id: str,
        phase: str,
        detail: dict,
        *,
        at: Optional[datetime] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO dream_actions (run_id, at, phase, detail) VALUES (?,?,?,?)",
                (run_id, iso(at or self._clock()), phase, json.dumps(detail, default=str)),
            )
            self._conn.commit()

    def finish_dream_run(
        self,
        run_id: str,
        *,
        journal: str,
        stats: dict,
        at: Optional[datetime] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE dream_runs SET finished_at=?, journal=?, stats=? WHERE run_id=?",
                (iso(at or self._clock()), journal, json.dumps(stats, default=str), run_id),
            )
            self._conn.commit()

    def dream_runs(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM dream_runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._dream_run_row(r) for r in rows]

    def dream_run(self, run_id: str) -> Optional[dict]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM dream_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if r is None:
                return None
            actions = self._conn.execute(
                "SELECT * FROM dream_actions WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
        out = self._dream_run_row(r)
        out["actions"] = [
            {
                "at": a["at"],
                "phase": a["phase"],
                "detail": json.loads(a["detail"]),
            }
            for a in actions
        ]
        return out

    @staticmethod
    def _dream_run_row(r: sqlite3.Row) -> dict:
        return {
            "run_id": r["run_id"],
            "started_at": r["started_at"],
            "finished_at": r["finished_at"],
            "journal": r["journal"],
            "stats": json.loads(r["stats"]) if r["stats"] else None,
        }

    # ----------------------------------------------------------------- traces

    def save_trace(self, trace: dict) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO recall_traces (
                    trace_id, at, query, answer, answer_confidence, abstained,
                    token_budget, tokens_used, used, excluded, counterfactual,
                    verifications
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trace["trace_id"],
                    trace["at"],
                    trace["query"],
                    trace.get("answer"),
                    trace.get("answer_confidence"),
                    1 if trace.get("abstained") else 0,
                    trace.get("token_budget"),
                    trace.get("tokens_used"),
                    json.dumps(trace.get("used", []), default=str),
                    json.dumps(trace.get("excluded", []), default=str),
                    json.dumps(trace.get("counterfactual", []), default=str),
                    json.dumps(trace.get("verifications", []), default=str),
                ),
            )
            self._conn.commit()

    def get_trace(self, trace_id: str) -> Optional[dict]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM recall_traces WHERE trace_id=?", (trace_id,)
            ).fetchone()
        return self._trace_row(r) if r else None

    def latest_trace(self) -> Optional[dict]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM recall_traces ORDER BY at DESC, trace_id DESC LIMIT 1"
            ).fetchone()
        return self._trace_row(r) if r else None

    @staticmethod
    def _trace_row(r: sqlite3.Row) -> dict:
        return {
            "trace_id": r["trace_id"],
            "at": r["at"],
            "query": r["query"],
            "answer": r["answer"],
            "answer_confidence": r["answer_confidence"],
            "abstained": bool(r["abstained"]),
            "token_budget": r["token_budget"],
            "tokens_used": r["tokens_used"],
            "used": json.loads(r["used"]) if r["used"] else [],
            "excluded": json.loads(r["excluded"]) if r["excluded"] else [],
            "counterfactual": json.loads(r["counterfactual"]) if r["counterfactual"] else [],
            "verifications": json.loads(r["verifications"]) if r["verifications"] else [],
        }

    # ------------------------------------------------------------ calibration

    def record_calibration(
        self,
        kind: str,
        *,
        predicted: float,
        outcome: bool,
        freshness_days: Optional[float] = None,
        trace_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        at: Optional[datetime] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO calibration_events (at, kind, trace_id, memory_id,"
                " predicted, outcome, freshness_days) VALUES (?,?,?,?,?,?,?)",
                (
                    iso(at or self._clock()),
                    kind,
                    trace_id,
                    memory_id,
                    predicted,
                    1 if outcome else 0,
                    freshness_days,
                ),
            )
            self._conn.commit()

    def calibration_events(self, kind: Optional[str] = None) -> list[dict]:
        q = "SELECT * FROM calibration_events"
        args: list[Any] = []
        if kind:
            q += " WHERE kind = ?"
            args.append(kind)
        q += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------- row codecs

    @staticmethod
    def _to_row(mem: MemoryItem) -> tuple:
        triple = mem.triple
        return (
            mem.id,
            mem.content,
            mem.content_verbatim,
            triple[0] if triple else None,
            triple[1] if triple else None,
            triple[2] if triple else None,
            slot_key_str(mem.slot_key),
            triple_key_str(mem.triple_key),
            json.dumps(mem.embedding),
            mem.mtype,
            mem.importance,
            mem.salience,
            mem.difficulty,
            mem.stability,
            iso(mem.last_review_at),
            mem.review_count,
            1 if mem.pinned else 0,
            iso(mem.valid_from),
            iso(mem.valid_to),
            mem.status,
            mem.superseded_by,
            mem.confidence,
            iso(mem.last_verified_at),
            mem.source_ref,
            iso(mem.created_at),
            iso(mem.last_accessed_at),
            json.dumps(mem.provenance),
        )

    @staticmethod
    def _from_row(r: sqlite3.Row) -> MemoryItem:
        triple = None
        if r["triple_s"] is not None:
            triple = (r["triple_s"], r["triple_r"], r["triple_o"])
        return MemoryItem(
            id=r["id"],
            content=r["content"],
            content_verbatim=r["content_verbatim"],
            triple=triple,
            embedding=json.loads(r["embedding"]),
            mtype=r["mtype"],
            importance=r["importance"],
            salience=r["salience"],
            difficulty=r["difficulty"],
            stability=r["stability"],
            last_review_at=parse_dt(r["last_review_at"]),
            review_count=r["review_count"],
            pinned=bool(r["pinned"]),
            valid_from=parse_dt(r["valid_from"]),
            valid_to=parse_dt(r["valid_to"]),
            status=r["status"],
            superseded_by=r["superseded_by"],
            confidence=r["confidence"],
            last_verified_at=parse_dt(r["last_verified_at"]),
            source_ref=r["source_ref"],
            created_at=parse_dt(r["created_at"]),
            last_accessed_at=parse_dt(r["last_accessed_at"]),
            provenance=json.loads(r["provenance"]),
        )
