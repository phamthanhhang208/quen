"""Verifiers — live-source checks backing the trust gate (spec §4.7).

A verifier answers one question: *is this memory's claim still true in the
live world?* The engine runs one before answering from a low-trust memory
(verify-before-answer). Outcomes are the closed set
``confirmed | refuted | unverifiable``; feeding an outcome back into
retention (the FSRS review) is `quen.trust.apply_verification`'s job —
verifiers themselves are side-effect free.

`RepoGrepVerifier` is the demo-corpus verifier: it greps a checked-out
repository for identifiers mentioned by the memory. Pure Python, offline,
deterministic.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol

from quen.models import MemoryItem

# Tokens as they appear in prose/code, dots allowed (module.attr paths).
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
# A greppable identifier: 3+ chars, identifier-shaped.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL_RE = re.compile(r"[a-z][A-Z]")

_MAX_IDENTIFIERS = 5

# English function words + generic verbs that are identifier-shaped but
# carry no greppable signal.
_STOPWORDS = frozenset(
    """
    the and for was were are has have had this that these those with from
    into over under about after before between during without within not
    all any some each because such only just very still there here what
    which who whom how why where when while then than else will would can
    could should may might must does did done being been be is it its
    they their them she her him his you your our who use uses used using
    also more most less other another same new old now yes etc per via
    said says say like get got make made take took
    """.split()
)


@dataclass
class VerificationEvent:
    """One verification outcome, as recorded on a recall trace."""

    memory_id: str
    verifier: str
    outcome: str              # "confirmed" | "refuted" | "unverifiable"
    evidence: Optional[str]
    at: datetime


class Verifier(Protocol):
    """A live-source checker the trust gate can call before answering."""

    name: str

    def can_verify(self, mem: MemoryItem) -> bool: ...

    def verify(self, mem: MemoryItem) -> tuple[str, Optional[str]]:
        """Check the memory's claim; returns (outcome, evidence)."""
        ...


def _is_preferred(token: str) -> bool:
    """Code-shaped tokens: camelCase, snake_case/UPPER_SNAKE, or dotted."""
    return "." in token or "_" in token or bool(_CAMEL_RE.search(token))


def _greppable_tokens(text: str) -> list[str]:
    """Extract greppable identifiers from text, preferred class first.

    Preferred tokens (camelCase / snake_case / UPPER_SNAKE / dotted) carry
    real signal; when any exist, plain English-word tokens are dropped so a
    stray common word cannot fake a confirmation.
    """
    preferred: list[str] = []
    plain: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(text):
        token = raw.strip(".")
        if not token or token in seen:
            continue
        seen.add(token)
        if token.lower() in _STOPWORDS:
            continue
        if "." in token:
            parts = token.split(".")
            if all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in parts):
                preferred.append(token)
            continue
        if not _IDENT_RE.fullmatch(token):
            continue
        if _is_preferred(token):
            preferred.append(token)
        else:
            plain.append(token)
    chosen = preferred if preferred else plain
    return chosen[:_MAX_IDENTIFIERS]


class RepoGrepVerifier:
    """Greps a live repository checkout for identifiers a memory names.

    The identifier to search for comes from the memory's triple object
    first (that's the slot value the claim is about); if the triple yields
    nothing greppable, identifier-shaped tokens from the content are used.

    Scanning is pure Python (`os.walk`), case-sensitive substring per line,
    deterministic order (sorted directories and filenames). `.git`,
    `node_modules`, `dist` and similar generated trees are skipped, as are
    binary files and files larger than ``max_file_kb``.
    """

    name = "repo_grep"

    _SKIP_DIRS = frozenset({".git", "node_modules", "dist", "__pycache__", ".venv"})

    # A grep can only verify positive existence claims: "we no longer use X"
    # would be CONFIRMED by finding X — outcome inverted. Such memories are
    # unverifiable by this verifier.
    _NEGATIVE_RE = re.compile(
        r"\b(no longer|not|never|removed|deleted|dropped|stopped|deprecated|gone)\b",
        re.IGNORECASE,
    )

    def __init__(self, repo_path: str, *, max_file_kb: int = 256):
        self.repo_path = repo_path
        self.max_file_kb = max_file_kb

    # ------------------------------------------------------------- protocol

    def can_verify(self, mem: MemoryItem) -> bool:
        return (
            os.path.isdir(self.repo_path)
            and not self._NEGATIVE_RE.search(mem.content)
            and bool(self._identifiers(mem))
        )

    def verify(self, mem: MemoryItem) -> tuple[str, Optional[str]]:
        if self._NEGATIVE_RE.search(mem.content):
            return ("unverifiable", "negative claims are not grep-verifiable")
        idents = self._identifiers(mem)
        if not idents:
            return ("unverifiable", "no greppable identifier in memory")
        if not os.path.isdir(self.repo_path):
            return ("unverifiable", f"repo path not found: {self.repo_path}")
        # word-boundary matching — 'useApi' must not confirm against a repo
        # that only contains 'useApiV2' (rename refactors are the core
        # staleness case)
        patterns = [
            re.compile(
                rf"(?<![A-Za-z0-9_]){re.escape(ident)}(?![A-Za-z0-9_])"
            )
            for ident in idents
        ]
        files_scanned = 0
        for path in self._iter_files():
            lines = self._read_text_lines(path)
            if lines is None:
                continue
            files_scanned += 1
            for lineno, line in enumerate(lines, start=1):
                for pattern in patterns:
                    if pattern.search(line):
                        rel = os.path.relpath(path, self.repo_path)
                        return ("confirmed", f"{rel}:{lineno}: {line.strip()}")
        if files_scanned == 0:
            return ("unverifiable", "no searchable files in repo")
        return ("refuted", f"searched {files_scanned} files; '{idents[0]}' not found")

    # -------------------------------------------------------------- helpers

    def _identifiers(self, mem: MemoryItem) -> list[str]:
        """Greppable identifiers: triple object first, else content tokens."""
        if mem.triple is not None:
            from_object = _greppable_tokens(mem.triple[2])
            if from_object:
                return from_object
        return _greppable_tokens(mem.content)

    def _iter_files(self):
        """Yield scannable file paths in a stable, sorted walk order."""
        max_bytes = self.max_file_kb * 1024
        for dirpath, dirnames, filenames in os.walk(self.repo_path):
            dirnames[:] = sorted(d for d in dirnames if d not in self._SKIP_DIRS)
            for fname in sorted(filenames):
                path = os.path.join(dirpath, fname)
                try:
                    if os.path.getsize(path) > max_bytes:
                        continue
                except OSError:
                    continue
                yield path

    @staticmethod
    def _read_text_lines(path: str) -> Optional[list[str]]:
        """Read a file's lines; None if unreadable or binary (NUL byte)."""
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            return None
        if b"\x00" in data:
            return None
        return data.decode("utf-8", errors="replace").splitlines()
