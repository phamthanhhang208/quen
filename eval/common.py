"""Shared eval plumbing: reader/embedder factories (identical across
configs, per spec §9), CLI args, output helpers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # the key lives in .env — must be loaded before any env check

OUT_DIR = Path(__file__).parent / "out"


def build_stack(live: bool):
    """(llm, embedder) — the same reader for every config.

    Offline (--dry-run, default): deterministic ScriptedLLM + HashingEmbedder;
    proves the pipeline end-to-end without a key. Live (--live): Qwen on
    Alibaba Cloud DashScope only.
    """
    if live:
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise SystemExit(
                "--live requires DASHSCOPE_API_KEY (Qwen via Alibaba Cloud "
                "DashScope; see .env.example)."
            )
        from quen.embeddings import QwenEmbedder
        from quen.llm import QwenLLM

        return QwenLLM(), QwenEmbedder()
    from quen.embeddings import HashingEmbedder
    from quen.llm import ScriptedLLM

    return ScriptedLLM.with_offline_defaults(), HashingEmbedder()


def make_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true",
                      help="use Qwen via DashScope (needs DASHSCOPE_API_KEY)")
    mode.add_argument("--dry-run", action="store_true", default=True,
                      help="deterministic offline stack (default)")
    p.add_argument("--limit", type=int, default=None,
                   help="run at most N cases")
    p.add_argument("--offset", type=int, default=0,
                   help="skip the first N cases (for sharded parallel runs)")
    p.add_argument("--budget", type=int, default=300,
                   help="token budget per answer (identical across configs)")
    p.add_argument("--out", type=Path, default=OUT_DIR,
                   help="output directory (default eval/out)")
    return p


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"wrote {path}")


def load_probe_cases(limit: int | None = None, offset: int = 0) -> list[dict]:
    data = json.loads((Path(__file__).parent / "probe_dataset.json").read_text())
    cases = data["cases"][offset:]
    return cases[:limit] if limit else cases
