# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for graph-structured session memory and its wiring into `unsloth chat`.

The integration assertions drive the real `chat` command (a non-new module)
with a fake backend that records the messages it is handed, proving the
`--memory-budget` flag and `/memory` command actually call into
`session_memory`.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.session_memory import (
    SessionGraph,
    compact_messages,
    estimate_tokens,
    extract_session_graph,
    render_resume_block,
)

# The integration tests drive the real `chat` command, which pulls in typer +
# rich + the studio backend path. Skip them gracefully where those aren't
# installed (the pure session_memory tests still run everywhere).
try:
    import typer
    from typer.testing import CliRunner

    import unsloth_cli.commands.chat as chatmod

    _HAS_CLI = True
except ModuleNotFoundError:
    _HAS_CLI = False

requires_cli = pytest.mark.skipif(not _HAS_CLI, reason = "typer/rich not installed")


# --- unit: extraction + rendering -------------------------------------------

def test_extract_captures_decision_with_rationale():
    messages = [
        {"role": "user", "content": "Let's use SQLite because it needs no server."},
    ]
    graph = extract_session_graph(messages)
    assert graph.decisions, "expected a decision node"
    body, rationale = graph.decisions[0]
    assert "sqlite" in body.lower()
    # The headline TokenMizer property: the *rationale* survives, not just the
    # mention.
    assert rationale is not None and "no server" in rationale.lower()


def test_extract_collects_files_tasks_entities():
    messages = [
        {"role": "user", "content": "Implement the parser in chat_history.py."},
        {"role": "assistant", "content": "I'll wire it through studio_db.py using FastAPI."},
    ]
    graph = extract_session_graph(messages)
    assert "chat_history.py" in graph.files
    assert "studio_db.py" in graph.files
    assert any("parser" in t.lower() for t in graph.tasks)
    assert "FastAPI" in graph.entities


def test_extract_is_fuzzy_deduped_and_skips_system():
    messages = [
        {"role": "system", "content": "Implement caching in cache.py."},  # skipped
        {"role": "user", "content": "Add  caching   to cache.py"},
        {"role": "user", "content": "add caching to cache.py."},  # dup of above
    ]
    graph = extract_session_graph(messages)
    assert graph.files == ["cache.py"]
    assert len(graph.tasks) == 1  # fuzzy dedup collapses the near-duplicate


def test_render_resume_block_is_compact_and_keeps_rationale():
    graph = SessionGraph(
        decisions = [("use SQLite for persistence", "simpler than Postgres")],
        tasks = ["add an index"],
        files = ["studio_db.py"],
        entities = ["SQLite"],
    )
    block = render_resume_block(graph)
    assert "[session memory]" in block
    assert "simpler than Postgres" in block
    assert "studio_db.py" in block


def test_render_empty_graph_is_empty_string():
    assert render_resume_block(SessionGraph()) == ""


def test_resume_block_beats_raw_text_on_tokens():
    # The paper's core claim: a structured resume block is materially cheaper
    # than the raw transcript it summarises.
    transcript = (
        "Let's use SQLite because a server is overkill here. "
        "Implement the migration in studio_db.py. "
        "We'll switch to FastAPI for the routes since it is async. "
        "Add an index on the chat_history.py table to speed lookups."
    ) * 4
    messages = [{"role": "user", "content": transcript}]
    block = render_resume_block(extract_session_graph(messages))
    assert estimate_tokens(block) < estimate_tokens(transcript)


# --- unit: compaction policy ------------------------------------------------

def _msgs(n):
    return [{"role": "user", "content": f"turn {i} about studio_db.py"} for i in range(n)]


def test_compact_disabled_when_budget_zero():
    msgs = _msgs(20)
    assert compact_messages(msgs, max_tokens = 0) == msgs


def test_compact_noop_under_budget():
    msgs = _msgs(2)
    assert compact_messages(msgs, max_tokens = 10_000) == msgs


def test_compact_folds_old_turns_into_system_block():
    # A long, chatty history whose structured facts compress well — the case
    # where folding actually pays off.
    filler = " ".join(["here is a lot of incidental discussion and back and forth"] * 6)
    msgs = [
        {"role": "user", "content": f"Let's use SQLite because it is simple. {filler}"},
        {"role": "assistant", "content": f"Implement the loader in studio_db.py. {filler}"},
        {"role": "user", "content": f"Add an index to chat_history.py. {filler}"},
        {"role": "assistant", "content": f"Done with that. {filler}"},
        {"role": "user", "content": "What did we pick for storage?"},
        {"role": "assistant", "content": "SQLite."},
    ]
    out = compact_messages(msgs, max_tokens = 1, keep_recent = 2)
    assert out[0]["role"] == "system"
    assert "[session memory]" in out[0]["content"]
    # Recent turns are preserved verbatim at the tail.
    assert out[-2:] == msgs[-2:]
    # And the fold genuinely shrank the payload.
    assert len(out) < len(msgs)


def test_compact_keeps_history_when_block_would_not_help():
    # Short turns with little structure: the resume block is no smaller than
    # what it replaces, so compaction is declined rather than losing turns.
    msgs = [
        {"role": "user", "content": "Let's use SQLite because it is simple."},
        {"role": "assistant", "content": "Implement it in studio_db.py."},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "np"},
    ]
    out = compact_messages(msgs, max_tokens = 1, keep_recent = 2)
    assert out == msgs


# --- integration: the chat command actually calls session_memory ------------

class _FakeConfig:
    is_gguf = False
    is_lora = True
    display_name = "fake-model"
    base_model = "fake/base"
    path = None


def _chat_app():
    cli = typer.Typer()
    cli.command()(chatmod.chat)
    return cli


@requires_cli
def test_chat_exposes_memory_budget_option():
    opt = inspect.signature(chatmod.chat).parameters["memory_budget"].default
    assert "--memory-budget" in (getattr(opt, "param_decls", None) or [])
    assert getattr(opt, "default", None) == 0  # off by default, no behavior change


@requires_cli
def test_memory_command_renders_resume_block(monkeypatch):
    class _FakeChatBackend:
        def stream(self, *a, **k):
            return iter(["ok"])

        def close(self):
            pass

    monkeypatch.setattr(chatmod, "resolve_model_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(chatmod, "load_chat_backend", lambda *a, **k: _FakeChatBackend())
    monkeypatch.setattr(chatmod, "_compare_needs_second_model", lambda: False)
    monkeypatch.setattr(chatmod, "connect_studio_server", lambda *a, **k: None)

    result = CliRunner().invoke(
        _chat_app(),
        ["fake-model"],
        input = "Let's use SQLite because it is simple\n/memory\n/exit\n",
    )
    assert result.exit_code == 0, result.output
    assert "[session memory]" in result.output
    assert "sqlite" in result.output.lower()


@requires_cli
def test_memory_budget_compacts_history_sent_to_backend(monkeypatch):
    """End-to-end: with a tiny budget, the backend receives a compacted list."""
    seen_messages = []

    class _RecordingBackend:
        def stream(self, messages, *a, **k):
            # Snapshot what the chat loop handed us this turn.
            seen_messages.append([dict(m) for m in messages])
            return iter(["ack"])

        def close(self):
            pass

    monkeypatch.setattr(chatmod, "resolve_model_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(chatmod, "load_chat_backend", lambda *a, **k: _RecordingBackend())
    monkeypatch.setattr(chatmod, "_compare_needs_second_model", lambda: False)
    monkeypatch.setattr(chatmod, "connect_studio_server", lambda *a, **k: None)

    # Several substantive turns, then one more that triggers a compacted send.
    convo = (
        "Let's use SQLite because a server is overkill\n"
        "Implement the loader in studio_db.py\n"
        "We'll switch to FastAPI since it is async\n"
        "Add an index to chat_history.py\n"
        "remind me what storage we chose\n"
    )
    result = CliRunner().invoke(
        _chat_app(), ["fake-model", "--memory-budget", "5"], input = convo + "/exit\n"
    )
    assert result.exit_code == 0, result.output

    last_sent = seen_messages[-1]
    # Compaction injected a system resume block at the head...
    assert last_sent[0]["role"] == "system"
    assert "[session memory]" in last_sent[0]["content"]
    # ...and the rationale-bearing decision survived the fold.
    assert "overkill" in last_sent[0]["content"].lower()
