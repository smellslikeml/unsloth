# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Graph-structured session memory for long-horizon `unsloth chat` sessions.

A multi-turn chat eventually outgrows the model's effective context window.
The usual fixes treat history as flat text and silently drop the oldest
turns, destroying the relational structure — which file mattered, what was
decided and *why* — that makes a session resumable.

This module keeps that structure. It extracts a small typed graph from the
older turns (decisions with their rationale, files, tasks, named entities)
and serialises it into a compact "resume block" that stands in for the raw
history. The extraction is pure heuristics with zero external dependencies,
so it adds no load to the chat path.

Adapted from "TokenMizer: Graph-Structured Session Memory for Long-Horizon
LLM Context Management" (arXiv:2606.06337). We keep a focused subset of the
paper's schema — the four node types that carry the most resume signal — and
its headline property: the resume block preserves the *rationale* behind a
decision, not just that something was mentioned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# Subset of TokenMizer's 14 node types — the ones that carry resume signal
# in a coding/assistant chat. Decisions additionally carry rationale.
NODE_TYPES = ("decision", "task", "file", "entity")

# A file path or a `module.py` style reference. Kept deliberately tight so
# ordinary prose ("e.g.") does not register as a file.
_FILE_RE = re.compile(
    r"\b[\w./-]+\.(?:py|md|toml|cfg|txt|json|ya?ml|sh|ps1|bat|ts|js|html|jinja)\b"
)

# "we decided to X", "let's use X", "I'll switch to X", "chose X" ... the verb
# group is what flags the clause as a decision; the tail is the decision text.
_DECISION_RE = re.compile(
    r"\b(?:decided to|let'?s|we'?ll|i'?ll|going to|chose to|chose|switch(?:ing)? to|"
    r"use|prefer|stick with|opt for)\b\s+(?P<body>.+)",
    re.IGNORECASE,
)

# Rationale connectors. The paper's ablation found rationale recall is the
# main thing flat-text baselines lose, so we split it out explicitly.
_RATIONALE_RE = re.compile(r"\b(because|since|so that|so as to|in order to|to avoid)\b", re.IGNORECASE)

# Imperative task phrasing — the high-recall "explicit imperative" register
# the paper reports scores best on.
_TASK_RE = re.compile(
    r"\b(?:implement|add|fix|refactor|remove|support|wire|build|write|create|"
    r"update|migrate|test|handle|investigate)\b\s+(?P<body>.+)",
    re.IGNORECASE,
)

# CamelCase / TitleCase tech names (FastAPI, SQLite, TokenMizer) and `code`
# spans. Common sentence-initial words are filtered out below.
_ENTITY_RE = re.compile(r"`([^`]+)`|\b([A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*)\b")

_STOPWORDS = frozenset(
    {"the", "a", "an", "it", "this", "that", "them", "those", "these", "i", "we", "you"}
)


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate.

    A word-plus-punctuation count scaled by 1.3 tracks real BPE counts closely
    enough to drive a compaction threshold without importing a tokenizer.
    """
    if not text:
        return 0
    pieces = re.findall(r"\w+|[^\w\s]", text)
    return max(1, round(len(pieces) * 1.3))


def _clean(body: str) -> str:
    # Trim to the first clause and drop trailing punctuation so dedup is stable.
    body = re.split(r"[.;!?\n]", body, maxsplit = 1)[0]
    return body.strip().rstrip(",:").strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _dedup(items: Iterable[str]) -> list[str]:
    """Order-preserving, fuzzy (case/space-insensitive) dedup.

    Fuzzy label matching is the single largest recall driver in the paper's
    ablation (+33pp), so collapsing near-duplicate mentions matters.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = _norm(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


@dataclass
class SessionGraph:
    """A typed, queryable snapshot of what a session established."""

    decisions: list[tuple[str, str | None]] = field(default_factory = list)
    tasks: list[str] = field(default_factory = list)
    files: list[str] = field(default_factory = list)
    entities: list[str] = field(default_factory = list)

    def is_empty(self) -> bool:
        return not (self.decisions or self.tasks or self.files or self.entities)


def _extract_entities(text: str) -> list[str]:
    found = []
    for backtick, camel in _ENTITY_RE.findall(text):
        token = (backtick or camel).strip()
        if token and token.lower() not in _STOPWORDS:
            found.append(token)
    return found


def extract_session_graph(messages: list[dict]) -> SessionGraph:
    """Populate a :class:`SessionGraph` from chat `messages`.

    `messages` is the OpenAI-style ``[{"role", "content"}, ...]`` list the chat
    loop already maintains. System messages (e.g. a previously injected resume
    block) are skipped so re-compaction stays idempotent.
    """
    decisions: list[tuple[str, str | None]] = []
    tasks: list[str] = []
    files: list[str] = []
    entities: list[str] = []

    for msg in messages:
        if msg.get("role") == "system":
            continue
        content = str(msg.get("content") or "")
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", content):
            sentence = sentence.strip()
            if not sentence:
                continue

            dmatch = _DECISION_RE.search(sentence)
            if dmatch:
                body = _clean(dmatch.group("body"))
                rmatch = _RATIONALE_RE.search(sentence)
                rationale = _clean(sentence[rmatch.start():]) if rmatch else None
                if rationale:
                    # Keep the decision text from leaking the rationale clause.
                    body = _clean(re.split(_RATIONALE_RE, body)[0]) or body
                if body:
                    decisions.append((body, rationale))

            tmatch = _TASK_RE.search(sentence)
            if tmatch:
                body = _clean(tmatch.group("body"))
                if body:
                    tasks.append(body)

            files.extend(_FILE_RE.findall(sentence))
            entities.extend(_extract_entities(sentence))

    # Files and entities are also extracted from decision/task bodies above, so
    # dedup across the whole graph at the end.
    deduped_decisions: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for body, rationale in decisions:
        key = _norm(body)
        if key and key not in seen:
            seen.add(key)
            deduped_decisions.append((body, rationale))

    return SessionGraph(
        decisions = deduped_decisions,
        tasks = _dedup(tasks),
        files = _dedup(files),
        entities = _dedup(entities),
    )


def render_resume_block(graph: SessionGraph, *, max_items: int = 6) -> str:
    """Serialise `graph` into a compact, model-readable resume block.

    Each category is capped at `max_items` (newest-wins is the caller's job;
    here we keep insertion order). Returns ``""`` for an empty graph so the
    caller can decide not to inject anything.
    """
    if graph.is_empty():
        return ""

    lines = ["[session memory]"]
    if graph.decisions:
        lines.append("decisions:")
        for body, rationale in graph.decisions[:max_items]:
            if rationale:
                lines.append(f"- {body} ({rationale})")
            else:
                lines.append(f"- {body}")
    if graph.tasks:
        lines.append("tasks: " + "; ".join(graph.tasks[:max_items]))
    if graph.files:
        lines.append("files: " + ", ".join(graph.files[:max_items]))
    if graph.entities:
        lines.append("entities: " + ", ".join(graph.entities[:max_items]))
    return "\n".join(lines)


def compact_messages(
    messages: list[dict],
    *,
    max_tokens: int,
    keep_recent: int = 4,
) -> list[dict]:
    """Return a token-trimmed copy of `messages` for sending to the model.

    When the estimated token footprint exceeds `max_tokens`, the older turns
    (everything but the last `keep_recent` messages) are collapsed into a
    single ``system`` resume block; the recent turns are kept verbatim. Below
    budget, or when there is nothing worth compacting, the original list is
    returned unchanged.

    `max_tokens <= 0` disables compaction entirely.
    """
    if max_tokens <= 0 or len(messages) <= keep_recent:
        return list(messages)

    full_text = "\n".join(str(m.get("content") or "") for m in messages)
    if estimate_tokens(full_text) <= max_tokens:
        return list(messages)

    head = messages[:-keep_recent]
    tail = messages[-keep_recent:]
    block = render_resume_block(extract_session_graph(head))
    if not block:
        # Nothing structured to keep — leave history untouched rather than
        # dropping turns blindly.
        return list(messages)

    note = {"role": "system", "content": block}
    compacted = [note, *tail]
    # Guard against the degenerate case where the block is larger than what it
    # replaced (very short histories): only compact if it actually helps.
    if estimate_tokens(block) >= estimate_tokens(
        "\n".join(str(m.get("content") or "") for m in head)
    ):
        return list(messages)
    return compacted
