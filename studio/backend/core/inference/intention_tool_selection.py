# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Intention-aware selection of MCP tools for a turn.

Every chat turn injects *all* enabled MCP tool schemas into the request via
``get_enabled_mcp_tools``. As the connected tool library grows that
closed-world injection gets expensive (more schema tokens, noisier tool
choice) — the exact cost "SING: Synthetic Intention Graph for Scalable
Active Tool Discovery in LLM Agents" targets.

This module delivers SING's core result as a small, dependency-free slice:
an **intention-tool graph** that links *user intentions* -> *tool
capabilities* (with a light capability-collaboration signal) and dynamically
narrows the injected MCP specs to the ones relevant to the current turn's
intention. It consumes and returns the same ``list[dict]`` OpenAI-function
spec shape the rest of the inference path expects, so it slots in as a
filter rather than parallel infrastructure.

Adapted from: SING (arXiv:2606.16591). The full method trains embeddings over
a 7,471-tool corpus and a learned graph; this is the lexical,
no-extra-dependency rendering of the same idea — sufficient to remove the
closed-world injection cost at scale. The selection is a deliberate no-op
when the library is small (injection isn't costly yet) or when no intention
can be read from the turn (filtering blind would be worse than exposing all).
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Optional

from loggers import get_logger

logger = get_logger(__name__)

# Per-turn cap on how many MCP tools survive intention selection. Only applied
# once the discovered library exceeds it, so small libraries are untouched.
# Overridable via env so operators can tune the exposure/coverage trade-off.
_DEFAULT_MAX_TOOLS = 12
_MAX_TOOLS_ENV = "UNSLOTH_MCP_TOOL_BUDGET"


# ── intention-tool graph ────────────────────────────────────────────
#
# SING's graph has three node kinds: user intentions, tool capabilities, and
# tool-collaboration patterns. We model the first two explicitly and the third
# via ``_COLLABORATION_HINTS`` below. Each intention carries:
#   * ``triggers``  — terms the user says that *activate* the intention, and
#   * ``capabilities`` — capability terms looked for in a tool's spec.
# The shared capability vocabulary is the intention -> tool edge: a tool whose
# spec mentions an active intention's capabilities is relevant to the turn.

_INTENTION_GRAPH: dict[str, dict[str, tuple[str, ...]]] = {
    "web_search": {
        "triggers": ("search", "google", "look up", "look-up", "find online", "web search"),
        "capabilities": ("search", "web", "online", "query", "bing", "duckduckgo", "fetch"),
    },
    "code_execution": {
        "triggers": ("run code", "python", "execute", "script", "compute", "calculate", "eval"),
        "capabilities": ("python", "code", "execute", "script", "repl", "compute", "calc", "eval"),
    },
    "file_read": {
        "triggers": (
            "read file",
            "open file",
            "list files",
            "directory",
            "file system",
            "filesystem",
        ),
        "capabilities": ("file", "read", "write", "directory", "filesystem", "path", "folder"),
    },
    "database_query": {
        "triggers": (
            "sql",
            "database",
            "select from",
            "query the db",
            "table",
            "postgres",
            "sqlite",
        ),
        "capabilities": (
            "sql",
            "database",
            "query",
            "table",
            "select",
            "insert",
            "db",
            "postgres",
            "mysql",
            "sqlite",
        ),
    },
    "http_request": {
        "triggers": ("http", "api request", "fetch url", "rest endpoint", "hit the endpoint"),
        "capabilities": ("http", "request", "api", "url", "fetch", "endpoint", "rest"),
    },
    "email": {
        "triggers": ("email", "e-mail", "mail", "inbox", "send a message to"),
        "capabilities": ("email", "mail", "inbox", "smtp", "send", "message"),
    },
    "calendar": {
        "triggers": ("calendar", "schedule", "meeting", "appointment", "event"),
        "capabilities": ("calendar", "schedule", "event", "meeting", "appointment"),
    },
    "image_generation": {
        "triggers": ("generate image", "draw", "picture of", "image of", "render image", "paint"),
        "capabilities": (
            "image",
            "generate",
            "draw",
            "picture",
            "render",
            "paint",
            "dall",
            "stable diffusion",
        ),
    },
    "math": {
        "triggers": ("solve equation", "integral", "derivative", "math", "arithmetic"),
        "capabilities": (
            "math",
            "solve",
            "equation",
            "integral",
            "derivative",
            "arithmetic",
            "compute",
        ),
    },
    "knowledge_retrieval": {
        "triggers": ("knowledge base", "my documents", "my files", "search notes", "rag"),
        "capabilities": (
            "document",
            "knowledge",
            "retrieval",
            "index",
            "rag",
            "notes",
            "embedding",
        ),
    },
    "git": {
        "triggers": ("git", "commit", "branch", "pull request", "repository", "merge"),
        "capabilities": ("git", "commit", "branch", "repository", "pull", "push", "merge", "repo"),
    },
    "shell_command": {
        "triggers": ("terminal", "shell", "bash", "command line", "run command"),
        "capabilities": ("shell", "terminal", "bash", "command", "exec", "run"),
    },
}

# Light rendering of SING's tool-collaboration node: when an intention is
# active, capabilities it commonly composes with get a small score boost, so
# complementary tools (e.g. a list-files tool beside a read-file task) survive.
_COLLABORATION_HINTS: dict[str, tuple[str, ...]] = {
    "code_execution": ("file_read",),
    "file_read": ("shell_command",),
    "database_query": ("http_request",),
}


_WORD_RE = re.compile(r"[a-z0-9]+")


def _resolve_max_tools(max_tools: Optional[int]) -> int:
    """Resolve the per-turn budget: explicit arg > env > default."""
    if max_tools is not None:
        return max(0, max_tools)
    raw = os.getenv(_MAX_TOOLS_ENV)
    if raw:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid %s=%r; using default %d.", _MAX_TOOLS_ENV, raw, _DEFAULT_MAX_TOOLS
            )
    return _DEFAULT_MAX_TOOLS


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens; multi-word terms split into their words."""
    return set(_WORD_RE.findall((text or "").lower()))


def _phrase_tokens(text: str) -> set[str]:
    """Tokens for a trigger/capability phrase, including the joined form."""
    words = _WORD_RE.findall((text or "").lower())
    out = set(words)
    if len(words) > 1:
        out.add("".join(words))  # "look up" -> also "lookup"
    return out


def _active_intentions(task_tokens: set[str]) -> set[str]:
    """Intentions whose trigger terms appear in the turn's task text."""
    active: set[str] = set()
    for intention, edges in _INTENTION_GRAPH.items():
        for trigger in edges["triggers"]:
            if _phrase_tokens(trigger) & task_tokens:
                active.add(intention)
                break
    return active


def _active_capability_terms(active: set[str]) -> set[str]:
    """Union of capability terms (as token sets) for the active intentions."""
    terms: set[str] = set()
    for intention in active:
        for cap in _INTENTION_GRAPH[intention]["capabilities"]:
            terms |= _phrase_tokens(cap)
    # Collaboration: pull in composed capabilities too.
    for intention in active:
        for related in _COLLABORATION_HINTS.get(intention, ()):
            if related in _INTENTION_GRAPH:
                for cap in _INTENTION_GRAPH[related]["capabilities"]:
                    terms |= _phrase_tokens(cap)
    return terms


def _spec_text(spec: dict[str, Any]) -> str:
    """Pull the textual surface a tool advertises (name + description)."""
    fn = spec.get("function") if isinstance(spec, dict) else None
    if not isinstance(fn, dict):
        return ""
    return " ".join(str(fn.get(k) or "") for k in ("name", "description"))


def _score_spec(spec_tokens: set[str], active_caps: set[str]) -> int:
    """How many active capability terms a tool's spec advertises."""
    return len(spec_tokens & active_caps)


def latest_user_text(messages: Iterable[Any]) -> str:
    """Text of the most recent user message, for intention reading.

    Tolerates both Pydantic ``ChatMessage`` objects and plain dicts, and
    content that is a string or a list of multimodal parts.
    """
    messages = list(messages or [])
    for msg in reversed(messages):
        role = getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")
        if role != "user":
            continue
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        return _content_to_text(content)
    return ""


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return " ".join(parts)


def select_mcp_tools_for_intention(
    task_text: str,
    mcp_specs: list[dict],
    *,
    max_tools: Optional[int] = None,
) -> list[dict]:
    """Narrow MCP tool specs to those relevant to the turn's intention.

    Returns the same ``list[dict]`` OpenAI-function spec shape, filtered to the
    top ``max_tools`` by intention-graph score. Selection is a deliberate
    no-op (returns ``mcp_specs`` unchanged) when:

      * the discovered library already fits the budget — closed-world injection
        isn't costly yet, so there's nothing to gain by dropping tools;
      * no intention can be read from ``task_text`` — we won't filter blind;
      * nothing scores above zero — no tool is clearly relevant, so exposing
        all is safer than an arbitrary cut.

    Slicing only happens at scale *and* with a readable intention, which is
    SING's regime and the only place active discovery pays off.
    """
    budget = _resolve_max_tools(max_tools)
    if budget <= 0 or len(mcp_specs) <= budget:
        return list(mcp_specs)

    task_tokens = _tokenize(task_text)
    active = _active_intentions(task_tokens)
    if not active:
        return list(mcp_specs)

    active_caps = _active_capability_terms(active)
    scored = [
        (idx, spec, _score_spec(_tokenize(_spec_text(spec)), active_caps))
        for idx, spec in enumerate(mcp_specs)
    ]
    if not any(score for _, _, score in scored):
        return list(mcp_specs)

    # Keep the top-scoring specs; ties keep original discovery order so the
    # selection is stable and deterministic across turns.
    kept = sorted(scored, key=lambda t: (-t[2], t[0]))[:budget]
    selected = [spec for _, spec, _ in kept]

    dropped = len(mcp_specs) - len(selected)
    logger.info(
        "intention_tool_selection: active intentions=%s; narrowed %d MCP tools "
        "to %d (dropped %d) for this turn.",
        sorted(active),
        len(mcp_specs),
        len(selected),
        dropped,
    )
    return selected
