# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compile user corrections into runtime tool-call deny-rules.

Adapted from *Getting Better at Working With You: Compiling User Corrections
into Runtime Enforcement for Coding Agents* (TRACE), which mines a user's
chat-history corrections, rewrites them as atomic rules, and compiles those
rules into checks that must pass before an agent runs future tool calls.

Unsloth Studio already has the enforcement half of that loop: ``_bash_exec``
blocks commands via ``_find_blocked_commands`` and the agentic loop can gate
each call through ``tool_approvals``. This module supplies the *acquisition*
half -- turning free-text user corrections ("never use curl", "use uv instead
of pip", "don't import pickle") into atomic deny-rules that feed that same
``(tool_name, arguments) -> allow/deny`` contract. The chat history is the
correction substrate: because history persists across sessions, a correction
compiled once keeps shaping later tool calls without the user restating it.

The miner is deliberately deterministic (regex over correction phrasings)
rather than LLM-driven, so enforcement is cheap, reproducible, and easy to
reason about at the security gate. Cross-session rule storage in ``studio.db``
is intentionally out of scope: compiling from the persisted conversation on
each turn already delivers the "tell me once" behavior the paper targets.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass


@dataclass(frozen = True)
class CorrectionRule:
    """One atomic, user-authored enforcement rule.

    ``kind`` is ``"deny_command"`` (match a terminal command at command
    position) or ``"deny_code_pattern"`` (match a regex inside Python code).
    ``matcher`` is the lowercased command basename for the former, or the
    regex source for the latter. ``reason`` is surfaced back to the model as
    the tool result when the rule fires, so it can adapt instead of repeating
    the corrected mistake.
    """

    kind: str
    matcher: str
    reason: str
    source: str


# Phrasings that introduce a forbidden terminal command:
#   "never use curl", "don't run rm", "avoid using wget", "please do not execute sudo"
_ACTION_RE = re.compile(
    r"\b(?:never|don['’]t|do\s+not|stop|avoid|quit|no\s+longer)\s+"
    r"(?:using|use|running|run|executing|execute|calling|call)\s+"
    r"([A-Za-z_][\w.\-]*)",
    re.IGNORECASE,
)
# "never import pickle", "don't import os", "avoid importing subprocess"
_IMPORT_RE = re.compile(
    r"\b(?:never|don['’]t|do\s+not|stop|avoid)\s+import(?:ing)?\s+([A-Za-z_][\w.]*)",
    re.IGNORECASE,
)
# "use uv instead of pip", "use ruff rather than black", "we use uv not pip"
_INSTEAD_RE = re.compile(
    r"\buse\s+([A-Za-z_][\w.\-]*)\s+"
    r"(?:instead\s+of|rather\s+than|not)\s+"
    r"([A-Za-z_][\w.\-]*)",
    re.IGNORECASE,
)

# Common words the phrasings can bind to that are never commands/packages, so
# "don't use that" / "never use it" do not mint junk rules.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "this", "that", "these", "those", "it", "its",
        "any", "some", "such", "your", "our", "my", "their", "them", "they",
        "you", "we", "i", "here", "there", "now", "then", "more", "less",
        "other", "another", "same", "own", "all", "both", "each", "every",
    }
)

# Shell structure mirrors ``_find_blocked_commands`` so corrections fire only
# at command position (``rm`` in ``grep -r rm .`` is an argument, not a command).
_SHELL_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "\n", "(", ")", "`", "{", "}"})
_SHELL_KEYWORDS_AS_SEP = frozenset({"then", "do", "else", "elif"})
_COMMAND_PREFIXES = frozenset(
    {
        "env", "command", "builtin", "exec", "time", "nohup", "nice",
        "setsid", "stdbuf", "timeout", "ionice", "chroot", "sudo", "doas",
        "su", "xargs",
    }
)
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _clean_term(term: str) -> str:
    """Lowercase and strip trailing punctuation so ``grep.`` / ``pip,`` -> ``grep`` / ``pip``.

    Internal dots are preserved (``os.path``), only a trailing run of ``.``/``-``
    from end-of-sentence punctuation is removed.
    """
    return term.lower().rstrip(".-")


def _is_term_like(term: str) -> bool:
    """Reject stopwords so the miner only binds to command/package-shaped tokens."""
    return bool(term) and term.lower() not in _STOPWORDS


def _excerpt(text: str, start: int, width: int = 64) -> str:
    """Short, single-line snippet around a match for the rule's provenance."""
    snippet = text[start:start + width].replace("\n", " ").strip()
    return snippet[:width]


def _code_import_pattern(module: str) -> str:
    """Regex matching ``import <module>`` / ``from <module> import ...`` in code."""
    escaped = re.escape(module)
    return rf"\b(?:import\s+{escaped}\b|from\s+{escaped}(?:\.|\s))"


def compile_correction(text: str) -> list[CorrectionRule]:
    """Mine a single user-authored correction string into atomic rules.

    Returns zero or more rules; duplicates (same ``kind`` + ``matcher``) are
    collapsed, keeping the first phrasing encountered as the reason.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    rules: list[CorrectionRule] = []
    seen: set[tuple[str, str]] = set()

    def _add(rule: CorrectionRule) -> None:
        key = (rule.kind, rule.matcher)
        if rule.matcher and _is_term_like(rule.matcher) and key not in seen:
            seen.add(key)
            rules.append(rule)

    for match in _ACTION_RE.finditer(text):
        term = _clean_term(match.group(1))
        _add(
            CorrectionRule(
                kind = "deny_command",
                matcher = term,
                reason = f"do not use {term}",
                source = _excerpt(text, match.start()),
            )
        )

    for match in _IMPORT_RE.finditer(text):
        module = _clean_term(match.group(1))
        _add(
            CorrectionRule(
                kind = "deny_code_pattern",
                matcher = _code_import_pattern(module),
                reason = f"do not import {module}",
                source = _excerpt(text, match.start()),
            )
        )

    # "use Y instead of X" forbids the displaced tool X (the correction's target).
    for match in _INSTEAD_RE.finditer(text):
        preferred = _clean_term(match.group(1))
        displaced = _clean_term(match.group(2))
        _add(
            CorrectionRule(
                kind = "deny_command",
                matcher = displaced,
                reason = f"use {preferred} instead of {displaced}",
                source = _excerpt(text, match.start()),
            )
        )

    return rules


def compile_corrections(messages: list[dict] | None) -> list[CorrectionRule]:
    """Compile the user's standing corrections from chat history.

    Only ``user`` turns are mined: the paper compiles *the user's own*
    corrections, not assistant prose or tool results, so a model musing "I'll
    use curl" can never mint a rule. History is the cross-session substrate --
    prior turns are loaded into ``conversation`` on each request, so a rule
    minted in one session keeps applying in the next without restating it.
    """
    rules: list[CorrectionRule] = []
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        rules.extend(compile_correction(content))
    return rules


def _command_position_basenames(command: str) -> set[str]:
    """Lowercased command-position basenames in a shell command.

    Mirrors the tokenizer in ``_find_blocked_commands``: a token counts only
    at the start of the string or after a separator / new-command keyword /
    command-prefix wrapper (``env``, ``time``, ``xargs`` ...). Argument
    tokens (``grep -r curl .``) are ignored, so a correction on ``curl`` does
    not fire on the word "curl" appearing inside an argument.
    """
    try:
        lexer = shlex.shlex(command, posix = True, punctuation_chars = ";&|()`")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = command.split()

    basenames: set[str] = set()
    expect_command = True
    prefix_pending = False
    for token in tokens:
        if token in _SHELL_SEPARATORS or token in _SHELL_KEYWORDS_AS_SEP:
            expect_command = True
            prefix_pending = False
            continue
        if token.startswith("-"):
            if not prefix_pending:
                expect_command = False
            continue
        if not expect_command:
            continue
        if _ASSIGNMENT_RE.match(token):  # FOO=bar prefix; real command follows
            continue
        if prefix_pending and token.lstrip("-").isdigit():  # timeout 5 cmd
            continue
        base = os.path.basename(token.strip(";&|()`{}")).lower()
        stem, ext = os.path.splitext(base)
        if ext in {".exe", ".com", ".bat", ".cmd"}:
            base = stem
        basenames.add(base)
        if base in _COMMAND_PREFIXES:
            prefix_pending = True
            continue
        expect_command = False
        prefix_pending = False
    return basenames


def find_violations(
    tool_name: str,
    arguments: dict | None,
    rules: list[CorrectionRule] | None,
) -> set[str]:
    """Return the reasons of any compiled rule the given tool call violates.

    Empty set means "no correction forbids this call" (allow). The set shape
    matches ``_find_blocked_commands`` so this composes cleanly with the
    existing blocklist gate.
    """
    if not rules:
        return set()

    args = arguments if isinstance(arguments, dict) else {}
    violations: set[str] = set()

    if tool_name == "terminal":
        command = args.get("command", "")
        if isinstance(command, str) and command.strip():
            basenames = _command_position_basenames(command)
            for rule in rules:
                if rule.kind == "deny_command" and rule.matcher in basenames:
                    violations.add(rule.reason)
    elif tool_name == "python":
        code = args.get("code", "")
        if isinstance(code, str) and code.strip():
            for rule in rules:
                if rule.kind == "deny_code_pattern":
                    try:
                        if re.search(rule.matcher, code):
                            violations.add(rule.reason)
                    except re.error:
                        continue
    return violations


def check_correction_rules(
    tool_name: str,
    arguments: dict | None,
    rules: list[CorrectionRule] | None,
) -> str | None:
    """Enforcement hook: return a deny-message the model sees, or ``None`` to allow.

    Sits beside ``_find_blocked_commands`` in the exec gate. The message is
    returned as the tool result so the agent can adapt and keep responding,
    exactly like the existing "Blocked command(s) for safety" path.
    """
    violations = find_violations(tool_name, arguments, rules)
    if not violations:
        return None
    return "Blocked by your earlier correction: " + ", ".join(sorted(violations)) + "."
