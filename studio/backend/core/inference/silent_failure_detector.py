# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detect *silent* tool failures that the error-prefix check misses.

The local tool loop classifies a result as failed only when the string
*starts* with one of ``TOOL_ERROR_PREFIXES``. That catches the loud cases
(``Error: ...``, ``Exit code ...``) but not the failures whose error signal
is buried mid-string: an HTTP 500 returned in a JSON envelope, a Python
traceback embedded after a banner line, a ``{"ok": false}`` body, or a tool
that "succeeded" while returning nothing at all.

The taxonomy in *"When Errors Become Narratives: A Longitudinal Taxonomy of
Silent Failures in a Production LLM Agent Runtime"* (arXiv:2606.14589) calls
this class **(C) error swallowing and dilution** — and notes it is the
on-ramp to the most dangerous class, **(D) chained hallucination**: once a
swallowed error is fed back as a normal ``role=tool`` result (and worse,
cached as a *successful* call), the model narrates fluently over it and the
failure reaches the user as a convincing but wrong story — a *fail-plausible*
outcome.

This module stays at the seam the paper is about: result-string-in,
failure-bool-out. It does not try to read the model's downstream narrative
(detecting class D requires the generated text, not the tool result), so the
scope here is the detectable upstream cause: a result that carries a failure
signal the prefix check would wave through as success.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


# Results that strip down to one of these are empty in disguise: the tool
# returned "nothing" rather than an error, so the swallowed failure is total.
_NULL_SENTINELS = frozenset({"", "null", "none", "undefined", "nil", "[]", "{}"})

# HTTP status code (4xx/5xx) co-occurring with an http-ish keyword or a known
# reason phrase. Requiring the co-occurrence keeps a stray "404" inside prose
# from tripping the detector.
_HTTP_REASON = (
    r"internal server error|bad gateway|service unavailable|gateway timeout|"
    r"not found|forbidden|unauthorized|bad request|too many requests|"
    r"request timed? ?out|moved permanently"
)
_HTTP_ERROR_RE = re.compile(
    r"(?i)(?:"
    r"(?:http|status|response|code)\b[^\n]{0,24}?\b[45]\d\d\b"
    r"|\b[45]\d\d\b[^\n]{0,24}?(?:" + _HTTP_REASON + r")"
    r")"
)

# A non-zero process exit reported mid-string (zero is success; the prefix
# list already owns a leading "Exit code").
_EXIT_CODE_RE = re.compile(r"(?i)\bexit (?:code|status)\s+(?!0\b)\d+")

# A Python traceback that did not happen to land at the front of the result.
_TRACEBACK_MARKER = "Traceback (most recent call last)"

# JSON-ish error envelopes a tool may hand back with a 200-shaped body.
_JSON_ERROR_STRING_RE = re.compile(r'(?i)"errors?"\s*:\s*"(?P<msg>[^"]+)"')
_JSON_ERROR_CONTAINER_RE = re.compile(r'(?i)"errors?"\s*:\s*[\[{]\s*[^\]}\s]')
_JSON_FALSE_FLAG_RE = re.compile(r'(?i)"(?:ok|success|succeeded)"\s*:\s*false\b')


@dataclass(frozen=True)
class SilentFailureSignal:
    """A matched silent-failure signal, kept lightweight for logging/audit.

    ``kind`` names the mechanism (so an audit trail can group incidents the
    way the paper's taxonomy does); ``evidence`` is a short snippet of the
    triggering text for the same reason.
    """

    kind: str
    evidence: str


def _snippet(text: str, *, limit: int = 120) -> str:
    """Collapse whitespace and clip, so evidence stays log-friendly."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _empty_result_signal(stripped: str) -> SilentFailureSignal | None:
    if stripped.casefold() in _NULL_SENTINELS:
        return SilentFailureSignal("empty_result", _snippet(stripped) or "<empty>")
    return None


def _json_envelope_signal(stripped: str) -> SilentFailureSignal | None:
    """Catch a swallowed error reported inside a structured (JSON) body."""
    # Fast path: a whole-result JSON object with a truthy error / falsey-ok.
    if stripped[:1] in "{[":
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            for key in ("error", "errors"):
                value = parsed.get(key)
                if value:
                    return SilentFailureSignal("json_error_field", _snippet(str(value)))
            for key in ("ok", "success", "succeeded"):
                if parsed.get(key) is False:
                    return SilentFailureSignal("json_false_flag", _snippet(stripped))

    # Substring path: the body may be wrapped in prose or be a partial dump,
    # so fall back to the regexes that don't need a clean parse.
    match = _JSON_ERROR_STRING_RE.search(stripped)
    if match:
        return SilentFailureSignal("json_error_field", _snippet(match.group("msg")))
    if _JSON_ERROR_CONTAINER_RE.search(stripped):
        return SilentFailureSignal("json_error_field", _snippet(stripped))
    if _JSON_FALSE_FLAG_RE.search(stripped):
        return SilentFailureSignal("json_false_flag", _snippet(stripped))
    return None


def detect_silent_failure(result: str) -> SilentFailureSignal | None:
    """Return the first silent-failure signal in ``result``, else ``None``.

    Only inspects results the prefix check would *not* already flag; the
    caller keeps the cheap ``startswith`` fast path. Detection is
    intentionally conservative — each signal requires corroborating context
    (a status keyword next to the code, a non-zero exit, a parsed error
    field) so ordinary successful output is not mislabeled as a failure.
    """
    if not isinstance(result, str):
        return None
    stripped = result.strip()

    empty = _empty_result_signal(stripped)
    if empty is not None:
        return empty

    if _TRACEBACK_MARKER in result:
        return SilentFailureSignal("python_traceback", _TRACEBACK_MARKER)

    http = _HTTP_ERROR_RE.search(result)
    if http is not None:
        return SilentFailureSignal("http_error_status", _snippet(http.group(0)))

    exit_match = _EXIT_CODE_RE.search(result)
    if exit_match is not None:
        return SilentFailureSignal("nonzero_exit", _snippet(exit_match.group(0)))

    envelope = _json_envelope_signal(stripped)
    if envelope is not None:
        return envelope

    return None


def is_silent_failure(result: str) -> bool:
    """Boolean form for the failure-detection seam (``is_tool_error``)."""
    return detect_silent_failure(result) is not None
