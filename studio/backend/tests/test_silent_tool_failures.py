# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Silent-failure coverage for the local tool loop's failure seam.

Exercises the wiring in ``core.inference.tool_loop_controller`` (the existing
call site) plus the ``silent_failure_detector`` it now delegates to. The
taxonomy comes from arXiv:2606.14589, class (C) error swallowing/dilution:
results that carry a failure signal the leading-prefix check waves through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.silent_failure_detector import (
    detect_silent_failure,
    is_silent_failure,
)
from core.inference.tool_loop_controller import ToolLoopController, is_tool_error


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def _call(name: str, args: dict, call_id: str = "call_0") -> dict:
    import json

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


SILENT_FAILURES = [
    ('{"ok": false, "data": null}', "json_false_flag"),
    ('{"error": "rate limit exceeded"}', "json_error_field"),
    ('Fetched page.\n{"errors": [{"code": 17}]}', "json_error_field"),
    ("GET /v1/models -> HTTP 503 service unavailable", "http_error_status"),
    ("Response status: 500 while contacting upstream", "http_error_status"),
    ("Ran build step\nexit code 127", "nonzero_exit"),
    ("partial output\nTraceback (most recent call last):\n  File ...", "python_traceback"),
    ("   ", "empty_result"),
    ("null", "empty_result"),
    ("[]", "empty_result"),
]

CLEAN_RESULTS = [
    "Search result\n__IMAGES__:{...}",
    "Rendered HTML canvas: Demo",
    "The HTTP protocol uses status code 200 for success.",
    '{"ok": true, "items": [1, 2, 3]}',
    "Process completed with exit code 0",
    "ok",
]


@pytest.mark.parametrize("result, kind", SILENT_FAILURES)
def test_detector_names_the_failure_mechanism(result, kind):
    signal = detect_silent_failure(result)
    assert signal is not None and signal.kind == kind
    assert is_silent_failure(result)


@pytest.mark.parametrize("result", CLEAN_RESULTS)
def test_clean_results_are_not_silent_failures(result):
    assert detect_silent_failure(result) is None
    assert not is_silent_failure(result)


@pytest.mark.parametrize("result, _kind", SILENT_FAILURES)
def test_is_tool_error_seam_now_catches_silent_failures(result, _kind):
    # The leading-prefix contract still holds for loud failures...
    assert is_tool_error("Error: boom")
    # ...and the seam now also flags the silent ones the prefix check missed.
    assert is_tool_error(result)


def test_swallowed_error_is_not_cached_as_a_successful_call():
    """A silent failure must not be recorded as success, or the model would
    be told the call "already completed successfully" and narrate over it."""
    controller = ToolLoopController(tools=[_tool("web_search")])

    decision = controller.prepare_call(_call("web_search", {"query": "gpu"}))
    completion = controller.record_result(decision, '{"ok": false, "data": null}')

    # Recorded as an error -> retry stays open, no duplicate suppression.
    assert completion.is_error
    assert controller.history[-1].is_error

    retry = controller.prepare_call(_call("web_search", {"query": "gpu"}))
    assert retry.should_execute
    assert retry.action == "execute"


def test_silent_failure_result_gets_the_error_nudge_for_the_model():
    controller = ToolLoopController(tools=[_tool("web_search")])
    decision = controller.prepare_call(_call("web_search", {"query": "gpu"}))
    completion = controller.record_result(decision, "GET /search -> HTTP 502 bad gateway")

    message = completion.model_message()
    assert message["role"] == "tool"
    assert "try a different" in message["content"].lower()


def test_clean_result_still_caches_as_success():
    controller = ToolLoopController(tools=[_tool("web_search")])
    decision = controller.prepare_call(_call("web_search", {"query": "gpu"}))
    completion = controller.record_result(decision, '{"ok": true, "items": [1]}')

    assert not completion.is_error
    duplicate = controller.prepare_call(_call("web_search", {"query": "gpu"}))
    assert duplicate.action == "duplicate"
