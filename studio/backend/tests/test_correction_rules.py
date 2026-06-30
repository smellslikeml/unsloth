# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for compiled-from-corrections tool-call enforcement (TRACE).

A user's chat-history correction is mined into an atomic deny-rule and then
enforced at the same gate as the command blocklist. The end-to-end test goes
through the existing ``execute_tool`` public surface (like
``test_sandbox_tools``): when a rule fires, ``_bash_exec`` returns the block
message *before* spawning any subprocess, so the blocked path is hermetic.
"""

from __future__ import annotations

from core.inference import correction_rules
from core.inference.correction_rules import compile_corrections
from core.inference.tools import execute_tool


def test_execute_tool_blocks_terminal_command_corrected_by_user():
    # "pip" is NOT in the static blocklist, so a block here is attributable to
    # the correction layer, not _find_blocked_commands.
    messages = [
        {"role": "user", "content": "Please use uv instead of pip for installs."},
        {"role": "assistant", "content": "Got it, I'll use uv."},
        {"role": "user", "content": "install numpy for me"},
    ]
    rules = compile_corrections(messages)
    assert any(r.kind == "deny_command" and r.matcher == "pip" for r in rules)

    result = execute_tool(
        "terminal",
        {"command": "pip install numpy"},
        correction_rules = rules,
    )
    assert result.startswith("Blocked by your earlier correction")
    assert "pip" in result


def test_execute_tool_runs_command_when_no_correction_applies():
    rules = compile_corrections([{"role": "user", "content": "Use uv instead of pip."}])
    # An unrelated command is not blocked by the correction gate.
    assert correction_rules.check_correction_rules("terminal", {"command": "ls -la"}, rules) is None


def test_compile_corrections_mines_action_and_instead_of():
    rules = compile_corrections(
        [
            {"role": "user", "content": "Never run make in this repo."},
            {"role": "user", "content": "We use ruff not black for formatting."},
        ]
    )
    terms = {(r.kind, r.matcher) for r in rules}
    assert ("deny_command", "make") in terms
    assert ("deny_command", "black") in terms
    # The preferred tools the user named are never denied.
    assert all(r.matcher != "ruff" for r in rules)


def test_correction_fires_only_at_command_position():
    rules = compile_corrections([{"role": "user", "content": "Don't use grep."}])
    # grep at command position is blocked...
    assert correction_rules.find_violations("terminal", {"command": "grep -r foo ."}, rules)
    # ...but grep appearing inside an argument is not.
    assert not correction_rules.find_violations(
        "terminal", {"command": "echo grep results here"}, rules
    )


def test_only_user_turns_are_mined():
    rules = compile_corrections(
        [
            {"role": "assistant", "content": "Sure, I'll use curl to fetch that."},
            {"role": "user", "content": "Actually never use curl here."},
        ]
    )
    assert len(rules) == 1
    assert rules[0].matcher == "curl"


def test_python_import_correction_compiles_and_matches():
    rules = compile_corrections(
        [{"role": "user", "content": "Never import pickle, use json instead."}]
    )
    assert correction_rules.check_correction_rules(
        "python", {"code": "import pickle"}, rules
    ) is not None
    # A different module is allowed.
    assert correction_rules.check_correction_rules(
        "python", {"code": "import json"}, rules
    ) is None
    # No rules -> allow anything.
    assert correction_rules.check_correction_rules("python", {"code": "import pickle"}, []) is None
