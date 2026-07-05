# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for intention-aware MCP tool selection (SING-style active discovery).

The unit cases cover the pure scorer in ``core.inference.intention_tool_selection``;
the ``wrapper_*`` cases import the existing ``core.inference.tools`` module and
exercise the call-site wiring end to end (proving the new code is reached from
the inference path, not just self-tested).
"""

import asyncio
import types

from core.inference import intention_tool_selection as its


def _spec(name: str, description: str) -> dict:
    """An MCP tool spec in the OpenAI-function shape get_enabled_mcp_tools emits."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _big_library() -> list[dict]:
    """3 tools relevant to web search + 12 unrelated filler tools (15 total)."""
    relevant = [
        _spec("mcp__s__web_search", "Search the web for information"),
        _spec("mcp__s__fetch_url", "Fetch a URL and return its contents"),
        _spec("mcp__s__bing", "Run an online bing search"),
    ]
    filler = [_spec(f"mcp__s__noop{i}", "Echo a value back") for i in range(12)]
    return relevant + filler


# ── latest_user_text ───────────────────────────────────────────────


def test_latest_user_text_from_objects_and_dicts():
    obj = types.SimpleNamespace(role="user", content="search the web")
    assert its.latest_user_text([obj]) == "search the web"
    assert its.latest_user_text([{"role": "user", "content": "hi"}]) == "hi"


def test_latest_user_text_picks_most_recent_user():
    msgs = [
        types.SimpleNamespace(role="user", content="old"),
        types.SimpleNamespace(role="assistant", content="ack"),
        types.SimpleNamespace(role="user", content="new"),
    ]
    assert its.latest_user_text(msgs) == "new"


def test_latest_user_text_handles_multimodal_parts():
    msgs = [
        {
            "role": "user",
            "content": [types.SimpleNamespace(text="alpha "), types.SimpleNamespace(text="beta")],
        }
    ]
    assert its.latest_user_text(msgs) == "alpha  beta"


def test_latest_user_text_empty_when_no_user():
    assert its.latest_user_text([]) == ""
    assert its.latest_user_text([{"role": "assistant", "content": "x"}]) == ""


# ── select_mcp_tools_for_intention ─────────────────────────────────


def test_noop_when_library_within_budget():
    specs = _big_library()
    # Budget >= library size: everything ships (injection isn't costly yet).
    out = its.select_mcp_tools_for_intention("search the web", specs, max_tools=len(specs))
    assert out == specs


def test_narrows_to_budget_and_keeps_relevant():
    specs = _big_library()
    out = its.select_mcp_tools_for_intention("search the web", specs, max_tools=5)
    assert len(out) == 5
    names = {s["function"]["name"] for s in out}
    # All three intention-relevant tools survive; the remaining slots fill in order.
    assert {"mcp__s__web_search", "mcp__s__fetch_url", "mcp__s__bing"} <= names


def test_noop_when_no_intention_signal():
    specs = _big_library()
    # No trigger terms in the task: never filter blind.
    out = its.select_mcp_tools_for_intention("hello there, how are you", specs, max_tools=5)
    assert out == specs


def test_noop_when_nothing_scores():
    # web_search is active but no tool advertises a matching capability.
    specs = [_spec(f"mcp__s__noop{i}", "Just an echo tool") for i in range(15)]
    out = its.select_mcp_tools_for_intention("search the web", specs, max_tools=5)
    assert out == specs


def test_collaboration_surfaces_complementary_tool():
    # "run python code" activates code_execution; file_read is a collaboration
    # hint, so a file tool scores via the graph even though "file" isn't in the task.
    specs = (
        [_spec("mcp__s__python", "Execute python code")]
        + [_spec("mcp__s__read_file", "Read a file from disk")]
        + [_spec(f"mcp__s__noop{i}", "Echo") for i in range(12)]
    )
    out = its.select_mcp_tools_for_intention("run some python code", specs, max_tools=4)
    names = {s["function"]["name"] for s in out}
    assert "mcp__s__read_file" in names


def test_budget_resolvable_from_env(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MCP_TOOL_BUDGET", "2")
    specs = _big_library()
    out = its.select_mcp_tools_for_intention("search the web", specs)
    assert len(out) == 2


# ── integration: tools.get_intention_selected_mcp_tools wiring ─────


def test_wrapper_filters_via_get_enabled_mcp_tools(monkeypatch):
    """The tools-module wrapper discovers via get_enabled_mcp_tools then narrows
    through intention selection — proves the call-site wiring reaches the new code."""
    from core.inference import tools as tools_mod

    specs = _big_library()

    async def fake_get_enabled():
        return list(specs)

    monkeypatch.setattr(tools_mod, "get_enabled_mcp_tools", fake_get_enabled)

    msgs = [{"role": "user", "content": "please search the web"}]
    out = asyncio.run(tools_mod.get_intention_selected_mcp_tools(msgs, max_tools=5))
    assert len(out) == 5
    assert {"mcp__s__web_search", "mcp__s__fetch_url", "mcp__s__bing"} <= {
        s["function"]["name"] for s in out
    }


def test_wrapper_noop_when_no_intention(monkeypatch):
    from core.inference import tools as tools_mod

    specs = _big_library()

    async def fake_get_enabled():
        return list(specs)

    monkeypatch.setattr(tools_mod, "get_enabled_mcp_tools", fake_get_enabled)

    msgs = [{"role": "user", "content": "hello"}]
    out = asyncio.run(tools_mod.get_intention_selected_mcp_tools(msgs, max_tools=5))
    assert out == specs


def test_wrapper_empty_when_no_tools(monkeypatch):
    from core.inference import tools as tools_mod

    async def fake_get_enabled():
        return []

    monkeypatch.setattr(tools_mod, "get_enabled_mcp_tools", fake_get_enabled)
    out = asyncio.run(
        tools_mod.get_intention_selected_mcp_tools([{"role": "user", "content": "x"}])
    )
    assert out == []
