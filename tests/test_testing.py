import json
from typing import Any, cast

import pytest

from padwan_llm import AgentSession, LLMClientBase, McpTool
from padwan_llm.testing import ScriptedClient, Step


async def _drain(stream) -> str:
    return "".join([chunk async for chunk in stream])


async def test_text_step_streams_text_and_usage() -> None:
    client = ScriptedClient(
        [Step(text="Bonjour", usage={"total": 5, "input": 3, "output": 2})]
    )
    stream = client.stream_chat([{"role": "user", "content": "salut"}])
    assert await _drain(stream) == "Bonjour"
    assert stream.usage == {"total": 5, "input": 3, "output": 2}
    assert stream.tool_calls is None


async def test_tool_step_exposes_openai_shaped_tool_calls() -> None:
    client = ScriptedClient([Step(tool_calls=[("search", {"query": "pelle"})])])
    stream = client.stream_chat(
        [], tools=[{"name": "search", "description": "", "parameters": {}}]
    )
    await _drain(stream)
    (call,) = stream.tool_calls or []
    assert call["type"] == "function"
    assert call["function"]["name"] == "search"
    assert json.loads(call["function"]["arguments"]) == {"query": "pelle"}
    assert call["id"] == "call_1_0"


async def test_requests_are_recorded_and_the_script_ends() -> None:
    client = ScriptedClient([Step(text="ok")])
    await _drain(
        client.stream_chat(
            [{"role": "user", "content": "q"}],
            tools=[{"name": "t", "description": "", "parameters": {}}],
            extra_params={"metadata": {"trace_id": "abc"}},
        )
    )
    (request,) = client.requests
    assert request.messages == [{"role": "user", "content": "q"}]
    assert request.tool_names == ["t"]
    assert request.extra_params == {"metadata": {"trace_id": "abc"}}
    assert client.remaining == 0
    with pytest.raises(AssertionError, match="exhausted after 1 round"):
        client.stream_chat([])


async def test_complete_chat_returns_a_chat_response() -> None:
    client = ScriptedClient([Step(tool_calls=[("echo", {"x": 1})]), Step(text="done")])
    response, usage = await client.complete_chat([{"role": "user", "content": "go"}])
    assert response["finish_reason"] == "tool_calls"
    assert response["content"] is None
    assert [c["function"]["name"] for c in response["tool_calls"]] == ["echo"]
    assert usage["total"] == 10
    response, _ = await client.complete_chat([])
    assert response == {"content": "done", "finish_reason": "stop"}


async def test_it_drives_an_agent_session() -> None:
    seen: list[dict[str, Any]] = []

    async def echo(args: dict[str, Any]) -> dict[str, Any]:
        seen.append(args)
        return {"found": 1}

    client = ScriptedClient([Step(tool_calls=[("echo", {"x": 1})]), Step(text="done")])
    session = AgentSession(
        client=cast(LLMClientBase, client),
        system="s",
        mcp_tools=[McpTool("echo", "", {"type": "object"}, echo)],
    )
    async with session:
        assert client.is_open
        assert await session.send("go") == "done"
    assert not client.is_open
    assert seen == [{"x": 1}]
    assert session.total_usage["total"] == 20  # two steps, default usage each
    assert client.requests[1].messages[-1]["role"] == "tool"
