"""Shows why evals need recording: nested tool calls are invisible to the outer Runner."""
import asyncio

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from evals import recording


def lookup(x: str) -> str:
    """Inner tool."""
    return "found " + x


class Script(BaseLlm):
    model: str = "stub"
    call: str = ""  # tool to call first; empty means just answer

    async def generate_content_async(self, llm_request, stream=False):
        last = llm_request.contents[-1].parts[0]
        if self.call and not last.function_response:
            fc = types.FunctionCall(name=self.call, args={"x": "1"} if self.call == "lookup" else {"request": "go"})
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=fc)]))
        else:
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="done")]))


def test_nested_calls_are_recorded():
    inner = Agent(name="inner", model=Script(call="lookup"), description="d", instruction="x", tools=[lookup])
    outer = Agent(name="outer", model=Script(call="inner"), instruction="x", tools=[AgentTool(agent=inner)])
    recording.instrument(outer)
    recording.CALLS.clear()

    async def go():
        svc = InMemorySessionService()
        runner = Runner(agent=outer, app_name="t", session_service=svc)
        await svc.create_session(app_name="t", user_id="u", session_id="s")
        msg = types.Content(role="user", parts=[types.Part(text="hi")])
        return [c.name async for ev in runner.run_async(user_id="u", session_id="s", new_message=msg)
                for c in (ev.get_function_calls() or [])]

    outer_seen = asyncio.run(go())
    assert "inner" in outer_seen and "lookup" not in outer_seen  # outer Runner can't see it
    assert recording.CALLS == ["lookup"]                          # the recorder can
