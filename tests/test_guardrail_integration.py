"""Runs the guard inside a real ADK Agent/Runner with a stub model (no API key, no network)."""
import asyncio

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from telecomtriage.guardrails import REPLACEMENT, refund_amount_guard


class StubLlm(BaseLlm):
    model: str = "stub"
    reply: str = ""

    async def generate_content_async(self, llm_request, stream=False):
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.reply)]))


def run(reply: str, guarded: bool) -> str:
    agent = Agent(
        name="a",
        model=StubLlm(reply=reply),
        instruction="x",
        after_model_callback=refund_amount_guard if guarded else None,
    )

    async def go():
        svc = InMemorySessionService()
        runner = Runner(agent=agent, app_name="t", session_service=svc)
        await svc.create_session(app_name="t", user_id="u", session_id="s")
        msg = types.Content(role="user", parts=[types.Part(text="hi")])
        text = ""
        async for ev in runner.run_async(user_id="u", session_id="s", new_message=msg):
            if ev.is_final_response() and ev.content:
                text = "".join(p.text or "" for p in ev.content.parts)
        return text

    return asyncio.run(go())


def test_unguarded_agent_leaks_amount():
    assert "$50" in run("We will refund $50 today.", guarded=False)


def test_guarded_agent_blocks_amount():
    out = run("We will refund $50 today.", guarded=True)
    assert "$50" not in out and REPLACEMENT in out


def test_guarded_agent_passes_clean_answer():
    assert run("You are eligible for a refund.", guarded=True) == "You are eligible for a refund."
