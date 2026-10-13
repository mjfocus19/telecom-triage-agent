import asyncio

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from telecomtriage import tracing
from tests.test_recording import Script, lookup


def _run(outer):
    async def go():
        svc = InMemorySessionService()
        runner = Runner(agent=outer, app_name="t", session_service=svc)
        await svc.create_session(app_name="t", user_id="u", session_id="s")
        msg = types.Content(role="user", parts=[types.Part(text="hi")])
        async for _ in runner.run_async(user_id="u", session_id="s", new_message=msg):
            pass

    asyncio.run(go())


def _agents():
    inner = Agent(name="inner", model=Script(call="lookup"), description="d", instruction="x", tools=[lookup])
    return Agent(name="outer", model=Script(call="inner"), instruction="x", tools=[AgentTool(agent=inner)])


def test_inner_calls_reach_the_sink_in_order():
    outer = _agents()
    tracing.instrument(outer)
    lines = []
    token = tracing.set_sink(lines.append)
    try:
        _run(outer)
    finally:
        tracing.reset_sink(token)
    assert lines == ['   -> lookup({"x": "1"})', "   <- lookup returned"]


def test_no_sink_is_silent_and_instrument_is_idempotent():
    outer = _agents()
    tracing.instrument(outer)
    tracing.instrument(outer)
    inner_tool = outer.tools[0].agent.tools[0]
    assert getattr(inner_tool, "_traced", False) and not hasattr(inner_tool.__wrapped__, "_traced")
    _run(outer)  # must not raise without a sink


def _emit_from_thread(line):
    import threading

    t = threading.Thread(target=lambda: tracing._emit(line))  # a new thread starts with an empty context
    t.start()
    t.join()


def test_single_active_request_gets_lines_even_without_context():
    lines = []
    token = tracing.set_sink(lines.append)
    try:
        _emit_from_thread("x")
    finally:
        tracing.reset_sink(token)
    assert lines == ["x"]


def test_overlapping_requests_never_cross_streams():
    a, b = [], []
    ta, tb = tracing.set_sink(a.append), tracing.set_sink(b.append)
    try:
        _emit_from_thread("lost")  # ambiguous: dropped, not shown to the wrong visitor
        tracing._emit("b-line")    # same context as the last set_sink call (b)
    finally:
        tracing.reset_sink(tb)
        tracing.reset_sink(ta)
    assert a == [] and b == ["b-line"]
