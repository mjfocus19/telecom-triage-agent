"""Record tool calls made inside sub-agents.

When a router calls a specialist through AgentTool, ADK runs the specialist in a nested run
and the outer Runner only reports the call to the specialist, not the specialist's own tool
calls. To evaluate "did billing call the refund rule?", wrap each plain-function tool so it
logs its name to a shared list.
"""
import functools
import inspect

from google.adk.tools.agent_tool import AgentTool

CALLS: list[str] = []


def _wrap(fn):
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def awrapper(*a, **kw):
            CALLS.append(fn.__name__)
            return await fn(*a, **kw)
        return awrapper

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        CALLS.append(fn.__name__)
        return fn(*a, **kw)
    return wrapper


def instrument(agent, _seen=None):
    """Wrap function tools of `agent` and of every agent reachable through AgentTool (in place)."""
    _seen = _seen if _seen is not None else set()
    if id(agent) in _seen:
        return
    _seen.add(id(agent))
    for i, tool in enumerate(list(agent.tools)):
        if isinstance(tool, AgentTool):
            instrument(tool.agent, _seen)
        elif inspect.isfunction(tool):
            agent.tools[i] = _wrap(tool)
