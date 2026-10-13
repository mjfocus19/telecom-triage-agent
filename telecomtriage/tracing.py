"""Surface tool calls made inside sub-agents.

When the router calls a specialist through AgentTool, ADK runs the specialist in a nested
run, so the outer Runner only reports the call to the specialist. `instrument` wraps every
function tool reachable from an agent so each call is reported to the sink set for the
current request (a ContextVar, so concurrent users do not see each other's calls).
"""
import functools
import inspect
import json
import threading
from contextvars import ContextVar
from typing import Callable, Optional

from google.adk.tools.agent_tool import AgentTool

_sink: ContextVar[Optional[Callable[[str], None]]] = ContextVar("trace_sink", default=None)

# Fallback for runtimes that run tools where the ContextVar is not visible (worker threads, other
# event loops). If exactly one request is active the line must belong to it; with several active
# requests it cannot be attributed, so it is dropped rather than shown to the wrong visitor.
_active: list = []
_lock = threading.Lock()


def set_sink(fn: Optional[Callable[[str], None]]):
    """Route trace lines to `fn` for the current request. Returns a token for reset_sink."""
    with _lock:
        if fn:
            _active.append(fn)
    return (_sink.set(fn), fn)


def reset_sink(token) -> None:
    ctx_token, fn = token
    _sink.reset(ctx_token)
    with _lock:
        if fn in _active:
            _active.remove(fn)


def _emit(line: str) -> None:
    fn = _sink.get()
    if fn is None:
        with _lock:
            fn = _active[0] if len(_active) == 1 else None
    if fn:
        fn(line)


def _fmt_args(kwargs: dict) -> str:
    return json.dumps(kwargs, default=str)[:200]


def _wrap(fn):
    if getattr(fn, "_traced", False):
        return fn

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def awrapper(*a, **kw):
            _emit(f"   -> {fn.__name__}({_fmt_args(kw)})")
            result = await fn(*a, **kw)
            _emit(f"   <- {fn.__name__} returned")
            return result
        awrapper._traced = True
        return awrapper

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        _emit(f"   -> {fn.__name__}({_fmt_args(kw)})")
        result = fn(*a, **kw)
        _emit(f"   <- {fn.__name__} returned")
        return result
    wrapper._traced = True
    return wrapper


def instrument(agent, _seen: Optional[set] = None) -> None:
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
