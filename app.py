"""Streamlit front end for the telecom triage agent.

Prototype demo: mock data, in-memory sessions, no production hardening.

Run locally:  streamlit run app.py
Deploy:       set GOOGLE_API_KEY (and optionally APP_PASSWORD) in the host's secrets.
"""
import asyncio
import hmac
import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Telecom Care Triage Agent",
    page_icon="📡",
    layout="centered",
    menu_items={
        "About": (
            "**Telecom Care Triage Agent**: a multi-agent customer-care prototype built with Google ADK and Gemini. "
            "All data is fictional. Source: https://github.com/mjfocus19/telecom-triage-agent"
        ),
        "Report a bug": "https://github.com/mjfocus19/telecom-triage-agent/issues",
        "Get help": "https://github.com/mjfocus19/telecom-triage-agent#readme",
    },
)

APP_DIR = Path(__file__).parent
# agent.py opens ./billing.db and ./chroma_db_recap relative to the working
# directory, so pin it to the repo root no matter where Streamlit is launched from.
os.chdir(APP_DIR)

APP_NAME = "telecomtriage"
USER_ID = "demo_user"
MAX_TURNS = 20  # per browser session; keeps a shared demo link from running up API costs

log = logging.getLogger("triage_app")

# --- Configuration -----------------------------------------------------------
# The API key must be in the environment BEFORE the agent module is imported:
# agent.py creates a Gemini client and seeds the vector store at import time.
try:
    from dotenv import load_dotenv

    load_dotenv(APP_DIR / "telecomtriage" / ".env")  # local runs only; the file is gitignored
except ImportError:
    pass


def _secret(name: str):
    try:
        return st.secrets[name]
    except Exception:  # no secrets file locally, or key not set
        return None


if _secret("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = str(_secret("GOOGLE_API_KEY"))
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
    st.error("No Gemini API key found. Set GOOGLE_API_KEY in telecomtriage/.env (local) or in the app secrets (deployed).")
    st.stop()

# --- Optional access code ----------------------------------------------------
_password = _secret("APP_PASSWORD")
if _password and not st.session_state.get("authed"):
    st.title("Telecom Care Triage Agent")
    entered = st.text_input("Access code", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), str(_password).encode()):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Incorrect access code.")
    st.stop()

from google.genai import types  # noqa: E402  (after env setup)


# --- Agent runtime -----------------------------------------------------------
@st.cache_resource(show_spinner="Starting the agent (first start builds the knowledge base)...")
def load_runtime():
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    from telecomtriage import tracing
    from telecomtriage.agent import root_agent

    tracing.instrument(root_agent)  # show tool calls made inside the specialists in the trace
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    return runner, session_service


@st.cache_resource
def _event_loop() -> asyncio.AbstractEventLoop:
    """One long-lived loop on a background thread.

    Streamlit reruns the script on every interaction; reusing a single loop avoids
    'event loop is closed' errors from async HTTP clients that outlive one asyncio.run().
    """
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True, name="agent-loop").start()
    return loop


def run_turn(runner, session_service, session_id: str, text: str, on_trace, timeout: int = 120):
    """Run one turn on the background loop and relay trace lines to the UI thread as they arrive."""
    events: queue.Queue = queue.Queue()

    async def _produce():
        try:
            result = await ask_agent(
                runner, session_service, session_id, text, on_trace=lambda line: events.put(("trace", line))
            )
            events.put(("done", result))
        except Exception as exc:
            events.put(("error", exc))

    future = asyncio.run_coroutine_threadsafe(_produce(), _event_loop())
    deadline = time.monotonic() + timeout
    while True:
        try:
            kind, payload = events.get(timeout=max(0.1, deadline - time.monotonic()))
        except queue.Empty:
            future.cancel()
            raise TimeoutError("The agent did not respond in time.") from None
        if kind == "trace":
            on_trace(payload)
        elif kind == "done":
            return payload
        else:
            raise payload


async def ask_agent(runner, session_service, session_id: str, text: str, on_trace=None):
    """Send one user turn through the router agent. Returns (answer, trace_lines)."""
    existing = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    if existing is None:
        await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    message = types.Content(role="user", parts=[types.Part(text=text)])
    answer, trace = "", []

    def note(line: str):
        trace.append(line)
        if on_trace:
            on_trace(line)

    from telecomtriage import tracing

    token = tracing.set_sink(note)  # tool calls inside sub-agents report here
    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=message):
            _collect(event, note)
            if event.is_final_response() and event.content and event.content.parts:
                answer = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
    finally:
        tracing.reset_sink(token)
    return answer or "(The agent returned no text.)", trace


def _collect(event, note):
    """Turn one ADK event into trace lines."""
    for call in event.get_function_calls() or []:
        args = json.dumps(dict(call.args or {}), default=str)
        note(f"-> {call.name}({args[:200]})")
    for resp in event.get_function_responses() or []:
        note(f"<- {resp.name} returned")


# --- UI ------------------------------------------------------------------------
runner, session_service = load_runtime()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []
    st.session_state.turns = 0

GITHUB_URL = "https://github.com/mjfocus19/telecom-triage-agent"

EXAMPLES = [
    "My account is ABC123 and I had an outage. Am I eligible for a refund?",
    "Can I return my phone if I've had it for three weeks?",
    "Do I get a credit when my service is down?",
    "DEF123 had an outage. Am I eligible, and when will my pre-ordered phone ship?",
]

ARCHITECTURE_DOT = r"""
digraph {
  rankdir=LR;
  bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#1f2937", color="#60a5fa", fontcolor="#f3f4f6",
        fontname="Helvetica", fontsize=11];
  edge [color="#9ca3af"];
  Q  [label="Customer\nquestion"];
  R  [label="Router agent"];
  B  [label="Billing\nspecialist"];
  P  [label="Policy\nspecialist"];
  T1 [label="get_account_usage\n(SQLite lookup)"];
  T2 [label="check_refund_eligibility\n(business rule)"];
  T3 [label="search_policy_docs\n(ChromaDB + Gemini embeddings)"];
  Q -> R; R -> B; R -> P; B -> T1; B -> T2; P -> T3;
}
"""

st.title("Telecom Care Triage Agent")
st.caption(
    "A working prototype of a multi-agent AI assistant for customer care, built with Google's Agent Development Kit "
    "(ADK) and Gemini. All data is fictional."
)

with st.sidebar:
    st.subheader("Try these")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"side_{i}", width="stretch"):
            st.session_state.pending = example
    st.divider()
    show_trace = st.toggle("Show routing and tool trace", value=True)
    if st.button("Reset conversation", width="stretch"):
        for key in ("session_id", "messages", "turns"):
            st.session_state.pop(key, None)
        st.rerun()
    st.divider()
    st.caption(
        "Sample accounts: ABC123 (unlimited plan, 180 outage minutes) and DEF123 (limited plan, 50 outage minutes). "
        "All data is fictional. The agent never promises an exact refund amount; it flags eligibility for human review."
    )

first_visit = not st.session_state.messages and not st.session_state.get("pending")

with st.expander("About this demo", expanded=first_visit):
    st.markdown(
        "**What it is.** A router agent reads each customer question and hands it to a billing specialist or a "
        "policy specialist. The specialists use tools: a database lookup, a refund rule, and search over policy "
        "documents. The router decides who handles each question."
    )
    st.markdown("**What each example shows**")
    st.markdown(
        "| Example | What to look for |\n"
        "|---|---|\n"
        "| Outage refund (ABC123) | **Chained tools:** account lookup, then an eligibility rule. **Guardrail:** it confirms "
        "eligibility, and code blocks any response that states a refund amount. |\n"
        "| Phone return | **RAG:** finds the matching policy text in a vector store and answers from it. |\n"
        "| Credit during an outage | **Routing:** a general policy question with no account, so the router picks the policy "
        "specialist and never touches billing. |\n"
        "| Outage plus pre-order (DEF123) | **Multi-step routing:** one question needs both specialists. The trace shows the "
        "router calling billing and policy in the same turn, and the answer combines both. |"
    )
    st.markdown("**How it is wired**")
    st.graphviz_chart(ARCHITECTURE_DOT, width="stretch")
    st.markdown("**What is real, and what is not**")
    st.markdown(
        "- **Real:** a working router and two specialist agents, chained tool calls, retrieval over a vector store, "
        "a refund guardrail enforced in code (not only in the prompt), and unit tests plus an evaluation script in the "
        "GitHub repo.\n"
        "- **Mock or not built yet:** the data is two sample accounts and seven short policy documents. Sessions live in "
        "memory only. The guardrail is pattern-based, so it can miss unusual phrasings. The evaluation script is run "
        "by hand, not automatically, and this deployment has no login, monitoring or cost tracking. This is a "
        "prototype, not a production system."
    )
    st.markdown(f"Built by Amjad Hasnain. [Code and architecture notes on GitHub]({GITHUB_URL})")

if first_visit:
    st.markdown("**Start here:** pick an example, or type your own question below.")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"main_{i}", width="stretch"):
            st.session_state.pending = example
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace") and show_trace:
            with st.expander("Routing and tool trace"):
                st.code("\n".join(msg["trace"]), language="text")
        if msg.get("elapsed"):
            st.caption(f"Answered in {msg['elapsed']:.1f}s")

prompt = st.chat_input("Ask about a bill, an outage refund, or company policy")
if st.session_state.get("pending"):
    prompt = st.session_state.pop("pending")

if prompt:
    if st.session_state.turns >= MAX_TURNS:
        st.warning("This demo session has reached its message limit. Use 'Reset conversation' to start over.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        started = time.monotonic()
        with st.status("Routing your question...", expanded=True) as status:

            def on_trace(line: str):
                if show_trace:
                    status.write(f"`{line}`")

            try:
                answer, trace = run_turn(runner, session_service, st.session_state.session_id, prompt, on_trace)
                status.update(label=f"Answered in {time.monotonic() - started:.1f}s", state="complete", expanded=False)
            except Exception as exc:  # keep the demo friendly; full detail goes to the server log
                log.exception("agent call failed")
                answer = f"Sorry, the agent hit an error ({type(exc).__name__}). Please try again."
                trace = []
                status.update(label="Something went wrong", state="error", expanded=False)
        st.markdown(answer)
        elapsed = time.monotonic() - started

    st.session_state.turns += 1
    st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace, "elapsed": elapsed})
