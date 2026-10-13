"""Run the eval cases against the live agent and report pass/fail.

Usage (from the repo root, with GOOGLE_API_KEY set or telecomtriage/.env present):
    python -m evals.run_evals            # all cases
    python -m evals.run_evals --runs 3   # repeat each case to expose flakiness

Each case gets a fresh session. Exit code is 1 if any case fails, so this can gate CI.
Results are also written to evals/results.json.
"""
import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / "telecomtriage" / ".env")
except ImportError:
    pass
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

from google.genai import types  # noqa: E402

from evals.scoring import score_case  # noqa: E402


async def run_case(runner, svc, question: str):
    sid = uuid.uuid4().hex
    await svc.create_session(app_name="evals", user_id="eval", session_id=sid)
    msg = types.Content(role="user", parts=[types.Part(text=question)])
    tools, answer = [], ""
    async for ev in runner.run_async(user_id="eval", session_id=sid, new_message=msg):
        tools += [c.name for c in (ev.get_function_calls() or [])]
        if ev.is_final_response() and ev.content and ev.content.parts:
            answer = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))
    return answer, tools


async def main(runs: int) -> int:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    from telecomtriage.agent import root_agent

    svc = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="evals", session_service=svc)
    cases = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))

    results, failed = [], 0
    for case in cases:
        for n in range(runs):
            t0 = time.monotonic()
            try:
                answer, tools = await run_case(runner, svc, case["question"])
                verdict = score_case(case, answer, tools)
            except Exception as exc:  # an agent error is a failed case, not a crashed run
                answer, tools, verdict = "", [], {"passed": False, "failures": [f"error: {type(exc).__name__}: {exc}"]}
            failed += not verdict["passed"]
            results.append({"id": case["id"], "run": n + 1, "passed": verdict["passed"], "failures": verdict["failures"],
                            "tools": tools, "answer": answer, "seconds": round(time.monotonic() - t0, 1)})
            status = "PASS" if verdict["passed"] else "FAIL"
            print(f"{status}  {case['id']} (run {n + 1})  {results[-1]['seconds']}s")
            for f in verdict["failures"]:
                print(f"      - {f}")

    total = len(results)
    print(f"\n{total - failed}/{total} passed")
    (ROOT / "evals" / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    sys.exit(asyncio.run(main(parser.parse_args().runs)))
