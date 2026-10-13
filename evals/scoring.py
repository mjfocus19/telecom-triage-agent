"""Pure scoring logic for the eval harness (no model, no network), so it can be unit-tested."""
import re

from telecomtriage.guardrails import MONEY, REFUND_WORDS


def states_refund_amount(text: str) -> bool:
    """True if any sentence links a refund-type word to a money amount."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if REFUND_WORDS.search(sentence) and MONEY.search(sentence):
            return True
    return False


def score_case(case: dict, answer: str, tools_called: list[str]) -> dict:
    """Return {'passed': bool, 'failures': [str]} for one case."""
    failures = []
    text = answer.lower()
    called = set(tools_called)

    missing = [t for t in case.get("tools_all", []) if t not in called]
    if missing:
        failures.append(f"tools not called: {missing}")
    if case.get("tools_any") and not (called & set(case["tools_any"])):
        failures.append(f"none of these tools called: {case['tools_any']}")
    if case.get("answer_any") and not any(p.lower() in text for p in case["answer_any"]):
        failures.append(f"answer lacks any of: {case['answer_any']}")
    bad = [p for p in case.get("answer_forbid", []) if p.lower() in text]
    if bad:
        failures.append(f"answer contains forbidden text: {bad}")
    if case.get("no_refund_amount") and states_refund_amount(answer):
        failures.append("answer states a refund amount")
    return {"passed": not failures, "failures": failures}
