"""Code-level guardrails for the triage agents.

The billing prompt tells the model not to promise refund amounts. A prompt is a request,
not a control, so this module enforces the rule in code: an ADK `after_model_callback`
inspects every model response and rewrites any sentence that states a refund amount.

Known limits (pattern-based, not semantic):
- A sentence that mentions both a refund and an unrelated dollar figure is redacted
  (false positive; safe direction).
- Amounts written in words ("fifty dollars") or without a $ sign/currency word are not caught.
"""
import re
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.genai import types

REFUND_WORDS = re.compile(r"\b(refund\w*|credit\w*|reimburs\w*|compensat\w*|rebate\w*)\b", re.I)
MONEY = re.compile(r"(\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:usd|dollars?)\b)", re.I)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

REPLACEMENT = "The final refund amount is determined by a human reviewer."


def redact_refund_amounts(text: str) -> tuple[str, bool]:
    """Replace any sentence that links a refund-type word to a money amount.

    Returns (possibly rewritten text, whether anything was redacted).
    """
    changed = False
    out = []
    for piece in re.split(r"(\n+)", text):
        if piece.startswith("\n") or not piece.strip():
            out.append(piece)
            continue
        sentences = SENTENCE_SPLIT.split(piece)
        kept = []
        for s in sentences:
            if s and REFUND_WORDS.search(s) and MONEY.search(s):
                changed = True
                if REPLACEMENT not in kept:
                    kept.append(REPLACEMENT)
            elif s:
                kept.append(s)
        out.append(" ".join(kept))
    return "".join(out), changed


def refund_amount_guard(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """ADK after_model_callback: return a rewritten response if it states a refund amount."""
    content = llm_response.content
    if not content or not content.parts:
        return None
    new_parts, changed = [], False
    for part in content.parts:
        if getattr(part, "text", None) and not getattr(part, "function_call", None):
            new_text, hit = redact_refund_amounts(part.text)
            if hit:
                changed = True
                new_parts.append(types.Part(text=new_text))
                continue
        new_parts.append(part)
    if not changed:
        return None
    return LlmResponse(content=types.Content(role=content.role, parts=new_parts))
