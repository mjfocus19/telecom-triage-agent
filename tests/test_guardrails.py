import pytest

from telecomtriage.guardrails import REPLACEMENT, redact_refund_amounts


@pytest.mark.parametrize(
    "text",
    [
        "You will receive a refund of $25.00.",
        "We will credit your account 40 dollars for the outage.",
        "Your reimbursement would be $1,200 USD.",
    ],
)
def test_refund_amounts_are_redacted(text):
    out, changed = redact_refund_amounts(text)
    assert changed
    assert "$" not in out and REPLACEMENT in out


@pytest.mark.parametrize(
    "text",
    [
        "Your current bill is $95.50.",
        "You are eligible for a refund. A reviewer will confirm the amount.",
        "Our return window is 30 days.",
    ],
)
def test_safe_text_is_untouched(text):
    out, changed = redact_refund_amounts(text)
    assert not changed and out == text


def test_only_offending_sentence_is_replaced():
    out, changed = redact_refund_amounts("You qualify. We will refund $30. Anything else?")
    assert changed
    assert out == f"You qualify. {REPLACEMENT} Anything else?"


def test_known_false_positive_is_documented():
    # Same sentence mentions a bill amount and a refund: redacted on purpose (safe direction).
    _, changed = redact_refund_amounts("Your bill is $95.50 and you may get a refund.")
    assert changed
