from evals.scoring import score_case, states_refund_amount


def test_pass():
    case = {"tools_all": ["a"], "answer_any": ["eligible"], "no_refund_amount": True}
    assert score_case(case, "You are eligible.", ["a", "b"])["passed"]


def test_missing_tool_fails():
    r = score_case({"tools_all": ["a", "b"]}, "x", ["a"])
    assert not r["passed"] and "tools not called" in r["failures"][0]


def test_tools_any():
    assert score_case({"tools_any": ["a", "b"]}, "x", ["b"])["passed"]
    assert not score_case({"tools_any": ["a", "b"]}, "x", [])["passed"]


def test_forbidden_text_fails():
    assert not score_case({"answer_forbid": ["not eligible"]}, "You are NOT eligible", [])["passed"]


def test_refund_amount_detected():
    assert states_refund_amount("We will refund you $20.")
    assert not states_refund_amount("Your bill is $95.50.")
    assert not score_case({"no_refund_amount": True}, "Refund of $20 approved.", [])["passed"]


def test_en_dash_is_normalized():
    assert score_case({"answer_any": ["5-7"]}, "Takes 5\u20137 business days.", [])["passed"]


def test_tools_none():
    assert not score_case({"tools_none": ["a"]}, "x", ["a"])["passed"]
    assert score_case({"tools_none": ["a"]}, "x", ["b"])["passed"]
