from grpo_gsm8k.reward_fn import exact_match, extract_answer_colon, reward_from_text


def test_extract_answer_colon_variants() -> None:
    txt = """
    some reasoning...
    Answer:  42
    extra stuff
    ANSWER:  43
    """
    assert extract_answer_colon(txt) == "43"


def test_exact_match_answer_parser_match() -> None:
    pred = "chain of thought...\nANSWER: 128\n"
    gold = "work...\n#### 128"
    assert exact_match(pred, gold, parser="answer") == 1
    assert reward_from_text(pred, gold, parser="answer") == 1.0


def test_exact_match_answer_parser_mismatch() -> None:
    pred = "...\nANSWER: 127\n"
    gold = "...\n#### 128"
    assert exact_match(pred, gold, parser="answer") == 0
    assert reward_from_text(pred, gold, parser="answer") == 0.0
