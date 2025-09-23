from grpo_gsm8k.format_r1_traces import clean_reasoning


def test_clean_reasoning_filters_instructional_noise_and_keeps_reasoning() -> None:
    raw_lines = [
        "First step.",
        "",
        'Finally, output the answer as "ANSWER: 85".',
        (
            "Now, for the output, I need to put the final answer alone on its own line as: "
            "ANSWER: 5"
        ),
        'The problem says "show your reasoning" so I\'ll write it out.',
        (
            'I should make sure about the format. It says "output the final numeric answer alone '
            "on its own line\", so I'll do that."
        ),
        "So, ANSWER: 24",
        "ANSWER: 1080",
        "",
        r"Second step with inline box \boxed{42} that should go away.",
        "",
        "Last step.",
    ]
    raw = "\n".join(raw_lines)
    cleaned = clean_reasoning(raw)

    assert "Finally, output the answer" not in cleaned
    assert "Now, for the output" not in cleaned
    assert "show your reasoning" not in cleaned
    assert "I should make sure about the format" not in cleaned
    assert "output the final numeric answer alone on its own line" not in cleaned
    assert "ANSWER:" not in cleaned
    assert "\\boxed{" not in cleaned

    assert "First step." in cleaned
    assert "Second step with inline box" in cleaned and "that should go away." in cleaned
    assert "Last step." in cleaned

    lines = cleaned.splitlines()
    assert lines[0].strip() != ""
    assert lines[-1].strip() != ""
    for i in range(len(lines) - 1):
        assert not (lines[i].strip() == "" and lines[i + 1].strip() == "")
