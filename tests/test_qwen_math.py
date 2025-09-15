import math
import re

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def _extract_boxed_number(text: str) -> float:
    """
    Extract the final numeric value from LaTeX-like \\boxed{...} tokens.
    Robust to an extra trailing 'boxed{...}' without the backslash.
    Returns float to allow answers like '4.0' or '4e0'.
    Raises AssertionError if no numeric payload is found.
    """
    # Find all \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    # Also catch bare boxed{...} (missing backslash) just in case
    bare_boxed = re.findall(r"(?<!\\)boxed\{([^}]+)\}", text)

    candidates = boxed + bare_boxed
    assert candidates, f"No boxed answer found in output:\n{text}"

    # Take the last occurrence
    payload = candidates[-1].strip()

    # Extract the final numeric from the payload (handles things like '4', '4.0', ' 4 ', etc.)
    m = re.search(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", payload)
    assert m, f"Boxed payload is not numeric: {payload!r}"
    return float(m.group(1))


@pytest.mark.slow
def test_qwen_basic_math_answer_is_4() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")

    prompt = (
        "system\n"
        "You are a helpful assistant. Please show your reasoning. "
        "Return the final numeric answer as \\boxed{...}\n"
        "user\n"
        "What is 2+2?\n"
        "assistant"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,  # deterministic
        temperature=0.0,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    val = _extract_boxed_number(text)
    assert math.isclose(
        val, 4.0, rel_tol=0, abs_tol=0
    ), f"Expected 4, got {val}.\nFull output:\n{text}"
