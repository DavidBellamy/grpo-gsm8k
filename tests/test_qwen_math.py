from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def _extract_boxed_number(text: str) -> float:
    """
    Strictly extract the final numeric value from \\boxed{...} (or bare boxed{...}).
    Fails if:
      - No boxed pattern is present
      - The boxed payload does not contain a numeric token
    """
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    bare_boxed = re.findall(r"(?<!\\)boxed\{([^}]+)\}", text)
    candidates = boxed + bare_boxed
    assert candidates, f"No boxed answer found in output:\n{text}"
    payload = candidates[-1].strip()
    m = re.search(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", payload)
    assert m, f"Boxed payload is not numeric: {payload!r}\nFull output:\n{text}"
    return float(m.group(1))


@pytest.mark.slow
def test_qwen_basic_math_answer_is_4() -> None:
    tfm = pytest.importorskip(
        "transformers",
        reason="requires transformers (and likely torch) to run this slow test",
    )
    torch = pytest.importorskip("torch", reason="requires torch for model execution")

    AutoTokenizer = tfm.AutoTokenizer
    AutoModelForCausalLM = tfm.AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")

    # Use official chat template for more reliable instruction following.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful math assistant. Show concise reasoning, then output ONLY the "
                "final numeric answer wrapped exactly as \\boxed{<number>}>."
            ),
        },
        {"role": "user", "content": "What is 2+2?"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    val = _extract_boxed_number(text)
    assert math.isclose(
        val, 4.0, rel_tol=0, abs_tol=0
    ), f"Expected 4, got {val}.\nFull output:\n{text}"
