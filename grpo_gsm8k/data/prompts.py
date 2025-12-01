from transformers import AutoTokenizer

DEFAULT_SYSTEM = (
    "You are a careful mathematician. Solve step by step and put the final "
    "numeric answer in \\boxed{...}."
)

INSTRUCTION_SUFFIX = "Please show your reasoning. End with only \\boxed{<number>}."


def build_messages(
    question: str,
    system_text: str = DEFAULT_SYSTEM,
    few_shot_examples: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build chat messages with optional k-shot examples followed by target question."""
    messages = [{"role": "system", "content": system_text}]

    # Add few-shot examples as user/assistant pairs
    if few_shot_examples:
        for ex in few_shot_examples:
            messages.extend(
                [
                    {"role": "user", "content": f"{ex['prompt']}\n\n{INSTRUCTION_SUFFIX}"},
                    {"role": "assistant", "content": ex["response"]},
                ]
            )

    # Add target question
    messages.append({"role": "user", "content": f"{question}\n\n{INSTRUCTION_SUFFIX}"})

    return messages


def render_batch(
    tokenizer: AutoTokenizer,
    questions: list[str],
    add_generation_prompt: bool = True,
    few_shot_examples: list[dict[str, str]] | None = None,
) -> list[str]:
    """Render batch with optional k-shot examples prepended to each question."""
    return [
        tokenizer.apply_chat_template(
            build_messages(q, few_shot_examples=few_shot_examples),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        for q in questions
    ]
