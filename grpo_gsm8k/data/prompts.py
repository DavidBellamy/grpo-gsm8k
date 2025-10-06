import torch
from transformers import AutoTokenizer

DEFAULT_SYSTEM = (
    "You are a careful mathematician. Solve step by step and put the final "
    "numeric answer in \\boxed{...}."
)

INSTRUCTION_SUFFIX = "Please show your reasoning. End with only \\boxed{<number>}."


def build_messages(question: str, system_text: str = DEFAULT_SYSTEM) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"{question}\n\n{INSTRUCTION_SUFFIX}"},
    ]


def render_batch(
    tokenizer: AutoTokenizer, questions: list[str], add_generation_prompt: bool = True
) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            build_messages(q),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        for q in questions
    ]


def render_batch_ids(
    tokenizer: AutoTokenizer, questions: list[str]
) -> dict[torch.Tensor, torch.Tensor]:
    """
    Return *tensors* from Qwen's chat template, normalized to a dict with:
      - input_ids: (B, T) LongTensor
      - attention_mask: (B, T) LongTensor
    Works across HF versions that may return a Tensor or a BatchEncoding.
    """
    msgs_list = [
        [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": f"{q}\n\n{INSTRUCTION_SUFFIX}"},
        ]
        for q in questions
    ]

    enc = tokenizer.apply_chat_template(
        msgs_list,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    )

    # Normalize: some versions return a Tensor; some return a dict/BatchEncoding.
    if isinstance(enc, dict):
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
    elif isinstance(enc, torch.Tensor):
        input_ids = enc
        # Build an attention mask: non-pad tokens = 1
        pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        attention_mask = (input_ids != pad_id).long()
    else:
        # Final fallback: wrap single example
        input_ids = torch.as_tensor(enc, dtype=torch.long)
        pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        attention_mask = (input_ids != pad_id).long()

    # Ensure 2D (B, T)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

    return {"input_ids": input_ids, "attention_mask": attention_mask}
