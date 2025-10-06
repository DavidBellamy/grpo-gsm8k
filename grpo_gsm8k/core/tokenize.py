import torch
from transformers import PreTrainedTokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    """
    Tokenize prompt and output strings separately, concatenate them, then
    return left-shifted input_ids and right-shifted labels along with a
    response_mask that is 1 where labels correspond to response tokens and 0
    elsewhere (prompt or padding).

    Returns:
        {
            "input_ids": LongTensor (batch_size, max_len - 1),
            "labels": LongTensor (batch_size, max_len - 1),
            "response_mask": LongTensor (batch_size, max_len - 1) in {0,1}
        }
    """
    assert len(prompt_strs) == len(output_strs), "prompt_strs and output_strs must have same length"
    batch_size = len(prompt_strs)

    # Choose a padding id that exists
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # Tokenize each part without adding special tokens
    prompt_tok = [tokenizer.encode(p or "", add_special_tokens=False) for p in prompt_strs]
    output_tok = [tokenizer.encode(o or "", add_special_tokens=False) for o in output_strs]

    # Concatenate prompt + output per example
    full_ids = [p + o for p, o in zip(prompt_tok, output_tok)]
    full_lens = [len(ids) for ids in full_ids]
    max_len = max(full_lens) if full_lens else 0

    # Inputs/labels are shifted, thus length is (max_len - 1)
    target_len = max(max_len - 1, 0)

    input_ids = torch.full((batch_size, target_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, target_len), pad_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, target_len), dtype=torch.long)

    for i, (p_ids, o_ids, ids) in enumerate(zip(prompt_tok, output_tok, full_ids)):
        if len(ids) <= 1:
            # Nothing to shift; leave row as padding/zeros
            continue

        # Shifted sequences
        in_row = ids[:-1]
        lab_row = ids[1:]
        n = len(in_row)

        input_ids[i, :n] = torch.tensor(in_row, dtype=torch.long)
        labels[i, :n] = torch.tensor(lab_row, dtype=torch.long)

        # Mask 1s for labels that correspond to output tokens.
        # Label position j corresponds to token at ids[j+1].
        # Output tokens start at index len(p_ids), so j >= len(p_ids) - 1.
        start = max(len(p_ids) - 1, 0)
        if start < n:
            response_mask[i, start:n] = 1

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
