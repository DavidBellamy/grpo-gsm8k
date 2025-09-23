import torch

from grpo_gsm8k.tokenize import tokenize_prompt_and_output


class DummyTokenizer:
    """Minimal tokenizer stub for tests (char-level IDs)."""

    def __init__(self, pad_token_id: int | None = 0, eos_token_id: int = 1):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        # Simple, deterministic char-level encoding offset by +10 to avoid 0/1
        return [ord(c) + 10 for c in (text or "")]


def test_tokenize_basic_concat_shift_and_mask() -> None:
    tok = DummyTokenizer(pad_token_id=0, eos_token_id=1)

    prompts = ["ab", "wxyz"]
    outputs = ["XYZ", ""]  # first has response, second has none

    result = tokenize_prompt_and_output(prompts, outputs, tok)

    # Shapes: max sequence len is 5 -> shifted to 4
    assert result["input_ids"].shape == (2, 4)
    assert result["labels"].shape == (2, 4)
    assert result["response_mask"].shape == (2, 4)

    # Helper to encode easily in test
    def enc(s: str) -> list[int]:
        return [ord(c) + 10 for c in s]

    # Example 1
    ids1 = enc("ab") + enc("XYZ")  # [a,b,X,Y,Z]
    expected_input_1 = torch.tensor(ids1[:-1], dtype=torch.long)  # [a,b,X,Y]
    expected_labels_1 = torch.tensor(ids1[1:], dtype=torch.long)  # [b,X,Y,Z]
    expected_mask_1 = torch.tensor([0, 1, 1, 1], dtype=torch.long)  # prompt then response positions

    assert torch.equal(result["input_ids"][0], expected_input_1)
    assert torch.equal(result["labels"][0], expected_labels_1)
    assert torch.equal(result["response_mask"][0], expected_mask_1)

    # Example 2 (no output). Sequence length 4 -> shift 3, last col is padding
    ids2 = enc("wxyz")
    expected_input_2 = torch.tensor(ids2[:-1] + [tok.pad_token_id], dtype=torch.long)  # [w,x,y,_]
    expected_labels_2 = torch.tensor(ids2[1:] + [tok.pad_token_id], dtype=torch.long)  # [x,y,z,_]
    expected_mask_2 = torch.tensor([0, 0, 0, 0], dtype=torch.long)

    assert torch.equal(result["input_ids"][1], expected_input_2)
    assert torch.equal(result["labels"][1], expected_labels_2)
    assert torch.equal(result["response_mask"][1], expected_mask_2)


def test_padding_falls_back_to_eos_when_no_pad_token() -> None:
    eos_id = 9
    tok = DummyTokenizer(pad_token_id=None, eos_token_id=eos_id)

    prompts = ["a", "a"]
    outputs = ["bcd", "b"]  # first longer to force padding on second row

    res = tokenize_prompt_and_output(prompts, outputs, tok)

    # max seq lengths: [4,2] -> target_len = 3
    assert res["input_ids"].shape == (2, 3)

    # Second row has only 1 shifted token; remaining should be padded with eos_id
    row2_inp = res["input_ids"][1].tolist()
    row2_lab = res["labels"][1].tolist()

    assert row2_inp[1:] == [eos_id, eos_id]
    assert row2_lab[1:] == [eos_id, eos_id]
    # Mask on the padded positions should be 0
    assert res["response_mask"][1].tolist()[1:] == [0, 0]


def test_response_mask_with_empty_prompt_all_output_tokens() -> None:
    tok = DummyTokenizer()
    prompts = [""]
    outputs = ["pq"]  # two tokens -> shifted length 1

    res = tokenize_prompt_and_output(prompts, outputs, tok)

    assert res["input_ids"].shape == (1, 1)
    assert res["labels"].shape == (1, 1)
    # With empty prompt, all labels correspond to output tokens
    assert res["response_mask"].tolist() == [[1]]
