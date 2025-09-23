import math
from typing import Any

import pytest
import torch

from grpo_gsm8k.log_generations import log_generations


class DummyTokenizer:
    """
    Minimal tokenizer stub.

    - __call__(prompts) expects each prompt to be a string representing an integer
      length (e.g., "2"). It returns padded input_ids and attention_mask accordingly.
    - decode(ids, skip_special_tokens=True) maps nonzero ids to tokens "t<ID>" and
      skips zeros if skip_special_tokens is True.
    """

    pad_id: int = 0

    def __call__(
        self,
        prompts: list[str],
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        lens: list[int] = [int(p) for p in prompts]
        max_len: int = max(lens) if lens else 0
        B: int = len(lens)
        input_ids: torch.Tensor = torch.zeros((B, max_len), dtype=torch.long)
        attention_mask: torch.Tensor = torch.zeros((B, max_len), dtype=torch.long)
        for i, L in enumerate(lens):
            if L > 0:
                # Just fill with 1..L for determinism (values don't matter)
                input_ids[i, :L] = torch.arange(1, L + 1, dtype=torch.long)
                attention_mask[i, :L] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        toks: list[str] = []
        for i in ids:
            if skip_special_tokens and i == self.pad_id:
                continue
            toks.append(f"t{i}")
        return " ".join(toks)


class GenOut:
    def __init__(self, sequences: torch.Tensor, scores: list[torch.Tensor]) -> None:
        self.sequences = sequences
        self.scores = scores


class DummyModel(torch.nn.Module):
    """
    Minimal generate()-only model.

    - generate returns:
        sequences: (B, prompt_len + T_gen) with generated tokens appended and
                   zero-padding for prompts shorter than the batch max.
        scores: list length T_gen; each score is a (B, V) tensor of logits.
                We use zeros to yield uniform distributions (entropy=log(V)).
    """

    def __init__(self, vocab_size: int, gen_len: int = 3, start_token: int = 100) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        self.gen_len: int = gen_len
        self.start_token: int = start_token

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> GenOut:
        B, T_prompt_max = input_ids.shape
        device = input_ids.device
        # Determine actual prompt lengths from attention_mask
        if attention_mask is None:
            lens = [T_prompt_max] * B
        else:
            lens = attention_mask.sum(dim=1).tolist()

        # Build sequences with prompt + generated tokens
        total_len = T_prompt_max + self.gen_len
        sequences = torch.zeros((B, total_len), dtype=torch.long, device=device)
        sequences[:, :T_prompt_max] = input_ids  # include padded prompts
        for b, L in enumerate(lens):
            gen_tokens = torch.arange(
                self.start_token + b * 10,
                self.start_token + b * 10 + self.gen_len,
                dtype=torch.long,
                device=device,
            )
            sequences[b, L : L + self.gen_len] = gen_tokens  # append right after the true prompt

        # Uniform logits -> entropy = log(V)
        scores = [
            torch.zeros((B, self.vocab_size), dtype=torch.float32, device=device)
            for _ in range(self.gen_len)
        ]
        return GenOut(sequences=sequences, scores=scores)


def test_log_generations_basic_and_stats() -> None:
    torch.manual_seed(0)
    tokenizer = DummyTokenizer()
    # Two prompts with different prompt lengths
    prompts: list[str] = ["2", "1"]
    # Model generates 3 tokens for each example
    V: int = 5
    T_gen: int = 3
    model = DummyModel(vocab_size=V, gen_len=T_gen, start_token=100)

    # References: first matches exactly, second mismatches
    expected_resp0 = "t100 t101 t102"
    expected_resp1 = "t110 t111 t112"
    references: list[str] = [expected_resp0, "not-a-match"]

    out = log_generations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        references=references,
        max_new_tokens=T_gen,
        temperature=0.0,
    )

    # Responses decoded from generated tokens
    assert out["responses"] == [expected_resp0, expected_resp1]

    # Entropy: uniform logits -> log(V)
    expected_entropy = math.log(V)
    assert pytest.approx(out["avg_token_entropy"], rel=1e-6) == expected_entropy
    for ex in out["per_example"]:
        assert pytest.approx(ex["mean_token_entropy"], rel=1e-6) == expected_entropy

    # Lengths are the number of generated tokens (fixed by our dummy: T_gen)
    assert pytest.approx(out["avg_response_length"], rel=1e-6) == float(T_gen)

    # Correctness via fallback exact match:
    rewards: list[dict[str, Any]] = out["rewards"]
    assert rewards[0].get("is_correct", False) is True
    assert rewards[1].get("is_correct", False) is False

    # Length stats by correctness
    assert out["avg_response_length_correct"] == float(T_gen)
    assert out["avg_response_length_incorrect"] == float(T_gen)

    # Per-example structure
    assert len(out["per_example"]) == 2
    assert out["per_example"][0]["prompt"] == prompts[0]
    assert out["per_example"][0]["response"] == expected_resp0
    assert out["per_example"][0]["reference"] == references[0]
    assert out["per_example"][0]["length"] == T_gen


def test_log_generations_with_custom_reward_fn() -> None:
    tokenizer = DummyTokenizer()
    prompts: list[str] = ["1"]
    model = DummyModel(vocab_size=4, gen_len=2, start_token=200)

    # Mark responses correct iff they contain token "t200 t201"
    def reward_fn(response: str | None) -> dict[str, Any]:
        if response is None:
            return {"is_correct": False, "reward": 0.0}
        return {"is_correct": response.strip() == "t200 t201", "reward": 1.0}

    out = log_generations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        reward_fn=reward_fn,
        max_new_tokens=2,
        temperature=0.0,
    )

    assert out["rewards"][0]["is_correct"] is True
    assert out["rewards"][0]["reward"] == 1.0
    assert out["avg_response_length_correct"] == 2.0
    assert out["avg_response_length_incorrect"] is None


def test_raises_on_reference_length_mismatch() -> None:
    tokenizer = DummyTokenizer()
    model = DummyModel(vocab_size=3, gen_len=1)
    with pytest.raises(ValueError):
        _ = log_generations(
            model=model,
            tokenizer=tokenizer,
            prompts=["2", "1"],
            references=["only-one"],
        )
