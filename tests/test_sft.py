from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors.torch import load_file

import grpo_gsm8k.sft as sft


class FakeConfig:
    pad_token_id = 0
    eos_token_id = 0

    def to_json_string(self) -> str:
        return json.dumps({"architectures": ["Fake"]})


def test_sft_microbatch_train_step_basic() -> None:
    # Two sequences, three tokens each; mask only the last two as response.
    logp = torch.tensor(
        [[-0.1, -0.2, -1.5], [-0.3, -0.4, -0.5]], dtype=torch.float32, requires_grad=True
    )
    mask = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.long)
    grad_acc = 2
    normalize_constant = float(mask.sum().item())  # average per response token

    loss, meta = sft.sft_microbatch_train_step(
        policy_log_probs=logp,
        response_mask=mask,
        normalize_constant=normalize_constant,
    )
    # Expected NLL = -mean(logp on response tokens)
    expected_mean_lp = (-0.2 - 1.5 - 0.4 - 0.5) / 4.0
    expected_nll = -expected_mean_lp
    expected_loss = expected_nll / grad_acc

    assert torch.is_tensor(loss) and loss.ndim == 0
    assert pytest.approx(loss.item(), rel=1e-5) == expected_loss
    assert pytest.approx(meta["mean_log_prob_response"].item(), rel=1e-5) == -expected_nll
    assert pytest.approx(meta["mean_nll_response"].item(), rel=1e-5) == expected_nll


def test_sft_microbatch_train_step_shape_and_args_errors() -> None:
    logp = torch.zeros(2, 3)
    mask = torch.zeros(2, 2, dtype=torch.long)
    with pytest.raises(ValueError):
        sft.sft_microbatch_train_step(logp, mask)

    with pytest.raises(ValueError):
        sft.sft_microbatch_train_step(torch.zeros(2, 2), torch.zeros(2, 2), 0.0)


def test_ensure_pad_token_sets_from_eos() -> None:
    class TK:
        pad_token_id: int | None
        pad_token: str | None
        eos_token: str | None
        eos_token_id: int | None

        def __init__(self) -> None:
            self.pad_token_id = None
            self.pad_token = None
            self.eos_token = "</s>"
            self.eos_token_id = 2

        def add_special_tokens(self, _x: dict[str, str]) -> None:
            raise AssertionError("should not be called when eos_token exists")

    tk = TK()
    sft._ensure_pad_token(cast(Any, tk))
    assert tk.pad_token == "</s>"


def test_ensure_pad_token_adds_new_when_no_eos() -> None:
    class TK:
        pad_token_id: int | None
        pad_token: str | None
        eos_token: str | None
        eos_token_id: int | None
        _added: bool

        def __init__(self) -> None:
            self.pad_token_id = None
            self.pad_token = None
            self.eos_token = None
            self.eos_token_id = None
            self._added = False

        def add_special_tokens(self, x: dict[str, str]) -> None:
            self._added = True
            assert x == {"pad_token": "<|pad|>"}
            self.pad_token = x["pad_token"]

    tk = TK()
    sft._ensure_pad_token(cast(Any, tk))
    assert tk._added is True
    assert tk.pad_token == "<|pad|>"


def test_resolve_resume_path_picks_latest_step(tmp_path: Path) -> None:
    root = tmp_path / "ckpts"
    (root / "step_2").mkdir(parents=True)
    (root / "step_10").mkdir(parents=True)
    (root / "misc").mkdir(parents=True)

    resolved = sft._resolve_resume_path(root)
    assert resolved.name == "step_10"


def test_save_policy_checkpoint_for_vllm(tmp_path: Path) -> None:
    class FakePolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self.config = FakeConfig()

    policy = FakePolicy()
    ckpt_dir = sft.save_policy_checkpoint_for_vllm(
        policy, step=3, out_root=tmp_path, base="unit", dtype=torch.bfloat16
    )
    assert ckpt_dir.exists()
    assert (ckpt_dir / "config.json").exists()
    st_path = ckpt_dir / "pytorch_model.safetensors"
    assert st_path.exists()
    assert (ckpt_dir / "READY").exists()
    tensors = load_file(str(st_path))
    # Parameter key present
    assert any(k.startswith("lin") for k in tensors.keys())
    # Dtype cast check (bfloat16)
    for t in tensors.values():
        assert t.dtype == torch.bfloat16
        break


def test_vllm_hot_reload_from_dir(tmp_path: Path) -> None:
    # Minimal directory to point at
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "READY").write_text("", encoding="utf-8")

    class EngineCore:
        def sleep(self) -> None:
            pass

        def wake_up(self) -> None:
            pass

    class EngineWrap:
        def __init__(self) -> None:
            self.engine_core = EngineCore()

    class FakeLLM:
        def __init__(self) -> None:
            self.engine = EngineWrap()
            self.calls: list[tuple[str, Any]] = []

        def collective_rpc(self, name: str, args: Any = ()) -> None:
            self.calls.append((name, args))

    llm = FakeLLM()
    sft._vllm_hot_reload_from_dir(llm, tmp_path)
    assert [c[0] for c in llm.calls] == ["update_config", "reload_weights"]
