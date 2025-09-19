import argparse
import json
from datetime import datetime
from pathlib import Path

import wandb
from transformers import AutoTokenizer
from vllm import LLM

from grpo_gsm8k.prompts import render_batch
from grpo_gsm8k.repro import SEED, seed_everything, vllm_sampling_params, wandb_run_init
from grpo_gsm8k.reward_fn import reward_from_text


def load_jsonl(p: str) -> list[dict]:
    return [json.loads(line) for line in Path(p).open()]


def main(
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    eval_path: str = "artifacts/gsm8k/test_eval.jsonl",
    limit: int | None = None,
    max_new_tokens: int = 1024,
    tp_size: int = 1,
    wandb_project: str = "grpo-gsm8k",
    gpu_mem_util: float = 0.92,
    run_name: str | None = None,
) -> None:
    seed_everything(SEED, deterministic=False)
    run = wandb_run_init(wandb_project, run_name, dict(model_id=model_id, tp_size=tp_size))

    tok = AutoTokenizer.from_pretrained(model_id)

    # Load eval set (optionally limit), then extract questions
    data = load_jsonl(eval_path)
    if limit:
        data = data[:limit]
    questions = [d["question"] for d in data]

    prompts = render_batch(tok, questions, add_generation_prompt=True)

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem_util,
        enable_prefix_caching=True,  # reuse shared system prompt or suffix
        max_model_len=4096,
        tensor_parallel_size=tp_size,
    )

    # Deterministic greedy: temp=0, top_p=1. Don’t rely on defaults.
    sp = vllm_sampling_params(max_tokens=max_new_tokens, seed=SEED)

    outputs = llm.generate(prompts, sp, use_tqdm=True)
    texts = [o.outputs[0].text for o in outputs]

    rewards = [reward_from_text(o, d["answer"], "boxed") for o, d in zip(texts, data)]
    mean = sum(rewards) / len(rewards)
    print(f"vLLM pass@1: {mean:.3f} on n={len(rewards)}")

    Path("artifacts/baselines").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"artifacts/baselines/qwen25_7b_eval_vllm_{ts}.jsonl")
    with out_path.open("w") as f:
        for d, o, r in zip(data, texts, rewards):
            obj = {
                "id": d["id"],
                "question": d["question"],
                "output": o,
                "reward": r,
                "gold": d["answer"],
            }
            f.write(json.dumps(obj) + "\n")
    print("Wrote", out_path)

    wandb.log({"metrics/pass@1": mean, "n": len(rewards)})
    art = wandb.Artifact("vllm-eval", type="eval")
    art.add_file(str(out_path))
    run.log_artifact(art)
    run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--eval_path", type=str, default="artifacts/gsm8k/test_eval.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--gpu_mem_util", type=float, default=0.92)
    args = p.parse_args()
    main(**vars(args))
