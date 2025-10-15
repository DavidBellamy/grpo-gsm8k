from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import wandb
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from grpo_gsm8k.data.prompts import render_batch
from grpo_gsm8k.evaluation.ci import analyze_eval_variance
from grpo_gsm8k.evaluation.reward_fn import reward_from_text
from grpo_gsm8k.utils.repro import SEED, seed_everything

logger = logging.getLogger(__name__)


def load_jsonl(p: str) -> list[dict]:
    return [json.loads(line) for line in Path(p).open()]


def single_eval_run(
    model_id: str,
    eval_path: str,
    limit: int | None,
    max_new_tokens: int,
    tp_size: int,
    gpu_mem_util: float,
    k_shot: int,
    run_idx: int = 0,
) -> tuple[float, list[dict]]:
    """Run a single evaluation and return pass@1 score and detailed results."""
    # Use different seed for each run to get true variance
    run_seed = SEED + run_idx
    seed_everything(run_seed, deterministic=False)

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Load eval set
    data = load_jsonl(eval_path)
    if limit:
        data = data[:limit]

    questions = [d["question"] for d in data]

    # For k-shot, load examples from training set
    few_shot_examples = None
    if k_shot > 0:
        train_data = load_jsonl("artifacts/r1_sft_pairs.jsonl")
        few_shot_examples = train_data[:k_shot]

    prompts = render_batch(
        tok, questions, add_generation_prompt=True, few_shot_examples=few_shot_examples
    )

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem_util,
        enable_prefix_caching=True,
        max_model_len=4096,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
        seed=run_seed,
    )

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        seed=run_seed,
    )

    outputs = llm.generate(prompts, sp, use_tqdm=True)
    texts = [o.outputs[0].text for o in outputs]

    rewards = [reward_from_text(o, d["answer"], "boxed") for o, d in zip(texts, data)]
    pass_at_1 = sum(rewards) / len(rewards)

    # Build detailed results for this run
    results = []
    for d, o, r in zip(data, texts, rewards):
        results.append(
            {
                "id": d["id"],
                "question": d["question"],
                "output": o,
                "reward": r,
                "gold": d["answer"],
            }
        )

    return pass_at_1, results


def main(
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    eval_path: str = "artifacts/gsm8k/test.jsonl",
    limit: int | None = None,
    max_new_tokens: int = 1024,
    tp_size: int = 1,
    wandb_project: str = "grpo-gsm8k",
    gpu_mem_util: float = 0.92,
    run_name: str | None = None,
    k_shot: int = 8,
    ci_reps: int = 1,
) -> dict[str, float]:
    """Run evaluation with optional confidence interval analysis."""
    if wandb_project:
        run = wandb.init(
            project=wandb_project,
            name=run_name,
            config=dict(model_id=model_id, tp_size=tp_size, k_shot=k_shot, ci_reps=ci_reps),
        )
    else:
        run = None

    scores = []
    all_results = []

    logger.info(f"Running {ci_reps} evaluation(s) for {model_id}")

    for i in range(ci_reps):
        if ci_reps > 1:
            logger.info(f"Evaluation run {i+1}/{ci_reps}")

        score, results = single_eval_run(
            model_id=model_id,
            eval_path=eval_path,
            limit=limit,
            max_new_tokens=max_new_tokens,
            tp_size=tp_size,
            gpu_mem_util=gpu_mem_util,
            k_shot=k_shot,
            run_idx=i,
        )

        scores.append(score)
        all_results.extend(results)
        logger.info(f"Run {i+1} pass@1: {score:.3f}")

    # Analyze variance if multiple runs
    if ci_reps > 1:
        ci_analysis = analyze_eval_variance(scores)
        logger.info(f"Final results: {ci_analysis}")
    else:
        ci_analysis = {"mean": scores[0], "n_runs": 1}

    # Save results
    Path("artifacts/baselines").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if ci_reps > 1:
        # Save CI analysis
        ci_path = Path(f"artifacts/baselines/{model_id.replace('/', '_')}_ci_{ts}.json")
        ci_path.write_text(json.dumps(ci_analysis, indent=2), encoding="utf-8")
        logger.info(f"Wrote CI analysis to {ci_path}")

    # Save detailed results from all runs
    out_path = Path(f"artifacts/baselines/{model_id.replace('/', '_')}_eval_{ts}.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Wrote detailed results to {out_path}")

    # Log to W&B
    if run is not None:
        final_score = ci_analysis["mean"]
        wandb.log({"metrics/pass@1": final_score, "n": len(all_results)})

        if ci_reps > 1:
            # Log CI info
            wandb.log(
                {
                    "metrics/pass@1_std": ci_analysis.get("std", 0),
                    "metrics/pass@1_ci_lower": ci_analysis.get("ci_95", [0, 0])[0],
                    "metrics/pass@1_ci_upper": ci_analysis.get("ci_95", [0, 0])[1],
                }
            )

        art = wandb.Artifact("vllm-eval", type="eval")
        art.add_file(str(out_path))
        if ci_reps > 1:
            art.add_file(str(ci_path))
        run.log_artifact(art)
        run.finish()

    return ci_analysis


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--eval_path", type=str, default="artifacts/gsm8k/test.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--gpu_mem_util", type=float, default=0.92)
    p.add_argument(
        "--k_shot", type=int, default=8, help="Number of few-shot examples (0 for zero-shot)"
    )
    p.add_argument(
        "--ci_reps",
        type=int,
        default=1,
        help="Number of evaluation runs for confidence intervals (1 for single run)",
    )
    args = p.parse_args()
    main(**vars(args))
