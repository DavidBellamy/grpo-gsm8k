# Post-Training Small Math Language Models

By [David Bellamy](davidbellamy.github.io).

[![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-view%20project-yellow)](https://wandb.ai/username/projectname)

A comparison of the efficacy of popular post-training algorithms for improving math abilities of language models. Specifically, supervised fine-tuning (SFT), REINFORCE (with and without a baseline) and DeepSeek's Group Relative Policy Optimization (GRPO) with Qwen2.5-Math-1.5B on GSM8k problems. Only two GPUs are needed – one for inference, one for training. lm-eval is used to check for capability regressions. Bootstrap confidence intervals are computed for key comparisons.

For independent engineers/researchers interested in post-training without torch wrappers, this repo can get you started on a personal budget. See [RunPod Instructions](#runpod-instructions).

---

## Quickstart

This project can run on any GPU cloud or on-prem machines with Docker and NVIDIA drivers. For RunPod instructions specifically, see [RunPod Instructions](#runpod-instructions).

Prerequisites: sanity check your Docker, NVIDIA driver and NVIDIA Container Toolkit are installed:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

1. Clone this repo.
2. Copy env template and add your DeepSeek API key if you intend on generating DeepSeek R1 tokens for SFT.
```bash
cp .env.example .env
```
3. Start an interactive container:
```bash
docker compose run --rm --gpus all grpo
```
(Or select specific GPUs: `docker compose run --rm --gpus "device=0,1" grpo`.)

4. Verify the container has GPU access with `nvidia-smi`.
5. For developers: install dev deps `uv sync --dev` and confirm that all tests pass: `pytest`.
6. Run project commands inside the container (see [Project Commands Summary](#project-commands-summary)).

---

## Project Commands Summary

Below are the main project commands. Each is described in more detail in subsequent sections. Run each command from the project's root directory.

```python
python cli.py command=data # downloads & splits gsm8k
python cli.py command=traces # generates R1 completions to gsm8k-train
python cli.py command=preprocess # preps R1 completions & GSM8k val data
python cli.py command=sft # trains qwen2.5-math-1.5b (base) with SFT
python cli.py command=policy_gradient # trains with grpo_clip loss
python cli.py command=eval # evals qwen2.5-math-1.5b (base) on eval suite
```

`cli.py` is the orchestrator. All configuration is handled by Hydra. Config files are in [conf/](conf/). [config.yaml](conf/config.yaml) is the primary config. You can modify the parameters for each command by making your own yaml file(s) in the sub-directories of [conf/](conf/).

## Command: Data

This command downloads GSM8k train and test splits from HuggingFace datasets. We reserve 512 problems from the training set for the validation split. Splits are saved under [artifacts/gsm8k/](artifacts/gsm8k/).

## Command: Traces

This command generates DeepSeek R1 chain of thought (aka reasoning traces) via DeepSeek's API and formats them for SFT. By default, it generates 1 R1 CoT (max 2048 tokens) per prompt in the project's GSM8k train split. Note that this doesn't include the validation split. The formatting step cleans the reasoning body by dropping lines that look like meta/instruction echoes (e.g., "I will output ANSWER...", "show your reasoning...", etc.) via a big regex and normalizes the final numeric response.

## Command: Preprocess

This command tokenizes, pads and batches the R1 SFT data and applies Qwen's CoT prompt template to the GSM8k training and validation splits in order to accelerate inference during training and validation.

## Command: SFT

This command trains Qwen2.5-Math-1.5B (base) with SFT on R1 reasoning traces. It uses a dual GPU setup with one running vLLM for validation and one running torch. The trainer does forward/backward passes, gradient accumulation and optimizer steps. The vLLM worker evaluates model checkpoints on this project's GSM8k validation split.

Both GPU processes communicate to each other via two multiprocessing queues (OS IPC). After `eval_every` steps, the trainer GPU saves a model checkpoint in the RAM-backed temporary filesystem at `/dev/shm/` (very fast) and enqueues a job payload for the vLLM worker. The vLLM worker is a blocking consumer of jobs and once one arrives it hot-reloads the weights, generates model completions, then enqueues the results in the results queue. The trainer is a non-blocking consumer of the results queue and so it drains results opportunistically just before each optimizer step so that training never stalls. For each result, the trainer logs metrics to W&B.

## Command: Policy Gradient

This command trains Qwen2.5-Math-1.5B (base) with the chosen variant of policy gradient method (REINFORCE with no baseline, REINFORCE with group-mean reward baseline, or GRPO with clipped loss) on problems in the GSM8k train split. It uses a dual GPU setup with one running vLLM for inference and one running torch for training. The inference GPU generates tokens (aka rollouts) from the model for each prompt in the GSM8k train split, which are rewarded and used for gradient descent on the trainer GPU.

Just as with SFT, after `eval_every` steps, the trainer GPU sends a model checkpoint to the inference GPU for evaluation on the GSM8k validation split. The results are returned to the trainer, which logs metrics to W&B.

## Command: Eval

This command runs the eval suite, which includes a custom Qwen-friendly GSM8k evaluation with k-shot CoT prompting as well as lm-eval tasks: hendrycks_math500, mmlu, arc_challenge, hellaswag, winogrande, truthfulqa_mc2, and wikitext.

By default, this command evaluates Qwen's Qwen2.5-Math-1.5B base model on the entire eval suite. A local model checkpoint can be evaluated by overriding the `model_path` field in [conf/eval/default.yaml](conf/eval/default.yaml). By default, GSM8k evaluation is done with k=8 shot CoT prompting and lm-eval tasks use k=4 shot. Both values of k can be modified in the config. If you only wish to run the GSM8k eval, override the `eval_suites` field with 'gsm8k'. If you only wish to run the lm-eval suite, override `eval_suites` with 'lm_eval'. You can also run just specific lm-eval tasks by either changing the `lm_eval_tasks` field or overriding the `eval_suites` field with the specific name of an lm-eval task.

The GSM8k eval uses data parallel to distribute the job across available GPUs. Set `num_shards` in [conf/eval/default.yaml](conf/eval/default.yaml) to specify the number of available GPUs. lm-eval is distributed with task parallelism.

## Manual lm-eval Evaluation

Spinning up a vLLM instance is slow so it is quicker to do development with a persistent vLLM server. For that, you can do the following.

Launch a vllm server for a HuggingFace model:
```bash
vllm serve qwen/qwen2.5-math-1.5b --host 127.0.0.1 --port 8000 --dtype auto
```

Or for your own model checkpoint:
```bash
vllm serve path/to/your/model   --host 127.0.0.1 --port 8000 --dtype auto   --served-model-name {your_model_name}
```

Then run lm-eval against that server, either with a public model:
```bash
python -m lm_eval \
--model local-completions \
--model_args "model=qwen/qwen2.5-math-1.5b,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=10,tokenized_requests=True,tokenizer_backend=huggingface,max_length=4096" \
--tasks hendrycks_math500,mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,wikitext \
--num_fewshot 4 \
--batch_size 8 \
--gen_kwargs '{"temperature":0,"do_sample":false,"max_new_tokens":2048}' \
--output_path ./artifacts/lm_eval_out/qwen_vllm_math_4shot_bs8
```

or your own checkpoint (note: enter `your_model_name` in model_args):

```bash
export OPENAI_API_KEY=EMPTY
export TOKENIZER_ABS="path/to/your/checkpoints/tokenizer"

python -m lm_eval \
--model local-completions \
--model_args "base_url=http://127.0.0.1:8000/v1/completions,model={your_model_name},num_concurrent=10,tokenized_requests=False,tokenizer=${TOKENIZER_ABS},tokenizer_backend=huggingface,max_length=4096" \
--tasks hendrycks_math500,mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,wikitext \
--num_fewshot 4 \
--batch_size 8 \
--gen_kwargs '{"temperature":0,"do_sample":false,"max_new_tokens":2048}' \
--output_path ./lm_eval_out/your_model_4shot_bs8
```

## RunPod Instructions

If you are using RunPod, you can use this project's public RunPod template called `rainbow_unicorn`.

1. **Network Volume**: rent it in the same region as the GPUs you will rent.
2. **Launch a GPU pod** (H100 SXM or similar). Set the pod template to `rainbow_unicorn`.

On the first pod you deploy, configure your git info:
```bash
  git config -f /workspace/dotfiles/gitconfig user.name "your_name"
  git config -f /workspace/dotfiles/gitconfig user.email "123456+youruser@users.noreply.github.com"
  git config -f /workspace/dotfiles/gitconfig user.useConfigOnly true
```

## Dev Tooling Summary

| Task | Command |
|------|---------|
| Run tests | pytest |
| Lint | ruff check . |
| Format | ruff format . |
| Type-check | mypy grpo_gsm8k |

Note: ruff and mypy are both run automatically as pre-commit hooks.

---

## License

MIT — see [LICENSE](LICENSE).
