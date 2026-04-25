"""
RL fine-tuning with GRPO using AdaptiveTutorEnvironment as the reward source.

Install (Colab):
    !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install trl datasets huggingface_hub openenv

Usage:
    python rl_train.py \
        --sft_model Bumpeet/qwen2.5-7b-adaptive-tutor-sft \
        --output runs/rl-v1 \
        --hub_repo Bumpeet/qwen2.5-7b-adaptive-tutor-rl

Environment variables:
    HF_TOKEN        — Hugging Face token (for Hub push and model download)
    OPENAI_API_KEY  — optional; enables LLM-backed student simulation
                      (falls back to analytic sigmoid formula if not set)
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List

from datasets import Dataset
from huggingface_hub import HfApi
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOTrainer, GRPOConfig

# ---------------------------------------------------------------------------
# Local imports — requires: pip install -e . (from the repo root)
# ---------------------------------------------------------------------------

from adaptive_tutor_env.server.tutor_environment import AdaptiveTutorEnvironment
from adaptive_tutor_env.server.student_model import DEFAULT_MASTERY
from adaptive_tutor_env.models import TASKS, TASK_DIFFICULTY, DIFFICULTY_LABELS
from openenv.core.env_server.mcp_types import CallToolAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SEQ_LENGTH = 2048
CONCEPTS = list(DEFAULT_MASTERY.keys())   # ["arrays", "stack", "trees", "backtracking", "dp"]

# Mirrors inference.py's TEACHING_SYSTEM_PROMPT so the SFT checkpoint is
# already conditioned on this exact format.
SYSTEM_PROMPT = (
    "You are an expert DSA tutor and problem setter.\n\n"
    "INPUT:\n"
    "- Concept: {concept}\n"
    "- Current mastery: {mastery:.2f} (0–1)\n"
    "- Previous mistakes: {mistakes}\n"
    "- Target difficulty: {difficulty}\n\n"
    "TASK:\n"
    "Generate both:\n"
    "1. Teaching material that will help the student improve.\n"
    "2. One follow-up question that directly tests the material you generated.\n\n"
    "OUTPUT RULES:\n"
    "- Return valid JSON only. No markdown fences.\n"
    "- No text before or after the JSON object.\n"
    "- Escape newlines inside strings as \\n.\n\n"
    'OUTPUT: {{"explanation": "...", "question": "..."}}'
)


# ---------------------------------------------------------------------------
# Dataset — one row per (concept × task) combination
# ---------------------------------------------------------------------------

def make_prompt_dataset() -> Dataset:
    rows = []
    for concept in CONCEPTS:
        for task in TASKS:
            difficulty = TASK_DIFFICULTY[task]
            label = DIFFICULTY_LABELS[difficulty]
            mastery = DEFAULT_MASTERY.get(concept, 0.3)
            prompt = SYSTEM_PROMPT.format(
                concept=concept,
                mastery=mastery,
                mistakes="none",
                difficulty=label,
            )
            rows.append({
                "prompt": prompt,
                "concept": concept,
                "mastery": mastery,
                "difficulty": difficulty,
                "task": task,
            })
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Reward function — called by GRPOTrainer with a batch of completions
# ---------------------------------------------------------------------------

# One shared env instance; reset() is called per example inside the batch.
_env = AdaptiveTutorEnvironment()


def reward_fn(completions: List[Any], prompts: List[Any], **kwargs) -> List[float]:
    """
    For each model completion:
      1. Parse JSON to extract explanation + question.
      2. Reset the environment with the row's concept/mastery/task.
      3. Call submit_teaching_action via env.step().
      4. Return the scalar reward from the Observation.

    Returns 0.0 for any parse failure or environment error.
    """
    concepts: List[str]   = kwargs["concept"]
    masteries: List[float] = kwargs["mastery"]
    difficulties: List[int] = kwargs["difficulty"]
    tasks: List[str]      = kwargs["task"]

    rewards = []

    for i, completion in enumerate(completions):
        # GRPOTrainer passes completions as list-of-messages or raw strings
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)

        concept    = concepts[i]
        mastery    = float(masteries[i])
        difficulty = int(difficulties[i])
        task       = tasks[i]

        try:
            # Parse JSON output from the model
            m = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(m.group()) if m else json.loads(text)
            explanation = str(parsed.get("explanation", "")).strip()
            question    = str(parsed.get("question", "")).strip()
            if not explanation or not question:
                rewards.append(0.0)
                continue

            # Reset env for this concept/task
            _env.reset(
                seed=i,
                task=task,
                concept=concept,
                concept_mastery={concept: mastery},
            )

            # Submit the teaching action — this triggers simulate_student,
            # update_mastery, and compute_reward inside the environment.
            obs = _env.step(
                CallToolAction(
                    tool_name="submit_teaching_action",
                    arguments={"question": question, "explanation": explanation},
                )
            )

            reward = float(obs.reward) if obs.reward is not None else 0.0
            rewards.append(reward)

        except Exception as exc:
            print(f"[reward_fn] error for example {i}: {exc}", flush=True)
            rewards.append(0.0)

    return rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.sft_model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,         # auto-detects bf16/fp16
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = make_prompt_dataset()
    print(f"Dataset size: {len(dataset)} prompts ({len(CONCEPTS)} concepts × {len(TASKS)} tasks)")

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=GRPOConfig(
            output_dir=args.output,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,         # lower than SFT — RL is sensitive
            bf16=True,
            logging_steps=5,
            save_strategy="epoch",
            num_generations=4,          # rollouts per prompt per GRPO step
            max_new_tokens=600,
            temperature=0.8,            # diversity for rollouts
            report_to="none",
        ),
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved RL adapters to {args.output}")

    upload_path = args.output

    if args.merge:
        model = FastLanguageModel.for_inference(model)
        merged = f"{args.output}-merged"
        model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")
        print(f"Merged model saved to {merged}")
        upload_path = merged

    if args.hub_repo:
        token = os.environ.get("HF_TOKEN")
        api = HfApi()
        api.create_repo(args.hub_repo, repo_type="model", exist_ok=True, token=token)
        api.upload_folder(
            folder_path=upload_path,
            repo_id=args.hub_repo,
            repo_type="model",
            token=token,
        )
        print(f"Pushed to https://huggingface.co/{args.hub_repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model", default="Bumpeet/qwen2.5-7b-adaptive-tutor-sft",
                        help="HF repo or local path of the SFT checkpoint to start from")
    parser.add_argument("--output",    default="runs/rl-v1")
    parser.add_argument("--merge",     action="store_true",
                        help="merge LoRA into base after training")
    parser.add_argument("--hub_repo",  default=None,
                        help="HF repo to push to, e.g. Bumpeet/qwen2.5-7b-adaptive-tutor-rl")
    main(parser.parse_args())
