"""
RL fine-tuning with GRPO using AdaptiveTutorEnvironment as the reward source.

Install (Colab):
    !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install trl datasets huggingface_hub openenv

Usage:
    python rl_train.py \
        --sft_model Bumpeet/qwen2.5-1.5b-adaptive-tutor-sft \
        --output runs/rl-v1 \
        --hub_repo Bumpeet/qwen2.5-1.5b-adaptive-tutor-rl

Environment variables:
    HF_TOKEN        — Hugging Face token (for Hub push and model download)
    OPENAI_API_KEY  — optional; enables LLM-backed student simulation
                      (falls back to analytic sigmoid formula if not set)
"""

import argparse
import json
import os
import re
from typing import Any, List

from datasets import Dataset
from huggingface_hub import HfApi
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOTrainer, GRPOConfig

# ---------------------------------------------------------------------------
# Local imports — requires: pip install -e . (from the repo root)
# ---------------------------------------------------------------------------

from adaptive_tutor_env.server.tutor_environment import AdaptiveTutorEnvironment
from adaptive_tutor_env.models import DIFFICULTY_LABELS
from openenv.core.env_server.mcp_types import CallToolAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SEQ_LENGTH = 2048

_DIFFICULTY_NAME_TO_INT = {"easy": 1, "medium": 2, "hard": 3}

EPISODE_FLOWS_PATH = os.path.join(os.path.dirname(__file__), "data", "episode_flows.json")

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
# Dataset — loaded from episode_flows.json
# ---------------------------------------------------------------------------

def make_prompt_dataset(flows_path: str = EPISODE_FLOWS_PATH) -> Dataset:
    with open(flows_path, encoding="utf-8") as f:
        flows = json.load(f)

    rows = []
    for flow in flows:
        concept    = flow["concept"]
        mastery    = float(flow["mastery"])
        raw_diff   = flow["target_difficulty"]
        difficulty = _DIFFICULTY_NAME_TO_INT.get(str(raw_diff).lower(), int(raw_diff))
        label      = DIFFICULTY_LABELS[difficulty]
        past_wrong = flow.get("past_wrong_questions") or []

        history = [
            {"question": tag, "correct": False, "step": i + 1}
            for i, tag in enumerate(past_wrong)
        ]

        prompt = SYSTEM_PROMPT.format(
            concept=concept,
            mastery=mastery,
            mistakes="; ".join(past_wrong) if past_wrong else "none",
            difficulty=label,
        )
        rows.append({
            "prompt": prompt,
            "concept": concept,
            "mastery": mastery,
            "difficulty": difficulty,
            "history": history,
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
      2. Reset the environment with the row's concept/mastery/difficulty/history.
      3. Call submit_teaching_action via env.step().
      4. Return the scalar reward from the Observation.

    Returns 0.0 for any parse failure or environment error.
    """
    concepts: List[str]    = kwargs["concept"]
    masteries: List[float] = kwargs["mastery"]
    difficulties: List[int] = kwargs["difficulty"]
    histories: List[List]  = kwargs["history"]

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
        history    = histories[i] if histories[i] else []

        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(m.group()) if m else json.loads(text)
            explanation = str(parsed.get("explanation", "")).strip()
            question    = str(parsed.get("question", "")).strip()
            if not explanation or not question:
                rewards.append(0.0)
                continue

            _env.reset(
                seed=i,
                concept=concept,
                concept_mastery={concept: mastery},
                targeted_difficulty=difficulty,
                history=history,
            )

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
        dtype=None,
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

    dataset = make_prompt_dataset(args.flows)
    print(f"Dataset size: {len(dataset)} prompts loaded from {args.flows}")

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=GRPOConfig(
            output_dir=args.output,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            bf16=True,
            logging_steps=5,
            save_strategy="epoch",
            num_generations=4,
            max_new_tokens=600,
            temperature=0.8,
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
    parser.add_argument("--sft_model", default="Bumpeet/qwen2.5-1.5b-adaptive-tutor-sft",
                        help="HF repo or local path of the SFT checkpoint to start from")
    parser.add_argument("--output",    default="runs/rl-v1")
    parser.add_argument("--flows",     default=EPISODE_FLOWS_PATH,
                        help="Path to episode_flows.json")
    parser.add_argument("--merge",     action="store_true",
                        help="merge LoRA into base after training")
    parser.add_argument("--hub_repo",  default=None,
                        help="HF repo to push to, e.g. Bumpeet/qwen2.5-1.5b-adaptive-tutor-rl")
    main(parser.parse_args())
