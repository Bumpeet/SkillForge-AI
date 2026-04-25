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
import sys
import time
from typing import Any, List, Optional

import torch

from datasets import Dataset
from huggingface_hub import HfApi
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOTrainer, GRPOConfig

# ---------------------------------------------------------------------------
# Local imports — add repo root and server/ to sys.path for direct execution
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.path.join(_HERE, "server") not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, "server"))

from server.tutor_environment import AdaptiveTutorEnvironment
from models import DIFFICULTY_LABELS
from openenv.core.env_server.mcp_types import CallToolAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SEQ_LENGTH = 2048

_DIFFICULTY_NAME_TO_INT = {"easy": 1, "medium": 2, "hard": 3}

EPISODE_FLOWS_PATH = os.path.join(os.path.dirname(__file__), "data", "episode_flows.json")

# Keep OUTPUT RULES in sync with inference.TEACHING_SYSTEM_PROMPT (same JSON contract).
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
    "OUTPUT RULES (critical — reward parsing depends on this):\n"
    "- Return exactly one JSON object and nothing else. The first non-whitespace character MUST be '{' and the last non-whitespace character MUST be '}'.\n"
    "- No markdown, no code fences (no ```), no labels such as OUTPUT:, Example:, EXPLANATION:, or NOTE: outside the JSON.\n"
    "- The object MUST have exactly two keys, both non-empty strings: \"explanation\" (teaching material) and \"question\" (one follow-up). No other top-level keys.\n"
    "- Do not use placeholders: never output the literal ellipsis \"...\" or empty strings for either field. Do not echo template text or \"fill in the blank\" examples.\n"
    "- Do not output a second JSON object, JavaScript/Python/C++ samples, or extra '{' / '}' blocks outside that single object. Put any code or examples inside the two string values only, with valid JSON escaping.\n"
    "- Inside strings, escape newlines as \\n and internal double quotes as \\\". Do not paste raw multi-line JSON or unescaped control characters inside a string.\n"
    "- Use double quotes for all keys and string values.\n"
)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _sanitize_json(text: str) -> str:
    """Fix invalid escape sequences and literal control characters inside JSON strings."""
    result = []
    in_string = False
    escaped = False
    _valid_escapes = set('"\\\/bfnrtu')
    for ch in text:
        if escaped:
            escaped = False
            if ch not in _valid_escapes:
                result[-1] = "\\\\"
            result.append(ch)
        elif ch == "\\" and in_string:
            escaped = True
            result.append(ch)
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        elif in_string and ord(ch) < 0x20:
            result.append(f"\\u{ord(ch):04x}")
        else:
            result.append(ch)
    return "".join(result)


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "372f04",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(os.path.join(_HERE, "debug-372f04.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _reward_payload_score(d: dict) -> int:
    """Prefer non-empty explanation+question; reject placeholders and template stubs."""
    exp = str(d.get("explanation", "")).strip()
    q = str(d.get("question", "")).strip()
    if not exp or not q:
        return -1
    if exp in ("...", "…") and q in ("...", "…"):
        return -1
    if exp == "..." or q == "...":
        return -1
    return len(exp) + len(q)


def _try_raw_decode_reward_dict(stripped: str) -> Optional[dict]:
    decoder = json.JSONDecoder()
    if not stripped.startswith("{"):
        return None
    for body in (stripped, _sanitize_json(stripped)):
        try:
            obj, _ = decoder.raw_decode(body)
            if isinstance(obj, dict) and ("explanation" in obj or "question" in obj):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _fenced_json_blobs(text: str) -> List[str]:
    out = []
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        chunk = m.group(1).strip()
        if chunk:
            out.append(chunk)
    return out


def _collect_reward_dict_candidates(text: str) -> List[dict]:
    seen: set[tuple] = set()
    found: List[dict] = []

    def add(obj: dict) -> None:
        key = (str(obj.get("explanation", "")), str(obj.get("question", "")))
        if key in seen:
            return
        seen.add(key)
        found.append(obj)

    blobs: List[str] = []
    blobs.extend(_fenced_json_blobs(text))
    blobs.append(text)

    for blob in blobs:
        stripped = blob.lstrip()
        if stripped.startswith("{"):
            d = _try_raw_decode_reward_dict(stripped)
            if d:
                add(d)
        for idx, ch in enumerate(blob):
            if ch != "{":
                continue
            d = _try_raw_decode_reward_dict(blob[idx:].lstrip())
            if d:
                add(d)
    return found


_REWARD_KV_PATTERN = re.compile(
    r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"question"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)


def _regex_extract_reward_kv(text: str) -> Optional[dict]:
    m = _REWARD_KV_PATTERN.search(text)
    if not m:
        return None
    try:
        exp = json.loads('"' + m.group(1) + '"')
        q = json.loads('"' + m.group(2) + '"')
    except json.JSONDecodeError:
        return None
    return {"explanation": str(exp), "question": str(q)}


def _extract_reward_json_impl(text: str) -> tuple[dict, dict]:
    """
    Returns (payload, meta) where meta supports debug logs (candidate counts, path taken).
    """
    candidates = _collect_reward_dict_candidates(text)
    scored = [(d, _reward_payload_score(d)) for d in candidates]
    viable = [(d, s) for d, s in scored if s >= 0]
    meta: dict = {
        "candidate_count": len(candidates),
        "viable_count": len(viable),
        "max_raw_score": max((s for _, s in scored), default=-1),
    }
    if viable:
        best_d, best_s = max(viable, key=lambda x: x[1])
        meta["chosen_score"] = best_s
        meta["via"] = "json"
        return best_d, meta
    fallback = _regex_extract_reward_kv(text)
    if fallback is not None and _reward_payload_score(fallback) >= 0:
        meta["chosen_score"] = _reward_payload_score(fallback)
        meta["via"] = "regex"
        return fallback, meta
    raise json.JSONDecodeError("No valid reward JSON object found", text, 0)


def _extract_reward_json(text: str) -> dict:
    """
    Robustly extract one JSON object containing explanation/question from noisy LLM output.
    Collects all raw_decode candidates (incl. fenced ```json blocks), scores them, picks best.
    Falls back to regex when JSON is truncated but key strings are well-formed.
    """
    d, _ = _extract_reward_json_impl(text)
    return d


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
        difficulty = _DIFFICULTY_NAME_TO_INT.get(str(raw_diff).lower()) or (int(raw_diff) if str(raw_diff).isdigit() else 1)
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
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="rl_train.py:reward_fn:completion-shape",
            message="Incoming completion container shape",
            data={
                "example_index": i,
                "completion_type": type(completion).__name__,
                "is_list": isinstance(completion, list),
                "list_len": len(completion) if isinstance(completion, list) else None,
                "last_item_type": type(completion[-1]).__name__ if isinstance(completion, list) and completion else None,
            },
        )
        # endregion
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)

        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H2",
            location="rl_train.py:reward_fn:text-preview",
            message="Raw completion text preview and brace metrics",
            data={
                "example_index": i,
                "text_len": len(text),
                "open_braces": text.count("{"),
                "close_braces": text.count("}"),
                "preview_start": text[:180],
                "preview_end": text[-180:],
            },
        )
        # endregion

        concept    = concepts[i]
        mastery    = float(masteries[i])
        difficulty = int(difficulties[i])
        history    = histories[i] if histories[i] else []

        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            raw = m.group() if m else text
            # region agent log
            _debug_log(
                run_id="pre-fix",
                hypothesis_id="H3",
                location="rl_train.py:reward_fn:raw-capture",
                message="Regex capture details before JSON parse",
                data={
                    "example_index": i,
                    "used_regex_match": bool(m),
                    "raw_len": len(raw),
                    "raw_preview_start": raw[:180],
                    "raw_preview_end": raw[-180:],
                },
            )
            # endregion
            parsed, parse_meta = _extract_reward_json_impl(text)
            # region agent log
            _debug_log(
                run_id="post-parser-change",
                hypothesis_id="H5",
                location="rl_train.py:reward_fn:parser-result",
                message="Structured extractor produced reward JSON",
                data={
                    "example_index": i,
                    "has_explanation": "explanation" in parsed,
                    "has_question": "question" in parsed,
                    "explanation_len": len(str(parsed.get("explanation", ""))),
                    "question_len": len(str(parsed.get("question", ""))),
                    **parse_meta,
                },
            )
            # endregion
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

    # Set generation parameters on the model — works across all TRL versions.
    model.generation_config.max_new_tokens = 600
    model.generation_config.temperature = 0.8
    model.generation_config.do_sample = True

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
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=5,
            save_strategy="epoch",
            num_generations=4,
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
