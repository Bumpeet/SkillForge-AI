"""
inference.py — Adaptive Tutor environment evaluation script.

Hackathon submission script for the Meta × Hugging Face OpenEnv challenge.
Runs the RL agent (LLM explanation generator) against all three tasks and
emits structured logs for automated evaluation.

Required environment variables:
    API_BASE_URL      LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME        Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN          API key (also checked as API_KEY)
    OPENAI_API_KEY    API key for the judge model (ChatGPT). Falls back to HF_TOKEN.
    JUDGE_BASE_URL    Optional base URL for the judge model endpoint.
    LOCAL_IMAGE_NAME  Docker image name for from_docker_image() — optional

Stdout format (mandatory per hackathon spec):
    [START] task=<task_name> env=adaptive_tutor_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Example usage:
    HF_TOKEN=hf_xxx OPENAI_API_KEY=sk-xxx python inference.py
    LOCAL_IMAGE_NAME=adaptive-tutor:latest HF_TOKEN=hf_xxx python inference.py
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")

TASKS: List[str] = ["concept_recall", "application_practice", "advanced_analysis"]
SUCCESS_THRESHOLD: float = 0.3  # reward >= threshold → success
_EPS: float = 1e-6  # ensures scores are strictly between 0 and 1
TEMPERATURE: float = 0.7
MAX_TOKENS_EXPLANATION: int = 1500  # explanation needs more tokens
MAX_TOKENS_QUESTION: int = 400

EXPLANATION_SYSTEM_PROMPT: str = textwrap.dedent("""
    You are an expert DSA tutor.

    INPUT:
    - Concept: {concept}
    - Current mastery: {mastery} (0–1)
    - Previous mistakes: {past_wrong_questions}
    - Target difficulty: {difficulty}

    TASK:
    Generate learning material to improve the student.

    GUIDELINES:
    - Focus on mistakes from previous questions
    - Adapt to mastery:
      - <0.3 → simple, intuitive, step-by-step
      - 0.3–0.7 → balanced explanation + examples
      - >0.7 → concise, focus on edge cases
    - Include: intuition, key idea, worked example
    - Avoid unnecessary verbosity

    OUTPUT (strict JSON only, no extra text):
    {{
      "explanation": "..."
    }}
""").strip()

QUESTION_SYSTEM_PROMPT: str = textwrap.dedent("""
    You are an expert problem setter.

    INPUT:
    - Concept: {concept}
    - Difficulty: {difficulty}
    - Explanation: {explanation}

    TASK:
    Generate ONE question that tests the concepts taught in the explanation.

    GUIDELINES:
    - Must directly relate to explanation
    - Match difficulty:
      - Easy → definition/basic
      - Medium → application
      - Hard → reasoning/optimization
    - Avoid trivial or ambiguous questions
    - Ensure clear correct answer exists

    OUTPUT (strict JSON only, no extra text):
    {{
      "question": "..."
    }}
""").strip()


# ---------------------------------------------------------------------------
# Logging helpers (hackathon-required stdout format)
# ---------------------------------------------------------------------------


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=adaptive_tutor_env model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.6f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM call — agent generates question + explanation
# ---------------------------------------------------------------------------


def _parse_llm_json(text: str) -> dict:
    """
    Extract and parse the first complete JSON object from LLM output.

    Finds the outermost { ... } rather than splitting on backticks, which
    breaks when the model embeds ```code``` blocks inside the JSON string value.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    # Fallback: try to parse the whole text as-is
    return json.loads(text)


def get_explanation(
    client: Any,
    concept: str,
    mastery: float,
    difficulty: int,
    difficulty_label: str,
    past_wrong_questions: List[str],
) -> str:
    """
    Call Qwen to generate a teaching explanation.

    Conditions on concept, mastery, difficulty, and past wrong questions.
    Returns the explanation string.
    """
    past_q_str = "; ".join(past_wrong_questions) if past_wrong_questions else "none"
    try:
        # Escape braces in user content to prevent str.format() misinterpretation
        safe_past = past_q_str.replace("{", "{{").replace("}", "}}")
        prompt = EXPLANATION_SYSTEM_PROMPT.format(
            concept=concept,
            mastery=f"{mastery:.2f}",
            past_wrong_questions=safe_past,
            difficulty=difficulty_label,
        )
        print(f"[DEBUG] Calling Qwen for explanation: concept={concept} mastery={mastery:.2f} difficulty={difficulty_label}", flush=True)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_EXPLANATION,
        )
        raw = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] Explanation raw response (first 120 chars): {raw[:120]!r}", flush=True)
        explanation = _parse_llm_json(raw).get("explanation", "")
        print(f"[DEBUG] Explanation parsed OK, length={len(explanation)}", flush=True)
        return explanation
    except Exception as exc:
        print(f"[DEBUG] Explanation LLM call failed: {type(exc).__name__}: {exc}", flush=True)
        return (
            f"The concept of {concept} involves understanding its core principles. "
            f"At {difficulty_label} difficulty, focus on applying these principles step by step."
        )


def get_question(
    client: Any,
    concept: str,
    difficulty_label: str,
    explanation: str,
) -> str:
    """
    Call Qwen to generate a question conditioned on the explanation.

    Returns the question string.
    """
    try:
        # Escape braces in LLM-generated explanation to prevent str.format() misinterpretation
        safe_explanation = explanation.replace("{", "{{").replace("}", "}}")
        prompt = QUESTION_SYSTEM_PROMPT.format(
            concept=concept,
            difficulty=difficulty_label,
            explanation=safe_explanation,
        )
        print(f"[DEBUG] Calling Qwen for question: concept={concept} difficulty={difficulty_label}", flush=True)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_QUESTION,
        )
        raw = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] Question raw response (first 120 chars): {raw[:120]!r}", flush=True)
        question = _parse_llm_json(raw).get("question", "")
        print(f"[DEBUG] Question parsed OK, length={len(question)}", flush=True)
        return question
    except Exception as exc:
        print(f"[DEBUG] Question LLM call failed: {type(exc).__name__}: {exc}", flush=True)
        return f"What is the key principle behind {concept} at {difficulty_label} difficulty?"


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------


async def run_task(task: str, client: Any, env_factory) -> None:
    """Run one full episode for the given task and emit [START]/[STEP]/[END] logs."""
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = _EPS
    success: bool = False
    env = None

    log_start(task, MODEL_NAME)

    try:
        env = await env_factory(task)

        # --- Reset ---
        reset_result = await env.reset(task=task)
        reset_meta = reset_result.metadata if hasattr(reset_result, "metadata") else {}

        # --- Step 1: list available tools ---
        list_obs = await env.step({"type": "list_tools"})
        log_step(1, "list_tools", 0.0, False, None)
        rewards.append(0.0)
        steps_taken = 1

        # --- Step 2: get current state ---
        state_obs = await env.step(
            {"type": "call_tool", "tool_name": "get_state", "arguments": {}}
        )
        state_meta: Dict[str, Any] = {}
        if hasattr(state_obs, "result") and state_obs.result is not None:
            if hasattr(state_obs.result, "data") and isinstance(state_obs.result.data, dict):
                state_meta = state_obs.result.data
            elif isinstance(state_obs.result, dict):
                state_meta = state_obs.result
        if not state_meta and hasattr(state_obs, "metadata"):
            state_meta = state_obs.metadata.get("result", reset_meta)

        log_step(2, "get_state", 0.0, False, None)
        rewards.append(0.0)
        steps_taken = 2

        # Use fallback from reset metadata if tool result is empty
        context = state_meta if state_meta else reset_meta

        concept = context.get("concept", "?")
        mastery = context.get("mastery", 0.5)
        difficulty = context.get("difficulty", 1)
        difficulty_label = context.get("difficulty_label", "easy")

        # Extract past wrong questions from history
        history = context.get("history", [])
        past_wrong_questions = [
            h["question"]
            for h in history
            if not h.get("correct") and h.get("question")
        ]

        # --- Step 3a: agent generates explanation (Qwen call 1) ---
        explanation = get_explanation(
            client, concept, mastery, difficulty, difficulty_label, past_wrong_questions
        )

        # --- Step 3b: agent generates question conditioned on explanation (Qwen call 2) ---
        question = get_question(client, concept, difficulty_label, explanation)

        action_str = f"submit_teaching_action(concept={concept},q_len={len(question)},e_len={len(explanation)})"

        final_obs = await env.step(
            {
                "type": "call_tool",
                "tool_name": "submit_teaching_action",
                "arguments": {
                    "question": question,
                    "explanation": explanation,
                },
            }
        )

        reward = float(final_obs.reward) if final_obs.reward is not None else _EPS
        done = bool(final_obs.done)
        log_step(3, action_str, reward, done, None)
        rewards.append(reward)
        steps_taken = 3

        score = max(_EPS, min(1.0 - _EPS, reward))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        if not rewards:
            rewards = [_EPS]
        score = _EPS

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success, steps_taken, score, rewards)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env_factory(image_name: Optional[str], server_url: Optional[str] = None):
    """
    Return an async factory function that creates and resets an environment instance.

    Priority:
      1. Docker image (LOCAL_IMAGE_NAME) — spins up a container
      2. Server URL — connects to a running server
      3. Direct instantiation — runs environment in-process (for testing)
    """

    async def factory(task: str):
        if image_name:
            from adaptive_tutor_env.client import AdaptiveTutorEnv

            return await AdaptiveTutorEnv.from_docker_image(image_name)

        if server_url:
            from adaptive_tutor_env.client import AdaptiveTutorEnv
            from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

            _client = AdaptiveTutorEnv(base_url=server_url)

            class _ClientAdapter:
                """Converts dict actions to typed actions before forwarding to MCPToolClient."""

                async def reset(self, **kwargs):
                    return await _client.reset(**kwargs)

                async def step(self, action_dict):
                    atype = action_dict.get("type")
                    if atype == "list_tools":
                        return await _client.step(ListToolsAction())
                    elif atype == "call_tool":
                        return await _client.step(
                            CallToolAction(
                                tool_name=action_dict["tool_name"],
                                arguments=action_dict.get("arguments", {}),
                            )
                        )
                    raise ValueError(f"Unknown action type: {atype}")

                async def close(self):
                    await _client.close()

            return _ClientAdapter()

        # In-process fallback — wraps the environment in an async adapter
        from adaptive_tutor_env.server.tutor_environment import AdaptiveTutorEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

        env = AdaptiveTutorEnvironment()

        class _SyncAdapter:
            """Thin async adapter around the synchronous environment."""

            async def reset(self, **kwargs):
                return env.reset(**kwargs)

            async def step(self, action_dict):
                atype = action_dict.get("type")
                if atype == "list_tools":
                    return env.step(ListToolsAction())
                elif atype == "call_tool":
                    return env.step(
                        CallToolAction(
                            tool_name=action_dict["tool_name"],
                            arguments=action_dict.get("arguments", {}),
                        )
                    )
                raise ValueError(f"Unknown action type: {atype}")

            async def close(self):
                pass

        return _SyncAdapter()

    return factory


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    from openai import OpenAI

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "no-key")

    server_url = os.getenv("ADAPTIVE_TUTOR_URL")
    env_factory = make_env_factory(IMAGE_NAME, server_url)

    for task in TASKS:
        await run_task(task, client, env_factory)


if __name__ == "__main__":
    # Add the repo root to sys.path so local envs/ imports work
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    asyncio.run(main())
