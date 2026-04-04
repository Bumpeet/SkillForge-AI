"""
inference.py — Adaptive Tutor environment evaluation script.

Hackathon submission script for the Meta × Hugging Face OpenEnv challenge.
Runs the RL agent (LLM explanation generator) against all three tasks and
emits structured logs for automated evaluation.

Required environment variables:
    API_BASE_URL      LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME        Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN          API key (also checked as API_KEY)
    LOCAL_IMAGE_NAME  Docker image name for from_docker_image() — optional

Stdout format (mandatory per hackathon spec):
    [START] task=<task_name> env=adaptive_tutor_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Example usage:
    HF_TOKEN=hf_xxx python inference.py
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
TEMPERATURE: float = 0.7
MAX_TOKENS: int = 800

SYSTEM_PROMPT: str = textwrap.dedent("""
    You are an expert DSA (Data Structures and Algorithms) tutor.

    A student has answered a multiple-choice question incorrectly.
    Your task is to generate a clear, concise explanation of the correct concept
    and one concrete worked example that illustrates it.

    Focus on:
    1. Why the student's answer was wrong.
    2. Why the correct answer is right.
    3. A brief worked example (code snippet or step-by-step trace).

    Respond with a JSON object only — no extra text:
    {
        "explanation": "<explanation text, under 300 words>",
        "worked_example": "<worked example, under 200 words>"
    }
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


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def get_explanation(
    client: Any,
    question_context: Dict[str, Any],
) -> Dict[str, str]:
    """Call the LLM to generate an explanation for the student's wrong answer."""
    concept = question_context.get("concept", "")
    question = question_context.get("question", "")
    options = question_context.get("options", [])
    student_answer = question_context.get("student_answer", "?")
    difficulty = question_context.get("difficulty", "easy")

    user_prompt = (
        f"Concept: {concept}\n"
        f"Difficulty: {difficulty}\n"
        f"Question: {question}\n"
        f"Options:\n" + "\n".join(f"  {o}" for o in options) + "\n"
        f"Student answered: {student_answer}\n\n"
        f"Generate the explanation and worked example JSON."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        # Fallback: return a generic explanation so the episode can complete
        return {
            "explanation": f"The correct approach for {concept} involves understanding the core concept carefully.",
            "worked_example": "Consider the simplest example and build from there.",
        }


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------


async def run_task(task: str, client: Any, env_factory) -> None:
    """Run one full episode for the given task and emit [START]/[STEP]/[END] logs."""
    rewards: List[float] = []
    steps_taken: int = 0
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

        # --- Step 2: get current question context ---
        q_obs = await env.step(
            {"type": "call_tool", "tool_name": "get_current_question", "arguments": {}}
        )
        q_meta = {}
        if hasattr(q_obs, "result") and q_obs.result is not None:
            # CallToolResult.data holds the dict; fall back to parsing content text
            if hasattr(q_obs.result, "data") and isinstance(q_obs.result.data, dict):
                q_meta = q_obs.result.data
            elif isinstance(q_obs.result, dict):
                q_meta = q_obs.result
        if not q_meta and hasattr(q_obs, "metadata"):
            q_meta = q_obs.metadata.get("result", reset_meta)
        log_step(2, "get_current_question", 0.0, False, None)
        rewards.append(0.0)
        steps_taken = 2

        # Use fallback from reset metadata if tool result is empty
        context = q_meta if q_meta else reset_meta

        # --- Step 3: LLM generates and submits explanation ---
        parsed = get_explanation(client, context)
        explanation = parsed.get("explanation", "")
        worked_example = parsed.get("worked_example", "")
        action_str = f"submit_explanation(concept={context.get('concept','?')},len={len(explanation)})"

        final_obs = await env.step(
            {
                "type": "call_tool",
                "tool_name": "submit_explanation",
                "arguments": {
                    "explanation": explanation,
                    "worked_example": worked_example,
                },
            }
        )

        reward = float(final_obs.reward) if final_obs.reward is not None else 0.0
        done = bool(final_obs.done)
        log_step(3, action_str, reward, done, None)
        rewards.append(reward)
        steps_taken = 3

        success = reward >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        # Ensure we still emit [END]
        if not rewards:
            rewards = [0.0]

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success, steps_taken, rewards)


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

            return AdaptiveTutorEnv(base_url=server_url)

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
