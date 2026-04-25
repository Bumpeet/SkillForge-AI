"""
inference.py — Adaptive Tutor environment evaluation script.

Hackathon submission script for the Meta × Hugging Face OpenEnv challenge.
Runs the RL agent (LLM explanation generator) against all three tasks and
emits structured logs for automated evaluation.

Required environment variables:
    API_BASE_URL      LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME        Model identifier (default: Qwen/Qwen2.5-1.5B-Instruct)
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
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct:featherless-ai")
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")
SELECTED_CONCEPT: str = os.getenv("TUTOR_CONCEPT", "arrays")

TASKS: List[str] = ["concept_recall", "application_practice", "advanced_analysis"]
SUCCESS_THRESHOLD: float = 0.3  # reward >= threshold → success
_EPS: float = 1e-6  # tiny positive floor used only for unexpected exceptions
TEMPERATURE: float = 0.2
MAX_TOKENS_TEACHING_OUTPUT: int = 1800

TEACHING_SYSTEM_PROMPT: str = textwrap.dedent("""
    You are an expert DSA tutor and problem setter.

    INPUT:
    - Concept: {concept}
    - Current mastery: {mastery} (0–1)
    - Previous mistakes: {past_wrong_questions}
    - Target difficulty: {difficulty}

    TASK:
    Generate both:
    1. Teaching material that will help the student improve.
    2. One follow-up question that directly tests the material you generated.

    GUIDELINES:
    - Focus on mistakes from previous questions when relevant.
    - Adapt to mastery:
      - <0.3 → simple, intuitive, step-by-step
      - 0.3–0.7 → balanced explanation + examples
      - >0.7 → concise, focus on edge cases
    - Include in the material: intuition, key idea, worked example.
    - The follow-up question must directly relate to the material.
    - Match difficulty:
      - Easy → definition/basic
      - Medium → application
      - Hard → reasoning/optimization
    - Avoid trivial or ambiguous questions.
    - Ensure a clear correct answer exists.

    OUTPUT RULES:
    - Return valid JSON only.
    - Do not use markdown fences.
    - Do not include any text before or after the JSON object.
    - Escape all newlines inside JSON strings as \\n.
    - Use double quotes for all keys and string values.

    OUTPUT (strict JSON only):
    {{
      "explanation": "...",
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


def _sanitize_json_string(text: str) -> str:
    """
    Fix common LLM JSON issues without a full parser:
    - Literal control characters (newline, tab, etc.) inside string values.
    - Invalid escape sequences like \\( or \\e that JSON does not allow.
    """
    result = []
    in_string = False
    escaped = False
    _valid_escapes = set('"\\\/bfnrtu')
    for ch in text:
        if escaped:
            escaped = False
            if ch not in _valid_escapes:
                # Replace the lone backslash we already appended with a
                # double-backslash so the sequence becomes valid JSON.
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


def _parse_llm_json(text: str) -> dict:
    """
    Extract and parse the first complete JSON object from LLM output.

    Finds the outermost { ... } rather than splitting on backticks, which
    breaks when the model embeds ```code``` blocks inside the JSON string value.
    Falls back to sanitizing common LLM JSON quirks (bad escapes, literal
    control chars) before retrying.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_sanitize_json_string(text))


def get_teaching_output(
    client: Any,
    concept: str,
    mastery: float,
    difficulty: int,
    difficulty_label: str,
    past_wrong_questions: List[str],
) -> Dict[str, Any]:
    """
    Call Qwen once to generate both teaching material and a follow-up question.

    Conditions on concept, mastery, difficulty, and past wrong questions.
    Returns a dict with ``ok``, ``explanation``, ``question``, and ``error`` keys.
    """
    past_q_str = "; ".join(past_wrong_questions) if past_wrong_questions else "none"
    try:
        # Escape braces in user content to prevent str.format() misinterpretation
        safe_past = past_q_str.replace("{", "{{").replace("}", "}}")
        prompt = TEACHING_SYSTEM_PROMPT.format(
            concept=concept,
            mastery=f"{mastery:.2f}",
            past_wrong_questions=safe_past,
            difficulty=difficulty_label,
        )
        print(
            f"[DEBUG] Calling Qwen for teaching output: concept={concept} mastery={mastery:.2f} difficulty={difficulty_label}",
            flush=True,
        )
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_TEACHING_OUTPUT,
        )
        raw = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] Teaching raw response (first 120 chars): {raw[:120]!r}", flush=True)
        parsed = _parse_llm_json(raw)
        explanation = str(parsed.get("explanation", "")).strip()
        question = str(parsed.get("question", "")).strip()
        if not explanation or not question:
            raise ValueError("Model response missing explanation or question")
        print(
            f"[DEBUG] Teaching parsed OK, explanation_len={len(explanation)} question_len={len(question)}",
            flush=True,
        )
        return {
            "ok": True,
            "explanation": explanation,
            "question": question,
            "error": None,
        }
    except Exception as exc:
        print(f"[DEBUG] Teaching LLM call failed: {type(exc).__name__}: {exc}", flush=True)
        return {
            "ok": False,
            "explanation": "",
            "question": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


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
        reset_result = await env.reset(task=task, concept=SELECTED_CONCEPT)
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
        difficulty = context.get("targeted_difficulty", context.get("difficulty", 1))
        difficulty_label = context.get(
            "targeted_difficulty_label",
            context.get("difficulty_label", "easy"),
        )

        # Extract past wrong questions from history
        history = context.get("history", [])
        past_wrong_questions = [
            h["question"]
            for h in history
            if not h.get("correct") and h.get("question")
        ]

        # --- Step 3: agent generates teaching material + follow-up question (single Qwen call) ---
        teaching_output = get_teaching_output(
            client, concept, mastery, difficulty, difficulty_label, past_wrong_questions
        )
        if not teaching_output.get("ok"):
            log_step(3, "submit_teaching_action(parse_failed)", 0.0, True, teaching_output.get("error"))
            rewards.append(0.0)
            steps_taken = 3
            score = 0.0
            success = False
            return

        explanation = teaching_output["explanation"]
        question = teaching_output["question"]

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

        score = max(0.0, min(1.0, reward))
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
    server_path = os.path.join(repo_root, "server")
    if server_path not in sys.path:
        sys.path.insert(0, server_path)

    asyncio.run(main())
