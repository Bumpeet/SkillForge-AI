"""
Generate sample tutoring rollouts and save them to JSON.

This script runs multiple sequential episodes per simulated user, carrying
forward mastery and history so you can inspect how the tutor behaves over time.

Example:
    python generate_rollout_samples.py --users 2 --episodes-per-user 3 --output data/sample_rollouts.json
"""

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI

from inference import (
    API_BASE_URL,
    API_KEY,
    IMAGE_NAME,
    MODEL_NAME,
    TASKS,
    get_teaching_output,
    make_env_factory,
)

DEFAULT_CONCEPTS = ["arrays", "stack", "trees", "backtracking", "dp"]


def _extract_tool_result(obs: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool result payloads from the environment/client adapters."""
    result: Dict[str, Any] = {}
    if hasattr(obs, "result") and obs.result is not None:
        if hasattr(obs.result, "data") and isinstance(obs.result.data, dict):
            result = obs.result.data
        elif isinstance(obs.result, dict):
            result = obs.result
    if not result and hasattr(obs, "metadata"):
        result = obs.metadata.get("result", fallback)
    return result or fallback


def build_concept_mastery(user_index: int) -> Dict[str, float]:
    """Create a stable initial mastery profile for one simulated user."""
    rng = random.Random(user_index + 1)
    return {
        concept: round(rng.uniform(0.1, 0.95), 2)
        for concept in DEFAULT_CONCEPTS
    }


async def run_episode(
    user_id: str,
    episode_index: int,
    task: str,
    concept: str,
    client: Any,
    env_factory: Any,
    concept_mastery: Dict[str, float],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one tutoring episode for a simulated user."""
    env = await env_factory(task)
    try:
        reset_obs = await env.reset(
            task=task,
            concept=concept,
            episode_id=f"{user_id}-episode-{episode_index + 1}",
            concept_mastery=concept_mastery,
            history=history,
        )
        reset_meta = reset_obs.metadata if hasattr(reset_obs, "metadata") else {}

        await env.step({"type": "list_tools"})
        state_obs = await env.step(
            {"type": "call_tool", "tool_name": "get_state", "arguments": {}}
        )
        state = _extract_tool_result(state_obs, reset_meta)

        concept = state.get("concept", "?")
        mastery = state.get("mastery", 0.5)
        difficulty = state.get("targeted_difficulty", state.get("difficulty", 1))
        difficulty_label = state.get(
            "targeted_difficulty_label",
            state.get("difficulty_label", "easy"),
        )
        current_history = state.get("history", [])
        past_wrong_questions = [
            item["question"]
            for item in current_history
            if not item.get("correct") and item.get("question")
        ]

        teaching_output = get_teaching_output(
            client,
            concept,
            mastery,
            difficulty,
            difficulty_label,
            past_wrong_questions,
        )
        if not teaching_output.get("ok"):
            return {
                "episode_index": episode_index + 1,
                "episode_id": f"{user_id}-episode-{episode_index + 1}",
                "task": task,
                "model": MODEL_NAME,
                "state": {
                    "concept": concept,
                    "mastery": mastery,
                    "difficulty": difficulty,
                    "difficulty_label": difficulty_label,
                    "concept_mastery_before": concept_mastery,
                    "history": current_history,
                    "past_wrong_questions": past_wrong_questions,
                },
                "qwen_outputs": {
                    "explanation": "",
                    "question": "",
                    "error": teaching_output.get("error"),
                },
                "environment_feedback": {
                    "reward": 0.0,
                    "done": True,
                    "error": teaching_output.get("error"),
                    "student_correct": None,
                    "student_answer": None,
                    "student_confidence": None,
                    "student_reasoning": None,
                    "student_provider": None,
                    "mastery_before": mastery,
                    "mastery_after": mastery,
                    "updated_mastery": concept_mastery,
                    "history": current_history,
                },
            }
        explanation = teaching_output["explanation"]
        question = teaching_output["question"]

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
        metadata = final_obs.metadata if hasattr(final_obs, "metadata") else {}

        return {
            "episode_index": episode_index + 1,
            "episode_id": f"{user_id}-episode-{episode_index + 1}",
            "task": task,
            "model": MODEL_NAME,
            "state": {
                "concept": concept,
                "mastery": mastery,
                "difficulty": difficulty,
                "difficulty_label": difficulty_label,
                "concept_mastery_before": concept_mastery,
                "history": current_history,
                "past_wrong_questions": past_wrong_questions,
            },
            "qwen_outputs": {
                "explanation": explanation,
                "question": question,
            },
            "environment_feedback": {
                "reward": float(getattr(final_obs, "reward", 0.0) or 0.0),
                "done": bool(getattr(final_obs, "done", False)),
                "student_correct": metadata.get("student_correct"),
                "student_answer": metadata.get("student_answer"),
                "student_confidence": metadata.get("student_confidence"),
                "student_reasoning": metadata.get("student_reasoning"),
                "student_provider": metadata.get("student_provider"),
                "mastery_before": metadata.get("mastery_before"),
                "mastery_after": metadata.get("mastery_after"),
                "updated_mastery": metadata.get("updated_mastery"),
                "history": metadata.get("history"),
            },
        }
    finally:
        await env.close()


async def generate_samples(
    user_count: int,
    episodes_per_user: int,
    output_path: str,
    task_sequence: Optional[List[str]] = None,
    concept: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate multi-episode user trajectories and save them as JSON."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "no-key")
    server_url = os.getenv("ADAPTIVE_TUTOR_URL")
    env_factory = make_env_factory(IMAGE_NAME, server_url)

    tasks = task_sequence or TASKS
    users: List[Dict[str, Any]] = []
    all_episodes: List[Dict[str, Any]] = []

    for user_index in range(user_count):
        user_id = f"user-{user_index + 1}"
        concept_mastery = build_concept_mastery(user_index)
        history: List[Dict[str, Any]] = []
        episodes: List[Dict[str, Any]] = []

        for episode_index in range(episodes_per_user):
            task = tasks[episode_index % len(tasks)]
            episode_concept = concept or DEFAULT_CONCEPTS[
                (user_index + episode_index) % len(DEFAULT_CONCEPTS)
            ]
            print(
                f"[USER] {user_id} episode={episode_index + 1}/{episodes_per_user} task={task} concept={episode_concept} model={MODEL_NAME}",
                flush=True,
            )
            episode = await run_episode(
                user_id=user_id,
                episode_index=episode_index,
                task=task,
                concept=episode_concept,
                client=client,
                env_factory=env_factory,
                concept_mastery=concept_mastery,
                history=history,
            )
            episodes.append(episode)
            all_episodes.append(episode)

            feedback = episode["environment_feedback"]
            concept_mastery = dict(feedback.get("updated_mastery") or concept_mastery)
            history = list(feedback.get("history") or history)

        users.append(
            {
                "user_id": user_id,
                "initial_concept_mastery": build_concept_mastery(user_index),
                "final_concept_mastery": concept_mastery,
                "episodes": episodes,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user_count": user_count,
        "episodes_per_user": episodes_per_user,
        "sample_count": len(all_episodes),
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "tasks": tasks,
        "users": users,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tutor rollouts with multiple episodes per user."
    )
    parser.add_argument(
        "--users",
        type=int,
        default=1,
        help="Number of simulated users to generate.",
    )
    parser.add_argument(
        "--episodes-per-user",
        type=int,
        default=10,
        help="Number of sequential episodes per user.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Backward-compatible alias for --episodes-per-user when using one user.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "sample_rollouts.json"),
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        help="Optional task sequence to cycle through. Defaults to all tasks.",
    )
    parser.add_argument(
        "--concept",
        choices=DEFAULT_CONCEPTS,
        help="Optional concept to use for all episodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes_per_user = args.samples if args.samples is not None else args.episodes_per_user
    asyncio.run(
        generate_samples(
            user_count=args.users,
            episodes_per_user=episodes_per_user,
            output_path=args.output,
            task_sequence=args.tasks,
            concept=args.concept,
        )
    )


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    main()
