"""
Adaptive Tutor Environment Implementation.

An RL environment where the agent is an LLM that generates a teaching question
AND explanation conditioned on the student's current mastery and difficulty.

Episode flow:
  1. reset()  → returns state {concept, mastery, difficulty, history}
  2. Agent calls get_state() to inspect current state.
  3. Agent calls submit_teaching_action(question, explanation).
  4. Environment:
       a. Scores explanation quality via ChatGPT judge.
       b. Scores question quality via ChatGPT judge.
       c. Simulates student: P(correct) = sigmoid formula.
       d. Updates mastery: α*quality if correct, β*quality if not.
       e. Computes composite reward.
  5. Returns done=True Observation with reward + full metadata.

MCP tools exposed to the agent:
  - get_state()                         → concept, mastery, difficulty, history
  - submit_teaching_action(question, explanation)  → triggers evaluation
"""

import logging
import os
from random import Random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Action, Observation

from ..models import (
    DIFFICULTY_LABELS,
    TASK_DIFFICULTY,
    TASKS,
    TutorState,
)
from .rewards import _EPS, compute_reward
from .student_model import (
    DEFAULT_MASTERY,
    mastery_to_difficulty,
    pick_weakest_concept,
    score_explanation_with_judge,
    score_question_with_judge,
    simulate_student,
    update_mastery,
)

logger = logging.getLogger(__name__)


def _make_judge_client():
    """Create an OpenAI client for the judge model, or return None if unavailable."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    judge_base_url = os.getenv("JUDGE_BASE_URL")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if judge_base_url:
            kwargs["base_url"] = judge_base_url
        return OpenAI(**kwargs)
    except Exception:
        return None


class AdaptiveTutorEnvironment(MCPEnvironment):
    """
    Adaptive DSA tutor environment for RL training.

    Each episode:
      - reset() selects the weakest concept and sets difficulty from mastery.
      - submit_teaching_action() tool triggers judge scoring, student simulation,
        mastery update, and reward computation.

    Supports concurrent sessions (each instance is independent).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self._rng: Random = Random()
        self._judge_client = _make_judge_client()

        mcp = FastMCP("adaptive_tutor")
        env = self

        @mcp.tool
        def get_state() -> Dict[str, Any]:
            """
            Return the current tutoring state.

            Returns a dict with:
              - concept: DSA concept to teach this episode
              - mastery: student's current mastery score in [0.0, 1.0]
              - difficulty: target difficulty level (1=easy, 2=medium, 3=hard)
              - difficulty_label: human-readable difficulty string
              - history: list of past step summaries in this session
            """
            return {
                "concept": env._state.current_concept,
                "mastery": env._state.prev_skill,
                "difficulty": env._state.current_difficulty,
                "difficulty_label": env._state.difficulty_label,
                "history": list(env._state.history),
            }

        @mcp.tool
        def submit_teaching_action(question: str, explanation: str) -> Dict[str, Any]:
            """
            Submit a teaching question and explanation for the student.

            The agent should generate:
              - question: A question appropriate for the concept and difficulty.
              - explanation: A clear explanation that helps the student understand
                            the concept, adapted to their mastery level.

            The environment will:
              1. Score explanation quality via a judge model.
              2. Score question quality via a judge model.
              3. Simulate the student's response (sigmoid probability model).
              4. Update the student's mastery.
              5. Compute and return the composite reward.

            Args:
                question:    Agent-generated question for the student.
                explanation: Agent-generated explanation of the concept.

            Returns:
                dict with status and confirmation message.
                The reward and episode result are on the Observation returned by step().
            """
            if env._state.phase != "awaiting_action":
                return {
                    "status": "error",
                    "message": "No active episode or episode already completed.",
                }
            return {
                "status": "received",
                "concept": env._state.current_concept,
                "message": "Teaching action received. Evaluating with judge and simulating student.",
            }

        super().__init__(mcp)
        # Initialize with phase="done" so submit_teaching_action is rejected before reset()
        self._state = TutorState(phase="done")

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: str = "concept_recall",
        concept_mastery: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment for a new tutoring episode.

        Args:
            seed:            Random seed for reproducibility.
            episode_id:      Optional episode identifier.
            task:            Task name — drives difficulty selection.
                             One of: "concept_recall", "application_practice",
                             "advanced_analysis".
            concept_mastery: Initial mastery dict. Defaults to DEFAULT_MASTERY.

        Returns:
            Observation with the current state for the agent to condition on.
        """
        if task not in TASKS:
            task = "concept_recall"

        self._rng = Random(seed)

        mastery = dict(concept_mastery) if concept_mastery else dict(DEFAULT_MASTERY)

        difficulty = TASK_DIFFICULTY[task]
        difficulty_label = DIFFICULTY_LABELS[difficulty]

        concept = pick_weakest_concept(mastery)
        prev_skill = mastery.get(concept, 0.1)

        self._state = TutorState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task=task,
            concept_mastery=mastery,
            current_concept=concept,
            current_difficulty=difficulty,
            difficulty_label=difficulty_label,
            prev_skill=prev_skill,
            history=[],
            phase="awaiting_action",
        )

        logger.info(
            "Reset: task=%s concept=%s difficulty=%s mastery=%.2f",
            task,
            concept,
            difficulty_label,
            prev_skill,
        )

        return Observation(
            done=False,
            reward=_EPS,
            metadata={
                "task": task,
                "concept": concept,
                "mastery": prev_skill,
                "difficulty": difficulty,
                "difficulty_label": difficulty_label,
                "history": [],
                "phase": "awaiting_action",
            },
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        if (
            isinstance(action, CallToolAction)
            and action.tool_name == "submit_teaching_action"
            and self._state.phase == "awaiting_action"
        ):
            question = action.arguments.get("question", "")
            explanation = action.arguments.get("explanation", "")
            return self._evaluate_teaching_action(question, explanation, obs)

        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)

        if (
            isinstance(action, CallToolAction)
            and action.tool_name == "submit_teaching_action"
            and self._state.phase == "awaiting_action"
        ):
            question = action.arguments.get("question", "")
            explanation = action.arguments.get("explanation", "")
            return self._evaluate_teaching_action(question, explanation, obs)

        return obs

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        return Observation(
            done=False,
            reward=_EPS,
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    "Use ListToolsAction or CallToolAction."
                )
            },
        )

    # ------------------------------------------------------------------
    # Core evaluation logic
    # ------------------------------------------------------------------

    def _evaluate_teaching_action(
        self, question: str, explanation: str, base_obs: Observation
    ) -> Observation:
        """
        Evaluate the agent's teaching action end-to-end.

        1. Score explanation quality via the judge model (or keyword fallback).
        2. Score question quality via the judge model (or heuristic fallback).
        3. Simulate student: P(correct) = sigmoid((mastery-bias)/temp)+guess-slip
        4. Update mastery: α*quality if correct, β*quality if not.
        5. Compute composite reward.
        6. Append step summary to history.
        7. Return done=True Observation with full metadata.
        """
        concept = self._state.current_concept
        difficulty = self._state.current_difficulty
        difficulty_label = self._state.difficulty_label
        prev_skill = self._state.prev_skill

        # 1. Score explanation quality
        explanation_scores = score_explanation_with_judge(
            explanation, concept, self._judge_client
        )
        explanation_quality = float(explanation_scores.get("final_score", 0.5))

        # 2. Score question quality
        question_scores = score_question_with_judge(
            question, concept, difficulty, self._judge_client
        )
        question_quality = float(question_scores.get("final_score", 0.5))

        # 3. Simulate student
        correct = simulate_student(prev_skill, difficulty, self._rng)

        # 4. Update mastery
        new_skill = update_mastery(prev_skill, explanation_quality, correct)

        # 5. Compute composite reward
        reward = compute_reward(
            prev_skill=prev_skill,
            new_skill=new_skill,
            correct=correct,
            difficulty=difficulty,
            explanation_quality=explanation_quality,
            question_quality=question_quality,
        )

        # 6. Record history entry
        step_summary = {
            "step": self._state.step_count,
            "concept": concept,
            "difficulty": difficulty_label,
            "explanation_quality": round(explanation_quality, 4),
            "question_quality": round(question_quality, 4),
            "correct": correct,
            "mastery_before": round(prev_skill, 4),
            "mastery_after": round(new_skill, 4),
            "reward": round(reward, 4),
        }
        self._state.history.append(step_summary)

        # Update state
        self._state.concept_mastery[concept] = new_skill
        self._state.phase = "done"

        logger.info(
            "Episode %s done: concept=%s expl_quality=%.2f q_quality=%.2f "
            "prev_skill=%.2f new_skill=%.2f correct=%s reward=%.4f",
            self._state.episode_id,
            concept,
            explanation_quality,
            question_quality,
            prev_skill,
            new_skill,
            correct,
            reward,
        )

        return Observation(
            done=True,
            reward=reward,
            metadata={
                **base_obs.metadata,
                "task": self._state.task,
                "concept": concept,
                "difficulty": difficulty_label,
                "explanation_quality": explanation_quality,
                "explanation_subscores": {
                    k: explanation_scores.get(k)
                    for k in ("correctness", "clarity", "example_quality", "depth")
                },
                "explanation_issues": explanation_scores.get("issues", []),
                "question_quality": question_quality,
                "question_subscores": {
                    k: question_scores.get(k)
                    for k in (
                        "relevance",
                        "difficulty_match",
                        "clarity",
                        "non_triviality",
                        "answerability",
                    )
                },
                "student_correct": correct,
                "mastery_before": prev_skill,
                "mastery_after": new_skill,
                "updated_mastery": dict(self._state.concept_mastery),
                "history": list(self._state.history),
                "phase": "done",
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def state(self) -> TutorState:
        return self._state
