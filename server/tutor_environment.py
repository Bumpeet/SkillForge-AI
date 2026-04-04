"""
Adaptive Tutor Environment Implementation.

An RL environment where the agent is an LLM that generates explanations for
weak DSA concepts. The environment:
  1. Identifies the student's weakest concept (lowest mastery score).
  2. Presents an MCQ question the simulated student got wrong.
  3. Waits for the agent to submit an explanation + worked example.
  4. Simulates the student on a follow-up question to measure improvement.
  5. Returns a reward using the hackathon-specified graders.

Three tasks (selected via reset(task=...)):
  - concept_recall        (difficulty 1): graded by grade_easy
  - application_practice  (difficulty 2): graded by grade_medium
  - advanced_analysis     (difficulty 3): graded by grade_hard

MCP tools exposed to the agent:
  - get_mastery_state()          → mastery levels + weakest concept
  - get_current_question()       → question text, options, student's wrong answer
  - submit_explanation(...)      → scores explanation, simulates follow-up, returns reward
"""

import logging
from random import Random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Action, Observation

from models import (
    DIFFICULTY_LABELS,
    TASK_DIFFICULTY,
    TASKS,
    Question,
    TutorState,
)
from .rewards import compute_reward
from .student_model import (
    DEFAULT_MASTERY,
    load_questions,
    mastery_to_difficulty,
    pick_question,
    pick_weakest_concept,
    score_explanation,
    simulate_student,
    update_mastery,
)

logger = logging.getLogger(__name__)


class AdaptiveTutorEnvironment(MCPEnvironment):
    """
    Adaptive DSA tutor environment for RL training.

    Each episode:
      - reset() selects the weakest concept, picks an MCQ the student fails.
      - submit_explanation() tool scores the agent's response and runs a follow-up.
      - Reward is returned at done=True with the episode result.

    Supports concurrent sessions (each instance is independent).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self._questions: List[Question] = load_questions(data_dir)
        self._rng: Random = Random()

        mcp = FastMCP("adaptive_tutor")

        # Keep reference to self for closures
        env = self

        @mcp.tool
        def get_mastery_state() -> Dict[str, Any]:
            """
            Return the student's current concept mastery levels.

            Returns a dict with:
              - mastery: mapping of concept → score in [0.0, 1.0]
              - weakest_concept: concept with the lowest mastery score
            """
            return {
                "mastery": dict(env._state.concept_mastery),
                "weakest_concept": pick_weakest_concept(env._state.concept_mastery),
            }

        @mcp.tool
        def get_current_question() -> Dict[str, Any]:
            """
            Return the question the student got wrong along with their wrong answer.

            Returns a dict with:
              - concept: the DSA concept being tested
              - difficulty: "easy", "medium", or "hard"
              - task: active task name
              - question: the question text
              - options: list of MCQ options
              - student_answer: the wrong option the student chose
            """
            q = env._get_question(env._state.current_question_id)
            return {
                "concept": env._state.current_concept,
                "difficulty": env._state.difficulty_label,
                "task": env._state.task,
                "question": q.question if q else "",
                "options": q.options if q else [],
                "student_answer": env._state.prev_error,
            }

        @mcp.tool
        def submit_explanation(explanation: str, worked_example: str) -> Dict[str, Any]:
            """
            Submit an explanation and worked example for the student's wrong answer.

            The environment will:
              1. Score the explanation quality by keyword coverage.
              2. Update the student's mastery accordingly.
              3. Simulate the student on a follow-up question.
              4. Compute and return the episode reward.

            Args:
                explanation:    Text explaining the correct concept.
                worked_example: A concrete worked example illustrating the concept.

            Returns:
                dict with status, concept, and a confirmation message.
                The reward and episode result are on the Observation returned by step().
            """
            if env._state.phase != "awaiting_explanation":
                return {
                    "status": "error",
                    "message": "No active episode or episode already completed.",
                }
            return {
                "status": "received",
                "concept": env._state.current_concept,
                "message": "Explanation received. Evaluating student follow-up.",
            }

        super().__init__(mcp)
        # Initialize with phase="done" so submit_explanation is rejected before first reset()
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
            task:            Task name — selects difficulty and grader.
                             One of: "concept_recall", "application_practice",
                             "advanced_analysis".
            concept_mastery: Initial mastery dict. Defaults to DEFAULT_MASTERY.

        Returns:
            Observation with the question the student got wrong.
        """
        if task not in TASKS:
            task = "concept_recall"

        self._rng = Random(seed)

        mastery = dict(concept_mastery) if concept_mastery else dict(DEFAULT_MASTERY)

        difficulty = TASK_DIFFICULTY[task]
        difficulty_label = DIFFICULTY_LABELS[difficulty]

        # Pick weakest concept and a question the student will likely fail
        concept = pick_weakest_concept(mastery)
        seen: List[str] = []

        question = pick_question(concept, difficulty, seen, self._questions, self._rng)
        if question is None:
            # Fallback: pick any unseen question for this concept
            question = pick_question(concept, 1, seen, self._questions, self._rng)

        # Simulate initial student attempt (likely wrong for weak concepts)
        prev_skill = mastery.get(concept, 0.1)
        is_correct, chosen = simulate_student(
            prev_skill, difficulty, question.options, question.answer, self._rng
        )

        # If student got it right, still proceed — the agent explains anyway
        prev_error = None if is_correct else chosen

        self._state = TutorState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task=task,
            concept_mastery=mastery,
            seen_question_ids=[question.id],
            current_concept=concept,
            current_difficulty=difficulty,
            difficulty_label=difficulty_label,
            current_question_id=question.id,
            prev_skill=prev_skill,
            prev_error=prev_error,
            phase="awaiting_explanation",
        )

        logger.info(
            "Reset: task=%s concept=%s difficulty=%s question=%s",
            task,
            concept,
            difficulty_label,
            question.id,
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "task": task,
                "concept": concept,
                "difficulty": difficulty_label,
                "mastery_state": mastery,
                "question": question.question,
                "options": question.options,
                "student_answer": prev_error,
                "phase": "awaiting_explanation",
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
            and action.tool_name == "submit_explanation"
            and self._state.phase == "awaiting_explanation"
        ):
            explanation = action.arguments.get("explanation", "")
            worked_example = action.arguments.get("worked_example", "")
            return self._evaluate_explanation(explanation, worked_example, obs)

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
            and action.tool_name == "submit_explanation"
            and self._state.phase == "awaiting_explanation"
        ):
            explanation = action.arguments.get("explanation", "")
            worked_example = action.arguments.get("worked_example", "")
            return self._evaluate_explanation(explanation, worked_example, obs)

        return obs

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
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

    def _evaluate_explanation(
        self, explanation: str, worked_example: str, base_obs: Observation
    ) -> Observation:
        """
        Score the agent's explanation and simulate the student's follow-up.

        1. Score explanation quality by keyword coverage.
        2. Update the student's mastery.
        3. Pick a follow-up question (same concept, same difficulty, not seen).
        4. Simulate student on follow-up.
        5. Compute reward via the appropriate grader.
        6. Update state and return done=True observation.
        """
        concept = self._state.current_concept
        difficulty = self._state.current_difficulty
        difficulty_label = self._state.difficulty_label
        prev_skill = self._state.prev_skill
        prev_error = self._state.prev_error

        # Score explanation quality
        quality = score_explanation(explanation, worked_example, concept)
        new_skill = update_mastery(prev_skill, quality)

        # Pick follow-up question
        followup = pick_question(
            concept,
            difficulty,
            self._state.seen_question_ids,
            self._questions,
            self._rng,
        )
        if followup is None:
            # No unseen follow-up — use a conceptually equivalent question
            followup = self._get_question(self._state.current_question_id)

        # Simulate student on follow-up with updated mastery
        correct, followup_chosen = simulate_student(
            new_skill, difficulty, followup.options, followup.answer, self._rng
        )
        new_error = None if correct else followup_chosen

        # Compute reward
        reward = compute_reward(
            task=self._state.task,
            prev_skill=prev_skill,
            new_skill=new_skill,
            difficulty_label=difficulty_label,
            prev_error=prev_error,
            new_error=new_error,
            correct=correct,
        )

        # Update state
        self._state.concept_mastery[concept] = new_skill
        if followup and followup.id != self._state.current_question_id:
            self._state.seen_question_ids.append(followup.id)
        self._state.phase = "done"

        logger.info(
            "Episode %s done: concept=%s quality=%.2f prev_skill=%.2f "
            "new_skill=%.2f correct=%s reward=%.2f",
            self._state.episode_id,
            concept,
            quality,
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
                "explanation_quality": quality,
                "prev_skill": prev_skill,
                "new_skill": new_skill,
                "follow_up_question": followup.question if followup else "",
                "follow_up_options": followup.options if followup else [],
                "student_correct": correct,
                "updated_mastery": dict(self._state.concept_mastery),
                "phase": "done",
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_question(self, question_id: str) -> Optional[Question]:
        """Look up a question by ID."""
        for q in self._questions:
            if q.id == question_id:
                return q
        return None

    @property
    def state(self) -> TutorState:
        return self._state
