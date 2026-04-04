"""
State and data models for the Adaptive Tutor environment.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import State
from pydantic import BaseModel


TASKS = ("concept_recall", "application_practice", "advanced_analysis")
TASK_DIFFICULTY: Dict[str, int] = {
    "concept_recall": 1,
    "application_practice": 2,
    "advanced_analysis": 3,
}
DIFFICULTY_LABELS: Dict[int, str] = {1: "easy", 2: "medium", 3: "hard"}


@dataclass
class Question:
    """A single MCQ question in the question bank."""

    id: str
    concept: str
    difficulty: int
    question: str
    options: List[str]
    answer: str  # "A", "B", "C", or "D"


class TutorState(State):
    """
    Internal state for the Adaptive Tutor environment.

    Tracks everything needed to manage a tutoring episode:
    concept mastery, current question, and progress phase.

    Attributes (beyond inherited episode_id, step_count):
        task: Active task name — drives difficulty and grader selection.
        concept_mastery: Mastery score per concept, range [0.0, 1.0].
        seen_question_ids: IDs of questions already presented this session.
        current_concept: Concept being practiced this episode.
        current_difficulty: Difficulty level of the current question (1/2/3).
        difficulty_label: Human-readable difficulty ("easy"/"medium"/"hard").
        current_question_id: ID of the question the student got wrong.
        prev_skill: Mastery before the agent's explanation.
        prev_error: Wrong option the student chose on the initial attempt.
        phase: Current episode phase.
    """

    task: str = "concept_recall"
    concept_mastery: Dict[str, float] = {}
    seen_question_ids: List[str] = []
    current_concept: str = ""
    current_difficulty: int = 1
    difficulty_label: str = "easy"
    current_question_id: str = ""
    prev_skill: float = 0.0
    prev_error: Optional[str] = None
    phase: Literal["awaiting_explanation", "done"] = "awaiting_explanation"

    model_config = {"extra": "allow", "validate_assignment": True}


# ---------------------------------------------------------------------------
# Action / Output types (explicit Pydantic models for type safety)
# ---------------------------------------------------------------------------


class TutorAction(BaseModel):
    """Agent action: submit an explanation and worked example for a concept."""

    concept: str
    difficulty: Literal["easy", "medium", "hard"]
    explanation: str
    worked_example: str


class StudentProfile(BaseModel):
    """Snapshot of the student's skill levels per concept."""

    skill: Dict[str, float]


class MaterialFeedback(BaseModel):
    """Result of scoring the agent's explanation against concept keywords."""

    quality: float  # fraction of concept keywords covered, in [0.0, 1.0]
    concept: str


class QuestionResult(BaseModel):
    """Outcome of simulating the student on a follow-up question."""

    correct: bool
    difficulty: str  # "easy" | "medium" | "hard"
    concept: str


class StepOutput(BaseModel):
    """Structured output of a tutoring step (mirrors Observation fields)."""

    state: StudentProfile
    reward: float
    done: bool
    info: Dict[str, Any]
