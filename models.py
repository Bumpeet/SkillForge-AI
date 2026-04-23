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
    concept mastery, current teaching context, and episode history.

    Attributes (beyond inherited episode_id, step_count):
        task: Active task name — drives difficulty selection.
        concept_mastery: Mastery score per concept, range [0.0, 1.0].
        current_concept: Concept being practiced this episode.
        current_difficulty: Difficulty level (1/2/3).
        difficulty_label: Human-readable difficulty ("easy"/"medium"/"hard").
        prev_skill: Mastery before the agent's teaching action.
        history: List of step summary dicts for this session.
        phase: Current episode phase.
    """

    task: str = "concept_recall"
    concept_mastery: Dict[str, float] = {}
    current_concept: str = ""
    current_difficulty: int = 1
    difficulty_label: str = "easy"
    prev_skill: float = 0.0
    history: List[Dict[str, Any]] = []
    phase: Literal["awaiting_action", "done"] = "awaiting_action"

    model_config = {"extra": "allow", "validate_assignment": True}


# ---------------------------------------------------------------------------
# Action / Output types (explicit Pydantic models for type safety)
# ---------------------------------------------------------------------------


class TutorAction(BaseModel):
    """Agent action: submit a question and explanation for a concept."""

    concept: str
    difficulty: Literal["easy", "medium", "hard"]
    question: str
    explanation: str


class StudentProfile(BaseModel):
    """Snapshot of the student's skill levels per concept."""

    skill: Dict[str, float]


class MaterialFeedback(BaseModel):
    """Result of scoring the agent's explanation via the judge model."""

    quality: float  # final_score in [0.0, 1.0]
    concept: str
    subscores: Dict[str, float]


class QuestionResult(BaseModel):
    """Outcome of simulating the student on the agent's question."""

    correct: bool
    difficulty: str  # "easy" | "medium" | "hard"
    concept: str


class StepOutput(BaseModel):
    """Structured output of a tutoring step (mirrors Observation fields)."""

    state: StudentProfile
    reward: float
    done: bool
    info: Dict[str, Any]
