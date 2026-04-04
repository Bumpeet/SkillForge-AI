"""
Simulated student model for the Adaptive Tutor environment.

The student model provides:
- Default concept mastery levels for a new student
- Concept keyword dictionaries used to score explanation quality
- Functions for question selection, student simulation, and mastery updates
"""

import json
import os
from math import ceil
from random import Random
from typing import Dict, List, Optional, Tuple

from models import DIFFICULTY_LABELS, TASK_DIFFICULTY, Question


# Default starting mastery for a new student (weak in dp / backtracking / trees)
DEFAULT_MASTERY: Dict[str, float] = {
    "arrays": 0.8,
    "stack": 0.5,
    "trees": 0.3,
    "backtracking": 0.3,
    "dp": 0.1,
}

# Keywords per concept used to score explanation quality
CONCEPT_KEYWORDS: Dict[str, List[str]] = {
    "dp": [
        "memoization",
        "subproblem",
        "overlapping",
        "optimal",
        "recurrence",
        "tabulation",
    ],
    "arrays": [
        "index",
        "contiguous",
        "random access",
        "O(1)",
        "iteration",
        "element",
    ],
    "stack": [
        "LIFO",
        "push",
        "pop",
        "top",
        "overflow",
        "last in first out",
    ],
    "trees": [
        "node",
        "root",
        "leaf",
        "traversal",
        "height",
        "parent",
        "child",
    ],
    "backtracking": [
        "prune",
        "explore",
        "candidate",
        "constraint",
        "undo",
        "recursion",
    ],
}


def load_questions(data_dir: Optional[str] = None) -> List[Question]:
    """Load all questions from questions.json."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.join(data_dir, "questions.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        Question(
            id=q["id"],
            concept=q["concept"],
            difficulty=q["difficulty"],
            question=q["question"],
            options=q["options"],
            answer=q["answer"],
        )
        for q in raw["questions"]
    ]


def pick_weakest_concept(mastery: Dict[str, float]) -> str:
    """Return the concept with the lowest mastery score."""
    return min(mastery, key=lambda c: mastery[c])


def mastery_to_difficulty(mastery: float) -> int:
    """Map a mastery score [0,1] to a difficulty level 1/2/3."""
    return max(1, min(3, ceil(mastery * 3)))


def pick_question(
    concept: str,
    difficulty: int,
    seen_ids: List[str],
    questions: List[Question],
    rng: Random,
) -> Optional[Question]:
    """
    Pick a question for the given concept and difficulty not yet seen.

    Falls back to adjacent difficulties if no unseen question exists at the
    requested level. Returns None only when every question for the concept has
    been seen.
    """
    seen = set(seen_ids)
    for d in _difficulty_fallback_order(difficulty):
        candidates = [
            q
            for q in questions
            if q.concept == concept and q.difficulty == d and q.id not in seen
        ]
        if candidates:
            return rng.choice(candidates)
    return None


def _difficulty_fallback_order(difficulty: int) -> List[int]:
    """Return difficulty levels to try, starting with the requested one."""
    if difficulty == 1:
        return [1, 2, 3]
    elif difficulty == 2:
        return [2, 1, 3]
    else:
        return [3, 2, 1]


def get_wrong_options(options: List[str], answer: str) -> List[str]:
    """Return option letters that are not the correct answer."""
    return [opt[0] for opt in options if not opt.startswith(answer)]


def simulate_student(
    mastery: float,
    difficulty: int,
    options: List[str],
    answer: str,
    rng: Random,
) -> Tuple[bool, str]:
    """
    Simulate a student attempting a question.

    P(correct) = mastery / difficulty, clamped to [0, 1].
    If incorrect, picks uniformly from wrong options.

    Returns:
        (is_correct, chosen_option_letter)
    """
    p_correct = min(1.0, max(0.0, mastery / difficulty))
    if rng.random() < p_correct:
        return True, answer
    wrong = get_wrong_options(options, answer)
    chosen = rng.choice(wrong) if wrong else answer
    return False, chosen


def score_explanation(explanation: str, worked_example: str, concept: str) -> float:
    """
    Score explanation quality by keyword coverage.

    Counts how many concept keywords appear in the combined explanation +
    worked example text (case-insensitive). Returns fraction in [0.0, 1.0].
    """
    keywords = CONCEPT_KEYWORDS.get(concept, [])
    if not keywords:
        return 0.5
    combined = (explanation + " " + worked_example).lower()
    covered = sum(1 for kw in keywords if kw.lower() in combined)
    return covered / len(keywords)


def update_mastery(old_mastery: float, quality: float) -> float:
    """
    Update mastery after seeing an explanation.

    Max improvement per explanation is 0.2 × quality_score.
    Mastery is clamped to [0.0, 1.0].
    """
    return min(1.0, old_mastery + 0.2 * quality)
