"""
Simulated student model for the Adaptive Tutor environment.

The student model provides:
- Default concept mastery levels for a new student
- Sigmoid-based student simulation with guess and slip probabilities
- Differentiated mastery update (α when correct, β when incorrect)
- ChatGPT judge callers for explanation and question quality scoring
- Keyword fallback for when the judge API is unavailable
"""

import json
import os
from math import ceil, exp
from random import Random
from typing import Any, Dict, List, Optional, Tuple

from ..models import DIFFICULTY_LABELS, TASK_DIFFICULTY, Question


# ---------------------------------------------------------------------------
# Default mastery levels
# ---------------------------------------------------------------------------

DEFAULT_MASTERY: Dict[str, float] = {
    "arrays": 0.8,
    "stack": 0.5,
    "trees": 0.3,
    "backtracking": 0.3,
    "dp": 0.1,
}

# ---------------------------------------------------------------------------
# Concept keywords — used as fallback for explanation quality scoring
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Student simulation constants
# ---------------------------------------------------------------------------

TEMPERATURE: float = 0.3
GUESS_PROB: float = 0.10
SLIP_PROB: float = 0.05
DIFFICULTY_BIAS: Dict[int, float] = {1: 0.3, 2: 0.5, 3: 0.7}

# ---------------------------------------------------------------------------
# Mastery update constants
# ---------------------------------------------------------------------------

ALPHA: float = 0.3   # learning rate when student answers correctly
BETA: float = 0.1    # learning rate when student answers incorrectly (β < α)

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

EXPLANATION_JUDGE_PROMPT = (
    "You are an expert evaluator of teaching quality.\n"
    "INPUT:\n"
    "- Concept: {concept}\n"
    "- Explanation: {explanation}\n\n"
    "TASK:\n"
    "Score the explanation from 0 to 1 based on:\n"
    "1. Correctness (technical accuracy)\n"
    "2. Clarity (easy to understand)\n"
    "3. Example quality (useful worked example)\n"
    "4. Depth (appropriate for learning)\n\n"
    "SCORING:\n"
    "- Each criterion: 0 to 1\n"
    "- final_score = average of all 4\n\n"
    "OUTPUT FORMAT (strict JSON only, no extra text):\n"
    '{{"correctness": float, "clarity": float, "example_quality": float, '
    '"depth": float, "final_score": float, "issues": []}}'
)

QUESTION_JUDGE_PROMPT = (
    "You are an expert evaluator of assessment quality.\n"
    "INPUT:\n"
    "- Concept: {concept}\n"
    "- Difficulty: {difficulty}\n"
    "- Question: {question}\n\n"
    "TASK:\n"
    "Score the question from 0 to 1 based on:\n"
    "1. Relevance to concept\n"
    "2. Difficulty match\n"
    "3. Clarity (no ambiguity)\n"
    "4. Non-triviality\n"
    "5. Answerability\n\n"
    "OUTPUT FORMAT (strict JSON only, no extra text):\n"
    '{{"relevance": float, "difficulty_match": float, "clarity": float, '
    '"non_triviality": float, "answerability": float, "final_score": float}}'
)


# ---------------------------------------------------------------------------
# Question bank utilities (kept for reference; not used for agent actions)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Core student simulation
# ---------------------------------------------------------------------------


def simulate_student(mastery: float, difficulty: int, rng: Random) -> bool:
    """
    Simulate a student attempting a question.

    P(correct) = sigmoid((mastery - difficulty_bias) / temperature)
                 + guess_prob - slip_prob

    Higher mastery or lower difficulty → higher probability of success.
    guess_prob adds a floor; slip_prob caps the ceiling.

    Returns:
        True if the simulated student answers correctly.
    """
    bias = DIFFICULTY_BIAS[difficulty]
    p = 1.0 / (1.0 + exp(-(mastery - bias) / TEMPERATURE))
    p = min(1.0, max(0.0, p + GUESS_PROB - SLIP_PROB))
    return rng.random() < p


# ---------------------------------------------------------------------------
# Mastery update
# ---------------------------------------------------------------------------


def update_mastery(old_mastery: float, quality: float, correct: bool) -> float:
    """
    Update mastery after seeing an explanation.

    Uses a higher learning rate (ALPHA) when the student answered correctly,
    and a lower rate (BETA) when incorrect, reflecting that correct responses
    confirm actual learning while incorrect responses suggest partial learning.

    Args:
        old_mastery: Mastery before the explanation.
        quality:     Explanation quality score in [0.0, 1.0].
        correct:     Whether the simulated student answered correctly.

    Returns:
        Updated mastery clamped to [0.0, 1.0].
    """
    rate = ALPHA if correct else BETA
    return min(1.0, old_mastery + rate * quality)


# ---------------------------------------------------------------------------
# Explanation quality scoring (keyword fallback)
# ---------------------------------------------------------------------------


def score_explanation(explanation: str, concept: str) -> float:
    """
    Score explanation quality by keyword coverage (fallback).

    Returns fraction of concept keywords found in the explanation text.
    """
    keywords = CONCEPT_KEYWORDS.get(concept, [])
    if not keywords:
        return 0.5
    combined = explanation.lower()
    covered = sum(1 for kw in keywords if kw.lower() in combined)
    return covered / len(keywords)


# ---------------------------------------------------------------------------
# ChatGPT judge callers
# ---------------------------------------------------------------------------


def _parse_json_response(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def score_explanation_with_judge(
    explanation: str, concept: str, client: Any
) -> Dict[str, Any]:
    """
    Call the judge model (ChatGPT) to score explanation quality.

    Subscores: correctness, clarity, example_quality, depth (each 0–1).
    final_score = average of all four.

    Falls back to keyword coverage if the API call fails.

    Returns:
        Dict with keys: correctness, clarity, example_quality, depth,
        final_score, issues.
    """
    prompt = EXPLANATION_JUDGE_PROMPT.format(concept=concept, explanation=explanation)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        text = response.choices[0].message.content or ""
        return _parse_json_response(text)
    except Exception:
        quality = score_explanation(explanation, concept)
        return {
            "correctness": quality,
            "clarity": quality,
            "example_quality": quality,
            "depth": quality,
            "final_score": quality,
            "issues": ["judge unavailable — keyword fallback used"],
        }


def score_question_with_judge(
    question: str, concept: str, difficulty: int, client: Any
) -> Dict[str, Any]:
    """
    Call the judge model (ChatGPT) to score question quality.

    Subscores: relevance, difficulty_match, clarity, non_triviality,
    answerability (each 0–1). final_score = average of all five.

    Falls back to 0.5 across all subscores if the API call fails.

    Returns:
        Dict with keys: relevance, difficulty_match, clarity, non_triviality,
        answerability, final_score.
    """
    difficulty_label = DIFFICULTY_LABELS.get(difficulty, "easy")
    prompt = QUESTION_JUDGE_PROMPT.format(
        concept=concept, difficulty=difficulty_label, question=question
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = response.choices[0].message.content or ""
        return _parse_json_response(text)
    except Exception:
        return {
            "relevance": 0.5,
            "difficulty_match": 0.5,
            "clarity": 0.5,
            "non_triviality": 0.5,
            "answerability": 0.5,
            "final_score": 0.5,
        }
