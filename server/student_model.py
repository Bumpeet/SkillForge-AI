"""
Simulated student model for the Adaptive Tutor environment.

The student model provides:
- Default concept mastery levels for a new student
- LLM-backed student simulation with analytic fallback
- Differentiated mastery update (α when correct, β when incorrect)
- ChatGPT judge callers for explanation and question quality scoring
- Keyword fallback for when the judge API is unavailable
"""

import json
import os
from math import ceil, exp
from random import Random
from typing import Any, Dict, List, Optional

import requests
from models import DIFFICULTY_LABELS, TASK_DIFFICULTY, Question


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

TEMPERATURE: float = 0.2
GUESS_PROB: float = 0.10
SLIP_PROB: float = 0.05
DIFFICULTY_BIAS: Dict[int, float] = {1: 0.2, 2: 0.5, 3: 0.8}

# ---------------------------------------------------------------------------
# Mastery update constants
# ---------------------------------------------------------------------------

ALPHA: float = 0.2   # learning rate when student answers correctly
BETA: float = 0.05   # learning rate when student answers incorrectly (β < α)

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

EXPLANATION_JUDGE_PROMPT = (
    "You are an expert evaluator of teaching quality.\n"
    "INPUT:\n"
    "- Concept: {concept}\n"
    "- Explanation: {explanation}\n"
    "- Student mastery: {mastery}\n"
    "- Previous mistakes: {past_wrong_questions}\n"
    "- Target difficulty: {difficulty}\n\n"
    "TASK:\n"
    "Evaluate how well this explanation will help the student improve.\n\n"
    "Score (0–1):\n"
    "1. Correctness (technical accuracy)\n"
    "2. Clarity (based on mastery level)\n"
    "3. Example quality (useful worked example)\n"
    "4. Relevance to past mistakes\n"
    "5. Depth (appropriate for difficulty)\n\n"
    "SCORING:\n"
    "- Each criterion: 0 to 1\n"
    "- final_score = average of all 5\n\n"
    "OUTPUT FORMAT (strict JSON only, no extra text):\n"
    '{{"correctness": float, "clarity": float, "example_quality": float, '
    '"relevance": float, "depth": float, "final_score": float, "issues": []}}'
)

QUESTION_JUDGE_PROMPT = (
    "You are an expert evaluator of assessment quality.\n"
    "INPUT:\n"
    "- Concept: {concept}\n"
    "- Student mastery: {mastery}\n"
    "- Difficulty: {difficulty}\n"
    "- Explanation: {explanation}\n"
    "- Question: {question}\n\n"
    "TASK:\n"
    "Evaluate the quality of the question.\n\n"
    "Score (0–1):\n"
    "1. Relevance to concept\n"
    "2. Alignment with explanation\n"
    "3. Difficulty match\n"
    "4. Clarity (no ambiguity)\n"
    "5. Non-triviality\n\n"
    "OUTPUT FORMAT (strict JSON only, no extra text):\n"
    '{{"relevance": float, "alignment": float, "difficulty_match": float, '
    '"clarity": float, "non_triviality": float, "final_score": float}}'
)

STUDENT_SIMULATION_PROMPT = (
    "You are simulating a DSA student in a tutoring environment.\n\n"
    "Your job is to behave like a real student, not an expert tutor.\n\n"
    "You will be given:\n"
    "- concept\n"
    "- current mastery score from 0 to 1\n"
    "- target difficulty\n"
    "- previous mistakes\n"
    "- teacher explanation\n"
    "- teacher question\n\n"
    "You must answer like a student whose knowledge matches the mastery score.\n\n"
    "Behavior rules:\n"
    "- If mastery is low (< 0.3), you should often be confused, miss key ideas, or give incomplete answers.\n"
    "- If mastery is medium (0.3 to 0.7), you should understand some parts but still make application mistakes.\n"
    "- If mastery is high (> 0.7), you should usually answer correctly, but may still make minor mistakes on hard questions.\n"
    "- Use previous mistakes to stay consistent with past misunderstandings.\n"
    "- Do not suddenly sound like an expert if mastery is low.\n"
    "- Keep the answer short and realistic, like a student responding in an interview or classroom.\n"
    "- If the explanation is very clear and well matched to the student, you may improve your answer.\n"
    "- If the question is too hard for the mastery level, you should struggle.\n\n"
    "Return JSON only in this format:\n"
    '{{"student_answer": "...", "correct": true, "confidence": 0.0, "reasoning": "...", "question_tag": "concept_topic_keyword"}}\n\n'
    "question_tag rules:\n"
    "- 2-4 words joined by underscores, all lowercase\n"
    "- Derived from the concept and the specific topic the question tests\n"
    "- Examples: dp_base_case, arrays_two_pointer, trees_inorder_traversal\n\n"
    "INPUT:\n"
    "- Concept: {concept}\n"
    "- Mastery: {mastery}\n"
    "- Target difficulty: {difficulty}\n"
    "- Previous mistakes: {past_wrong_questions}\n"
    "- Teacher explanation: {explanation}\n"
    "- Teacher question: {question}\n"
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


def _simulate_student_formula(mastery: float, difficulty: int, rng: Random) -> bool:
    """
    Simulate a student attempting a question analytically.

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


def _call_openai_student(prompt: str) -> Dict[str, Any]:
    """Call an OpenAI-compatible model to simulate the student."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("STUDENT_OPENAI_BASE_URL"),
    )
    response = client.chat.completions.create(
        model=os.getenv("STUDENT_MODEL_NAME", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=250,
    )
    text = response.choices[0].message.content or ""
    return _parse_json_response(text)


def _call_gemini_student(prompt: str) -> Dict[str, Any]:
    """Call Gemini via REST to simulate the student."""
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 250,
        },
    }
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    body = response.json()
    candidates = body.get("candidates", [])
    text = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts)
    return _parse_json_response(text)


def simulate_student(
    concept: str,
    mastery: float,
    difficulty: int,
    explanation: str,
    question: str,
    past_wrong_questions: Optional[List[str]],
    rng: Random,
) -> Dict[str, Any]:
    """
    Simulate a student attempting a question.

    Prefers an LLM-backed student simulator using OpenAI, then Gemini.
    Falls back to the analytic sigmoid simulator if no API key is available
    or the LLM call fails.
    """
    difficulty_label = DIFFICULTY_LABELS.get(difficulty, "easy")
    past_q_str = "; ".join(past_wrong_questions) if past_wrong_questions else "none"
    safe_values = {
        "concept": concept.replace("{", "{{").replace("}", "}}"),
        "mastery": f"{mastery:.2f}",
        "difficulty": difficulty_label,
        "past_wrong_questions": past_q_str.replace("{", "{{").replace("}", "}}"),
        "explanation": explanation.replace("{", "{{").replace("}", "}}"),
        "question": question.replace("{", "{{").replace("}", "}}"),
    }
    prompt = STUDENT_SIMULATION_PROMPT.format(**safe_values)

    try:
        if os.getenv("OPENAI_API_KEY"):
            result = _call_openai_student(prompt)
            result["provider"] = "openai"
            return result
        if os.getenv("GEMINI_API_KEY"):
            result = _call_gemini_student(prompt)
            result["provider"] = "gemini"
            return result
    except Exception:
        pass

    correct = _simulate_student_formula(mastery, difficulty, rng)
    return {
        "student_answer": "",
        "correct": correct,
        "confidence": 0.5,
        "reasoning": "Analytic fallback used.",
        "provider": "formula",
    }


# ---------------------------------------------------------------------------
# Mastery update
# ---------------------------------------------------------------------------


def update_mastery(old_mastery: float, confidence: float, correct: bool) -> float:
    """
    Update mastery after a tutoring interaction.

    Uses the student outcome as the learning signal. Correct answers with
    higher confidence produce stronger mastery gains; incorrect answers still
    yield a small gain to reflect partial learning from exposure.

    Args:
        old_mastery: Mastery before the explanation.
        confidence:  Student confidence in [0.0, 1.0].
        correct:     Whether the simulated student answered correctly.

    Returns:
        Updated mastery clamped to [0.0, 1.0].
    """
    confidence = max(0.0, min(1.0, confidence))
    if correct:
        evidence = 0.5 + 0.5 * confidence
        rate = ALPHA
    else:
        evidence = 0.2 * (1.0 - 0.5 * confidence)
        rate = BETA
    gain = rate * evidence * (1.0 - old_mastery)
    return min(1.0, old_mastery + gain)


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
    """
    Extract and parse the first complete JSON object from LLM output.

    Uses { ... } boundaries instead of splitting on backticks, which breaks
    when the model embeds ```code``` blocks inside JSON string values.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    return json.loads(text)


def score_explanation_with_judge(
    explanation: str,
    concept: str,
    client: Any,
    mastery: float = 0.5,
    past_wrong_questions: Optional[List[str]] = None,
    difficulty: int = 1,
) -> Dict[str, Any]:
    """
    Call the judge model (ChatGPT) to score explanation quality.

    Subscores: correctness, clarity, example_quality, relevance, depth (each 0–1).
    final_score = average of all five.

    Falls back to keyword coverage if the API call fails.

    Returns:
        Dict with keys: correctness, clarity, example_quality, relevance,
        depth, final_score, issues.
    """
    difficulty_label = DIFFICULTY_LABELS.get(difficulty, "easy")
    past_q_str = (
        "; ".join(past_wrong_questions) if past_wrong_questions else "none"
    )
    # Escape braces in LLM-generated/user content to prevent str.format() misinterpretation
    safe_explanation = explanation.replace("{", "{{").replace("}", "}}")
    safe_past = past_q_str.replace("{", "{{").replace("}", "}}")
    prompt = EXPLANATION_JUDGE_PROMPT.format(
        concept=concept,
        explanation=safe_explanation,
        mastery=f"{mastery:.2f}",
        past_wrong_questions=safe_past,
        difficulty=difficulty_label,
    )
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
            "relevance": 0.5,
            "depth": quality,
            "final_score": (quality * 4 + 0.5) / 5,
            "issues": ["judge unavailable — keyword fallback used"],
        }


def score_question_with_judge(
    question: str,
    concept: str,
    difficulty: int,
    client: Any,
    explanation: str = "",
    mastery: float = 0.5,
) -> Dict[str, Any]:
    """
    Call the judge model (ChatGPT) to score question quality.

    Subscores: relevance, alignment, difficulty_match, clarity, non_triviality
    (each 0–1). final_score = average of all five.

    Falls back to 0.5 across all subscores if the API call fails.

    Returns:
        Dict with keys: relevance, alignment, difficulty_match, clarity,
        non_triviality, final_score.
    """
    difficulty_label = DIFFICULTY_LABELS.get(difficulty, "easy")
    # Escape braces in LLM-generated content to prevent str.format() misinterpretation
    safe_explanation = explanation.replace("{", "{{").replace("}", "}}")
    safe_question = question.replace("{", "{{").replace("}", "}}")
    prompt = QUESTION_JUDGE_PROMPT.format(
        concept=concept,
        mastery=f"{mastery:.2f}",
        difficulty=difficulty_label,
        explanation=safe_explanation,
        question=safe_question,
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
            "alignment": 0.5,
            "difficulty_match": 0.5,
            "clarity": 0.5,
            "non_triviality": 0.5,
            "final_score": 0.5,
        }
