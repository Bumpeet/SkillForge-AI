"""
Reward functions for the Adaptive Tutor environment.

Three graders matching the hackathon specification:
- grade_easy:   rewards mastery improvement (concept_recall task)
- grade_medium: rewards correctness + partial credit for different errors (application_practice)
- grade_hard:   rewards mastery improvement plus difficulty bonus (advanced_analysis)
"""

from typing import Optional

from .models import TASK_DIFFICULTY


def grade_easy(prev_skill: float, new_skill: float) -> float:
    """
    Reward for concept_recall (difficulty 1) episodes.

    Measures raw skill improvement. Gain of 0.5 → reward 1.0.

    Args:
        prev_skill: Mastery before the agent's explanation.
        new_skill:  Mastery after the agent's explanation.

    Returns:
        Reward in [0.0, 1.0].
    """
    improvement = new_skill - prev_skill
    return min(max(improvement * 2, 0.0), 1.0)


def grade_medium(
    prev_error: Optional[str],
    new_error: Optional[str],
    correct: bool,
) -> float:
    """
    Reward for application_practice (difficulty 2) episodes.

    Full reward for a correct follow-up answer; partial credit when the
    student makes a different mistake (evidence of partial learning).

    Args:
        prev_error: Wrong option chosen on the initial attempt.
        new_error:  Wrong option chosen on the follow-up (None if correct).
        correct:    Whether the student answered the follow-up correctly.

    Returns:
        1.0 if correct, 0.5 if different mistake, 0.0 if same mistake.
    """
    if correct:
        return 1.0
    if prev_error != new_error:
        return 0.5  # Different mistake → partial learning signal
    return 0.0


def grade_hard(
    prev_skill: float,
    new_skill: float,
    difficulty_label: str,
    correct: bool,
) -> float:
    """
    Reward for advanced_analysis (difficulty 3) episodes.

    Combines mastery improvement with a difficulty bonus when correct.

    Args:
        prev_skill:       Mastery before explanation.
        new_skill:        Mastery after explanation.
        difficulty_label: "easy", "medium", or "hard" — drives bonus size.
        correct:          Whether the student answered the follow-up correctly.

    Returns:
        Reward in [0.0, 1.0].
    """
    base = new_skill - prev_skill
    difficulty_bonus = {"easy": 0.2, "medium": 0.5, "hard": 1.0}[difficulty_label]
    if correct:
        return min(base + difficulty_bonus, 1.0)
    return max(base, 0.0)


def compute_reward(
    task: str,
    prev_skill: float,
    new_skill: float,
    difficulty_label: str,
    prev_error: Optional[str],
    new_error: Optional[str],
    correct: bool,
) -> float:
    """
    Dispatch to the appropriate grader based on the current task.

    Args:
        task:             Task name ("concept_recall", "application_practice",
                          or "advanced_analysis").
        prev_skill:       Mastery before explanation.
        new_skill:        Mastery after explanation.
        difficulty_label: Human-readable difficulty for grade_hard bonus.
        prev_error:       Initial wrong answer option letter.
        new_error:        Follow-up wrong answer option letter (None if correct).
        correct:          Whether the follow-up answer was correct.

    Returns:
        Reward in [0.0, 1.0].
    """
    difficulty_int = TASK_DIFFICULTY.get(task, 1)
    if difficulty_int == 1:
        return grade_easy(prev_skill, new_skill)
    elif difficulty_int == 2:
        return grade_medium(prev_error, new_error, correct)
    else:
        return grade_hard(prev_skill, new_skill, difficulty_label, correct)
