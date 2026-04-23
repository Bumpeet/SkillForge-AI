"""
Composite reward function for the Adaptive Tutor environment.

The reward is a weighted combination of five signals:
    w1 * mastery_gain       — learning progress
    w2 * student_correct    — direct success signal
    w3 * difficulty_bonus   — harder tasks rewarded more
    w4 * explanation_quality — teaching quality (judge score)
    w5 * question_quality   — assessment quality (judge score)

Weights (sum to 1.0):
    w1=0.4, w2=0.2, w3=0.1, w4=0.2, w5=0.1

This unified formula replaces the previous task-specific graders
(grade_easy / grade_medium / grade_hard), ensuring the agent is
incentivised for genuine learning gain regardless of task type.
"""

_EPS = 1e-6  # Scores must be strictly (0, 1), never exactly 0.0 or 1.0

W1, W2, W3, W4, W5 = 0.4, 0.2, 0.1, 0.2, 0.1

# Bonus scales with difficulty so harder tasks earn proportionally more
DIFFICULTY_BONUS = {1: 0.2, 2: 0.5, 3: 1.0}


def compute_reward(
    prev_skill: float,
    new_skill: float,
    correct: bool,
    difficulty: int,
    explanation_quality: float,
    question_quality: float,
) -> float:
    """
    Compute the composite RL reward for a tutoring step.

    Args:
        prev_skill:           Mastery before the agent's teaching action.
        new_skill:            Mastery after updating with explanation quality.
        correct:              Whether the simulated student answered correctly.
        difficulty:           Difficulty level 1/2/3.
        explanation_quality:  Judge score for the explanation in [0, 1].
        question_quality:     Judge score for the question in [0, 1].

    Returns:
        Reward strictly in (1e-6, 1 - 1e-6).
    """
    mastery_gain = max(0.0, new_skill - prev_skill)
    correct_signal = 1.0 if correct else 0.0
    diff_bonus = DIFFICULTY_BONUS.get(difficulty, 0.2)

    raw = (
        W1 * mastery_gain
        + W2 * correct_signal
        + W3 * diff_bonus
        + W4 * explanation_quality
        + W5 * question_quality
    )
    return max(_EPS, min(1.0 - _EPS, raw))
