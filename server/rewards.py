"""
Reward function for the Adaptive Tutor environment.

The reward is driven by student outcomes only:
    1. Mastery gain - did the student appear to improve?
    2. Student correctness - did the student solve the follow-up question?
    3. Student confidence - how strong was the evidence of understanding?
"""

_EPS = 1e-6
MASTERY_GAIN_WEIGHT = 0.6
CORRECTNESS_WEIGHT = 0.3
CONFIDENCE_WEIGHT = 0.1


def compute_reward(
    prev_skill: float,
    new_skill: float,
    correct: bool,
    difficulty: int,
    confidence: float,
) -> float:
    """
    Compute the RL reward for a tutoring step.

    Args:
        prev_skill:           Mastery before the agent's teaching action.
        new_skill:            Mastery after updating with explanation quality.
        correct:              Whether the simulated student answered correctly.
        difficulty:           Unused in the implicit-relevance reward.
        confidence:           Student confidence in [0, 1].

    Returns:
        Reward where student success and mastery growth are the only signals.
    """
    del difficulty

    mastery_gain = new_skill - prev_skill
    confidence = max(0.0, min(1.0, confidence))
    return (
        MASTERY_GAIN_WEIGHT * mastery_gain
        + CORRECTNESS_WEIGHT * float(correct)
        + CONFIDENCE_WEIGHT * (confidence if correct else 0.0)
    )
