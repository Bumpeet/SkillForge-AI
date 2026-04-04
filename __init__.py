"""
Adaptive Tutor Environment — Personalized DSA learning via RL-optimized explanations.

An agent generates concept explanations; reward measures student mastery improvement
on a follow-up question after seeing the explanation.

Example:
    >>> from adaptive_tutor_env import AdaptiveTutorEnv
    >>>
    >>> with AdaptiveTutorEnv(base_url="http://localhost:8000") as env:
    ...     env.reset(task="concept_recall")
    ...     tools = env.list_tools()
    ...     q = env.call_tool("get_current_question")
    ...     result = env.call_tool(
    ...         "submit_explanation",
    ...         explanation="Memoization stores subproblem results to avoid recomputation.",
    ...         worked_example="fib(5) = fib(4) + fib(3); cache each value.",
    ...     )
    ...     print(result)
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import AdaptiveTutorEnv

__all__ = ["AdaptiveTutorEnv", "CallToolAction", "ListToolsAction"]
