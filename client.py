"""
Client for the Adaptive Tutor Environment.

Example:
    >>> with AdaptiveTutorEnv(base_url="http://localhost:8000") as env:
    ...     env.reset(task="concept_recall")
    ...     tools = env.list_tools()
    ...     q = env.call_tool("get_current_question")
    ...     result = env.call_tool(
    ...         "submit_explanation",
    ...         explanation="...",
    ...         worked_example="...",
    ...     )

Example with Docker:
    >>> env = AdaptiveTutorEnv.from_docker_image("adaptive-tutor:latest")
    >>> try:
    ...     env.reset(task="advanced_analysis")
    ...     result = env.call_tool("submit_explanation", explanation="...", worked_example="...")
    ... finally:
    ...     env.close()
"""

from typing import Any, Dict, Optional

from openenv.core.mcp_client import MCPToolClient


class AdaptiveTutorEnv(MCPToolClient):
    """
    Client for the Adaptive Tutor Environment.

    Inherits all MCPToolClient functionality:
    - list_tools()
    - call_tool(name, **kwargs)
    - reset(**kwargs)         — pass task= and concept_mastery= here
    - step(action)
    - close()
    - from_docker_image(image_name)
    - from_env(hf_space_id)
    """

    async def get_mastery_state(self) -> Dict[str, Any]:
        """Return the student's current concept mastery levels and weakest concept."""
        return await self.call_tool("get_mastery_state")

    async def get_current_question(self) -> Dict[str, Any]:
        """Return the question the student got wrong along with their wrong answer."""
        return await self.call_tool("get_current_question")

    async def submit_explanation(
        self, explanation: str, worked_example: str
    ) -> Dict[str, Any]:
        """Submit an explanation and worked example. Ends the episode."""
        return await self.call_tool(
            "submit_explanation",
            explanation=explanation,
            worked_example=worked_example,
        )
