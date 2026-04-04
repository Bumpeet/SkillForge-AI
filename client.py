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

    pass
