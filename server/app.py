"""
FastAPI application for the Adaptive Tutor Environment.

Usage:
    # Development:
    uvicorn adaptive_tutor_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn adaptive_tutor_env.server.app:app --host 0.0.0.0 --port 8000

    # Or run directly:
    uv run --project . server
"""

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from .tutor_environment import AdaptiveTutorEnvironment

app = create_app(
    AdaptiveTutorEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="adaptive_tutor_env",
)


def main() -> None:
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
