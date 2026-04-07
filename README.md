---
title: Adaptive Tutor Environment Server
emoji: 📚
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - education
  - rl
  - dsa
---

# Adaptive Tutor Environment

An RL environment where an LLM agent acts as a personalized DSA (Data Structures & Algorithms) tutor. The agent generates targeted explanations for concepts a simulated student is struggling with, and is rewarded when the student improves on a follow-up question.

**Hackathon**: Meta x Hugging Face OpenEnv Challenge

## Overview

The environment tracks per-concept mastery for a simulated student across 5 DSA concepts. Each episode:

1. Identifies the student's **weakest concept** (lowest mastery score)
2. Presents an MCQ the student answered **incorrectly**
3. Waits for the agent to submit an **explanation + worked example**
4. Simulates the student on a **follow-up question** to measure improvement
5. Returns a **reward** proportional to learning gain

**Why this matters**: Adaptive explanation generation is an unsolved problem in ed-tech. This environment trains agents to optimize *how* to explain concepts, not just generate content - grounding reward in measurable student improvement.

## Quick Start

### Using Docker

```bash
# Build the image (from the adaptive_tutor_env directory)
docker build -t adaptive-tutor:latest .

# Run the server
docker run -p 8000:8000 adaptive-tutor:latest
```

### Using the Client

```python
import asyncio
from adaptive_tutor_env import AdaptiveTutorEnv, CallToolAction, ListToolsAction

async def main():
    async with AdaptiveTutorEnv(base_url="http://localhost:8000") as env:
        # Reset - picks weakest concept, presents failed question
        obs = await env.reset(task="concept_recall")
        print(f"Concept: {obs.metadata['concept']}")
        print(f"Question: {obs.metadata['question']}")
        print(f"Student answered: {obs.metadata['student_answer']}")

        # (Optional) Inspect available tools
        list_obs = await env.step(ListToolsAction())

        # (Optional) Get full question context
        q_obs = await env.step(CallToolAction(
            tool_name="get_current_question", arguments={}
        ))

        # Submit explanation - ends the episode
        result = await env.step(CallToolAction(
            tool_name="submit_explanation",
            arguments={
                "explanation": "Memoization stores overlapping subproblem results to avoid recomputation.",
                "worked_example": "fib(5) = fib(4) + fib(3), each computed once and cached.",
            }
        ))
        print(f"Student correct on follow-up: {result.metadata['student_correct']}")
        print(f"Reward: {result.reward:.2f}")
        print(f"Done: {result.done}")

asyncio.run(main())
```

### Running the Inference Script

```bash
# Against HuggingFace router (hackathon standard)
HF_TOKEN=hf_xxx python inference.py

# Against a local model endpoint
API_BASE_URL=http://localhost:8080/v1 MODEL_NAME=my-model HF_TOKEN=dummy python inference.py

# Against Docker image
LOCAL_IMAGE_NAME=adaptive-tutor:latest HF_TOKEN=hf_xxx python inference.py
```

## The 3 Tasks

Tasks are selected via `reset(task=...)`. Each task uses a different difficulty level and grader.

| Task | Difficulty | What the agent must explain | Grader |
|------|-----------|----------------------------|--------|
| `concept_recall` | Easy (1) | *What* the concept is - definitions, properties | `grade_easy`: reward ~ mastery gain |
| `application_practice` | Medium (2) | *How* to apply the concept to a problem | `grade_medium`: reward for correcting the error |
| `advanced_analysis` | Hard (3) | *Why* and *when* - trade-offs, complexity | `grade_hard`: mastery gain + difficulty bonus |

## DSA Concepts

The environment covers 5 DSA concepts, each with 10 MCQ questions across 3 difficulties:

| Concept | Initial Mastery | Keywords scored in explanations |
|---------|----------------|--------------------------------|
| `dp` | 0.1 (weakest by default) | memoization, subproblem, overlapping, optimal, recurrence, tabulation |
| `backtracking` | 0.3 | prune, explore, candidate, constraint, undo, recursion |
| `trees` | 0.3 | node, root, leaf, traversal, height, parent, child |
| `stack` | 0.5 | LIFO, push, pop, top, overflow, last in first out |
| `arrays` | 0.8 (strongest) | index, contiguous, random access, O(1), iteration, element |

## MCP Tools

The agent has access to 3 MCP tools:

| Tool | Arguments | Description |
|------|-----------|-------------|
| `get_mastery_state` | *(none)* | Returns `{mastery: {concept: score}, weakest_concept: str}` |
| `get_current_question` | *(none)* | Returns question text, options, student's wrong answer |
| `submit_explanation` | `explanation: str, worked_example: str` | Scores the explanation, runs follow-up simulation, ends episode |

## Reward Functions

Reward is dispatched by task difficulty:

**`grade_easy`** (concept_recall):
```
reward = min(max((new_skill - prev_skill) * 2, 0), 1)
```
Rewards any mastery gain - partial credit for partial explanations.

**`grade_medium`** (application_practice):
```
reward = 1.0 if correct else (0.5 if different_mistake else 0.0)
```
Full reward for correct follow-up; partial if student makes a *different* mistake (changed understanding).

**`grade_hard`** (advanced_analysis):
```
bonus = {easy: 0.2, medium: 0.5, hard: 1.0}[difficulty]
reward = min(skill_gain + bonus, 1.0) if correct else max(skill_gain, 0)
```
Rewards both measurable skill gain and getting the follow-up correct, with a difficulty bonus.

## Student Model

The simulated student uses a probabilistic model:

```
P(correct) = min(1.0, mastery / difficulty_int)
```

- `difficulty_int`: 1 (easy), 2 (medium), 3 (hard)
- Mastery starts at concept defaults and updates after each explanation
- `new_mastery = min(1.0, old_mastery + 0.2 * explanation_quality)`
- `explanation_quality` = fraction of concept keywords present in explanation + worked example

## Episode Flow

```
reset(task="application_practice", concept_mastery={"dp": 0.1, ...}, seed=42)
  ↓ picks weakest concept ("dp")
  ↓ picks an MCQ at difficulty 2
  ↓ simulates student: P(correct) = 0.1/2 = 0.05 → almost certainly wrong
  → Observation: question, options, student_answer="A", phase="awaiting_explanation"

step(get_current_question)    → question details (optional)
step(get_mastery_state)       → mastery dict (optional)

step(submit_explanation(explanation=..., worked_example=...))
  ↓ quality = keyword_coverage(explanation + worked_example, "dp")
  ↓ new_skill = min(1.0, 0.1 + 0.2 * quality)
  ↓ picks follow-up question (same concept, same difficulty, not seen before)
  ↓ simulates student: P(correct) = new_skill / 2
  ↓ reward = grade_medium(prev_error, new_error, correct)
  → Observation(done=True, reward=..., metadata={student_correct, new_skill, ...})
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint for inference.py |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier for inference.py |
| `HF_TOKEN` | *(required)* | HuggingFace API key |
| `LOCAL_IMAGE_NAME` | *(optional)* | Docker image name for from_docker_image() |
| `ADAPTIVE_TUTOR_URL` | *(optional)* | Connect to running server instead of Docker |

## Development

```bash
# Install dependencies
pip install -e .

# Run tests
pytest tests/ -v

# Run server locally
uvicorn adaptive_tutor_env.server.app:app --reload --port 8000

# Health check
curl http://localhost:8000/health
```

## Project Structure

```
adaptive_tutor_env/
├── __init__.py                  # Exports AdaptiveTutorEnv, CallToolAction, ListToolsAction
├── client.py                    # AdaptiveTutorEnv(MCPToolClient)
├── models.py                    # TutorState, Question dataclasses
├── inference.py                 # Hackathon evaluation script (LLM agent runner)
├── Dockerfile                   # Container image definition
├── openenv.yaml                 # OpenEnv spec (name, runtime, port)
├── pyproject.toml               # Package dependencies
├── README.md                    # This file (also HuggingFace Space card)
├── data/
│   └── questions.json           # 50 MCQ questions: 5 concepts × 3 difficulties
└── server/
    ├── __init__.py
    ├── app.py                   # create_app(AdaptiveTutorEnvironment, ...)
    ├── tutor_environment.py     # MCPEnvironment subclass with 3 MCP tools
    ├── student_model.py         # simulate_student, score_explanation, update_mastery
    └── rewards.py               # grade_easy, grade_medium, grade_hard, compute_reward
```

## Integration with RL Frameworks

### TRL (GRPO)

```python
import asyncio
from adaptive_tutor_env import AdaptiveTutorEnv, CallToolAction

async def rollout_func(prompts, completions, **kwargs):
    rewards = []
    async with AdaptiveTutorEnv(base_url="http://localhost:8000") as env:
        for explanation_text in completions:
            obs = await env.reset(task="concept_recall")
            result = await env.step(CallToolAction(
                tool_name="submit_explanation",
                arguments={"explanation": explanation_text, "worked_example": ""},
            ))
            rewards.append(result.reward)
    return rewards
```

### Direct In-Process (no server needed)

```python
from adaptive_tutor_env.server.tutor_environment import AdaptiveTutorEnvironment
from openenv.core.env_server.mcp_types import CallToolAction

env = AdaptiveTutorEnvironment()
obs = env.reset(task="concept_recall", seed=42)

result = env.step(CallToolAction(
    tool_name="submit_explanation",
    arguments={
        "explanation": "Memoization caches overlapping subproblem results.",
        "worked_example": "fib(n) = fib(n-1) + fib(n-2), cached per call.",
    }
))
print(f"Reward: {result.reward:.2f}, Done: {result.done}")
```
