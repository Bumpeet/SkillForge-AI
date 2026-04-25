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

An RL environment where an LLM agent acts as a personalized DSA (Data Structures & Algorithms) tutor. The agent generates a targeted **explanation** and a **question** conditioned on the student's mastery, difficulty, and past mistakes — evaluated by a judge model (ChatGPT), with reward driven by measurable student learning gain.

**Hackathon**: Meta x Hugging Face OpenEnv Challenge

## Overview

The environment tracks per-concept mastery for a simulated student across 5 DSA concepts. Each episode:

1. Identifies the student's **weakest concept** (lowest mastery score)
2. Sets **difficulty** based on the active task
3. Agent generates an **explanation** conditioned on `{concept, mastery, difficulty, past_wrong_questions}`
4. Agent generates a **question** conditioned on `{concept, difficulty, explanation}`
5. A **judge model (ChatGPT)** scores explanation quality (5 criteria) and question quality (5 criteria)
6. A **simulated student** attempts the question (sigmoid probability model)
7. **Mastery is updated** — faster if correct (α=0.2), slower if not (β=0.05)
8. Returns a **composite reward** across 5 weighted signals

**Why this matters**: Adaptive explanation generation is an unsolved problem in ed-tech. This environment trains agents to optimize *how* to explain concepts — grounding reward in measurable student improvement and teaching quality.

---

## End-to-End Flow

```
state = {concept, mastery, difficulty, history, past_wrong_questions}
    ↓
Qwen call 1: explanation = generate_explanation(concept, mastery, difficulty, past_wrong_questions)
    ↓
Qwen call 2: question = generate_question(concept, difficulty, explanation)
    ↓
ChatGPT judges explanation → {correctness, clarity, example_quality, relevance, depth}
ChatGPT judges question    → {relevance, alignment, difficulty_match, clarity, non_triviality}
    ↓
Student simulation: P(correct) = sigmoid((mastery - bias) / temp) + guess - slip
    ↓
Mastery update: mastery += α * explanation_quality  (if correct, α=0.2)
                mastery += β * explanation_quality  (if wrong,   β=0.05)
    ↓
Composite reward = w1*mastery_gain + w2*correct + w3*difficulty_bonus
                 + w4*explanation_quality + w5*question_quality
    ↓
RL updates agent (PPO / GRPO)
```

---

## Quick Start

### Using Docker

```bash
# Build the image (from the adaptive_tutor_env directory)
docker build -t adaptive-tutor:latest .

# Run the server
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-xxx \
  adaptive-tutor:latest
```

### Using the Client

```python
import asyncio
from adaptive_tutor_env import AdaptiveTutorEnv, CallToolAction, ListToolsAction

async def main():
    async with AdaptiveTutorEnv(base_url="http://localhost:8000") as env:
        # Reset — picks weakest concept, sets difficulty from task
        obs = await env.reset(task="concept_recall")
        print(f"Concept:    {obs.metadata['concept']}")
        print(f"Mastery:    {obs.metadata['mastery']:.2f}")
        print(f"Difficulty: {obs.metadata['difficulty_label']}")

        # (Optional) Inspect available tools
        list_obs = await env.step(ListToolsAction())

        # (Optional) Get current state
        state_obs = await env.step(CallToolAction(
            tool_name="get_state", arguments={}
        ))

        # Submit teaching action — ends the episode
        result = await env.step(CallToolAction(
            tool_name="submit_teaching_action",
            arguments={
                "question": "What technique avoids recomputing overlapping subproblems in DP?",
                "explanation": (
                    "Memoization stores the result of each subproblem the first time it is solved, "
                    "so subsequent calls return the cached value instead of recomputing. "
                    "Example: fib(5) = fib(4) + fib(3); with memoization, fib(3) is computed "
                    "once and reused in both branches."
                ),
            }
        ))
        print(f"Explanation quality: {result.metadata['explanation_quality']:.2f}")
        print(f"Question quality:    {result.metadata['question_quality']:.2f}")
        print(f"Student correct:     {result.metadata['student_correct']}")
        print(f"Mastery after:       {result.metadata['mastery_after']:.2f}")
        print(f"Reward:              {result.reward:.4f}")

asyncio.run(main())
```

### Running the Inference Script

```bash
# Full flow with judge model + Qwen agent
OPENAI_API_KEY=sk-xxx HF_TOKEN=hf_xxx python inference.py

# Without judge key (falls back to keyword scoring)
HF_TOKEN=hf_xxx python inference.py

# Against a local model endpoint
API_BASE_URL=http://localhost:8080/v1 MODEL_NAME=my-model HF_TOKEN=dummy python inference.py

# Against Docker image
LOCAL_IMAGE_NAME=adaptive-tutor:latest HF_TOKEN=hf_xxx OPENAI_API_KEY=sk-xxx python inference.py
```

---

## The 3 Tasks

Tasks are selected via `reset(task=...)`. Each maps to a difficulty level.

| Task | Difficulty | Teaching Focus |
|------|-----------|----------------|
| `concept_recall` | Easy (1) | Definitions and basic understanding |
| `application_practice` | Medium (2) | Applying the concept to a problem |
| `advanced_analysis` | Hard (3) | Trade-offs, optimisation, edge cases |

---

## DSA Concepts

5 concepts tracked, each with a default starting mastery:

| Concept | Default Mastery |
|---------|----------------|
| `dp` | 0.1 (weakest) |
| `backtracking` | 0.3 |
| `trees` | 0.3 |
| `stack` | 0.5 |
| `arrays` | 0.8 (strongest) |

The agent always teaches the **weakest concept** (lowest mastery).

---

## MCP Tools

The agent has access to 2 MCP tools:

| Tool | Arguments | Description |
|------|-----------|-------------|
| `get_state` | *(none)* | Returns `{concept, mastery, difficulty, difficulty_label, history}` |
| `submit_teaching_action` | `question: str, explanation: str` | Triggers judge scoring, student simulation, mastery update, and reward |

---

## Agent Prompts (Qwen)

The agent makes **two separate LLM calls** per episode.

### Step 1 — Generate Explanation

```
You are an expert DSA tutor.

INPUT:
- Concept: {concept}
- Current mastery: {mastery} (0–1)
- Previous mistakes: {past_wrong_questions}
- Target difficulty: {difficulty}

TASK:
Generate learning material to improve the student.

GUIDELINES:
- Focus on mistakes from previous questions
- Adapt to mastery:
  - <0.3 → simple, intuitive, step-by-step
  - 0.3–0.7 → balanced explanation + examples
  - >0.7 → concise, focus on edge cases
- Include: intuition, key idea, worked example

OUTPUT (strict JSON):
{"explanation": "..."}
```

### Step 2 — Generate Question (conditioned on explanation)

```
You are an expert problem setter.

INPUT:
- Concept: {concept}
- Difficulty: {difficulty}
- Explanation: {explanation}

TASK:
Generate ONE question that tests the concepts taught in the explanation.

GUIDELINES:
- Must directly relate to explanation
- Match difficulty: Easy → definition, Medium → application, Hard → reasoning
- Avoid trivial or ambiguous questions

OUTPUT (strict JSON):
{"question": "..."}
```

---

## Judge Model (ChatGPT)

### Explanation Quality

Scored on 5 criteria (each 0–1), `final_score` = average:

| Subscore | Description |
|----------|-------------|
| `correctness` | Technical accuracy |
| `clarity` | Ease of understanding relative to mastery level |
| `example_quality` | Usefulness of the worked example |
| `relevance` | Relevance to the student's past mistakes |
| `depth` | Appropriate depth for the target difficulty |

### Question Quality

Scored on 5 criteria (each 0–1), `final_score` = average:

| Subscore | Description |
|----------|-------------|
| `relevance` | Relevance to the concept |
| `alignment` | Alignment with the explanation content |
| `difficulty_match` | Matches the target difficulty |
| `clarity` | Unambiguous wording |
| `non_triviality` | Not too obvious or trivial |

> If `OPENAI_API_KEY` is not set, explanation quality falls back to keyword coverage and question quality defaults to 0.5.

---

## Student Model

Sigmoid-based probability with guess and slip noise:

```
P(correct) = sigmoid((mastery - difficulty_bias) / temperature)
             + guess_prob - slip_prob
```

| Parameter | Value |
|-----------|-------|
| `temperature` | 0.2 |
| `guess_prob` | 0.10 |
| `slip_prob` | 0.05 |
| `difficulty_bias` | 0.2 / 0.5 / 0.8 for easy / medium / hard |

---

## Mastery Update

Learning rate depends on whether the student answered correctly:

```
if correct:  mastery += α * explanation_quality   (α = 0.2)
else:        mastery += β * explanation_quality   (β = 0.05)

mastery = min(1.0, mastery)
```

---

## Composite Reward

```
reward = w1 * mastery_gain
       + w2 * student_correct
       + w3 * difficulty_bonus
       + w4 * explanation_quality
       + w5 * question_quality
```

| Component | Weight | Notes |
|-----------|--------|-------|
| Mastery gain `(new − old)` | 0.4 | Encourages genuine learning progress |
| Student correct | 0.2 | Direct success signal |
| Difficulty bonus | 0.1 | 0.1 / 0.2 / 0.3 for easy / medium / hard |
| Explanation quality | 0.2 | Judge score |
| Question quality | 0.1 | Judge score |

All rewards are clamped to `(1e-6, 1 - 1e-6)`.

---

## Episode History

Each completed step appends to `history` (accessible via `get_state()` and final `Observation.metadata`). Past questions the student got wrong are automatically surfaced as `past_wrong_questions` context in the next explanation prompt.

```json
{
  "step": 1,
  "concept": "dp",
  "difficulty": "easy",
  "question": "What is memoization and why is it used in DP?",
  "explanation_quality": 0.82,
  "question_quality": 0.75,
  "correct": true,
  "mastery_before": 0.10,
  "mastery_after": 0.264,
  "reward": 0.4284
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | HuggingFace API key for Qwen agent |
| `OPENAI_API_KEY` | *(optional)* | OpenAI API key for ChatGPT judge. Falls back to `HF_TOKEN` |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint for agent |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Agent model identifier |
| `JUDGE_BASE_URL` | *(optional)* | Custom base URL for judge model |
| `LOCAL_IMAGE_NAME` | *(optional)* | Docker image for `from_docker_image()` |
| `ADAPTIVE_TUTOR_URL` | *(optional)* | Connect to a running server instead of Docker |

---

## Development

```bash
# Install dependencies
pip install -e .

# Run server locally
uvicorn adaptive_tutor_env.server.app:app --reload --port 8000

# Run inference (in-process, no server needed)
OPENAI_API_KEY=sk-xxx HF_TOKEN=hf_xxx python inference.py

# Health check
curl http://localhost:8000/health
```

---

## Integration with RL Frameworks

### TRL (GRPO)

```python
import asyncio
from adaptive_tutor_env import AdaptiveTutorEnv, CallToolAction

async def rollout_func(prompts, completions, **kwargs):
    rewards = []
    async with AdaptiveTutorEnv(base_url="http://localhost:8000") as env:
        for output in completions:
            obs = await env.reset(task="concept_recall")
            result = await env.step(CallToolAction(
                tool_name="submit_teaching_action",
                arguments={
                    "question": output.get("question", ""),
                    "explanation": output.get("explanation", ""),
                },
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
    tool_name="submit_teaching_action",
    arguments={
        "question": "What is memoization?",
        "explanation": "Memoization caches overlapping subproblem results to avoid recomputation.",
    }
))
print(f"Reward: {result.reward:.4f}, Done: {result.done}")
```

---

## Project Structure

```
adaptive_tutor_env/
├── __init__.py                  # Exports AdaptiveTutorEnv, CallToolAction, ListToolsAction
├── client.py                    # AdaptiveTutorEnv(MCPToolClient)
├── models.py                    # TutorState, Question, TutorAction dataclasses
├── inference.py                 # Hackathon evaluation script (Qwen agent runner)
├── Dockerfile                   # Container image definition
├── openenv.yaml                 # OpenEnv spec (name, runtime, port)
├── pyproject.toml               # Package dependencies
├── README.md                    # This file (also HuggingFace Space card)
├── data/
│   └── questions.json           # 50 MCQ questions: 5 concepts × 3 difficulties
└── server/
    ├── __init__.py
    ├── app.py                   # create_app(AdaptiveTutorEnvironment, ...)
    ├── tutor_environment.py     # MCPEnvironment: get_state, submit_teaching_action
    ├── student_model.py         # sigmoid simulation, judge callers, mastery update
    └── rewards.py               # composite reward (w1–w5)
```
