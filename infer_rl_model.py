# Must match `SYSTEM_PROMPT` in rl_train.py (GRPO data + reward JSON contract).
import json
import os
import time
from typing import Any, List, Optional

import requests

SYSTEM_PROMPT = (
    "You are an expert DSA tutor and problem setter.\n\n"
    "INPUT:\n"
    "- Concept: {concept}\n"
    "- Current mastery: {mastery:.2f} (0–1)\n"
    "- Previous mistakes: {mistakes}\n"
    "- Target difficulty: {difficulty}\n\n"
    "TASK:\n"
    "Generate both:\n"
    "1. Teaching material that will help the student improve.\n"
    "2. One follow-up question that directly tests the material you generated.\n\n"
    "OUTPUT RULES (critical — reward parsing depends on this):\n"
    "- Return exactly one JSON object and nothing else. The first non-whitespace character MUST be '{{' and the last non-whitespace character MUST be '}}'.\n"
    "- No markdown, no code fences (no ```), no labels such as OUTPUT:, Example:, EXPLANATION:, or NOTE: outside the JSON.\n"
    "- The object MUST have exactly two keys, both non-empty strings: \"explanation\" (teaching material) and \"question\" (one follow-up). No other top-level keys.\n"
    "- Do not use placeholders: never output the literal ellipsis \"...\" or empty strings for either field. Do not echo template text or \"fill in the blank\" examples.\n"
    "- Do not output a second JSON object, JavaScript/Python/C++ samples, or extra '{{' / '}}' blocks outside that single object. Put any code or examples inside the two string values only, with valid JSON escaping.\n"
    "- Inside strings, escape newlines as \\n and internal double quotes as \\\". Do not paste raw multi-line JSON or unescaped control characters inside a string.\n"
    "- Use double quotes for all keys and string values.\n"
)

# Default: Hugging Face dedicated Inference Endpoint (see handler.py on the Hub)
TUTOR_HOSTED_ENDPOINT: str = os.environ.get(
    "HF_TUTOR_ENDPOINT",
    "https://emx0oc53cv608mb6.eu-west-1.aws.endpoints.huggingface.cloud",
)
# Name shown in the endpoint UI (single-model URL already selects the deployment; not sent in body).
TUTOR_HOSTED_MODEL: str = os.environ.get(
    "HF_TUTOR_MODEL",
    "qwen2-5-1-5b-adaptive-tutor--wdu",
)

# Example: same style as data/episode_flows.json (tags = past wrong questions = history)
topic: str = "dp"
mastery: float = 0.13
difficulty_label: str = "easy"  # easy | medium | hard
history: List[str] = ["dp_fibonacci", "dp_min_cost_climbing", "dp_climbing_stairs"]


def _fmt_safe(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


def build_user_prompt() -> str:
    joined = "; ".join(history) if history else "none"
    return SYSTEM_PROMPT.format(
        concept=_fmt_safe(str(topic)),
        mastery=mastery,
        mistakes=_fmt_safe(joined),
        difficulty=_fmt_safe(str(difficulty_label)),
    )


def _first_json_object(text: str) -> Optional[dict]:
    i = text.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_generated_text(data: Any) -> str:
    """Normalize Inference Endpoint / custom handler response bodies."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if "generated_text" in data and isinstance(data["generated_text"], str):
            return data["generated_text"]
        for key in ("outputs", "data", "output", "result"):
            if key in data:
                return _extract_generated_text(data[key])
    if isinstance(data, list) and data:
        return _extract_generated_text(data[0])
    raise ValueError(
        f"Could not read generated text from response: {str(data)[:500]!r}"
    )


def call_hosted_tutor(
    user_content: str,
    *,
    endpoint: Optional[str] = None,
    token: Optional[str] = None,
    max_new_tokens: int = 1024,
    max_time: Optional[float] = 180.0,
    temperature: float = 0.2,
    do_sample: bool = True,
) -> str:
    """
    Call the hosted EndpointHandler: JSON body with ``inputs`` = raw user message
    (the server runs ``apply_chat_template`` + generate). Same contract as local ``handler.py``.

    ``max_time`` is sent to the server (Transformers ``generate(max_time=...)``) to cap
    wall-clock generation time. The **first** request after idle may still take several
    minutes for **model load** (cold start) before generation begins — see endpoint logs in HF.
    """
    base = (endpoint or TUTOR_HOSTED_ENDPOINT).rstrip("/")
    t = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not t:
        raise ValueError(
            "Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) to call the private endpoint."
        )

    parameters: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
    }
    if max_time is not None and max_time > 0:
        parameters["max_time"] = max_time
    payload: dict = {"inputs": user_content, "parameters": parameters}

    headers = {
        "Authorization": f"Bearer {t}",
        "Content-Type": "application/json",
    }
    r = requests.post(
        base,
        json=payload,
        headers=headers,
        timeout=600,
    )
    r.raise_for_status()
    return _extract_generated_text(r.json())


if __name__ == "__main__":
    user_content = build_user_prompt()
    max_tok = int(os.environ.get("TUTOR_MAX_NEW_TOKENS", "1024"))
    _mts = os.environ.get("TUTOR_MAX_TIME_SEC", "180").strip()
    max_time_sec: Optional[float] = float(_mts) if _mts else None

    t0 = time.perf_counter()
    _mt = f"{max_time_sec}s" if max_time_sec is not None else "server default"
    print(
        f"Calling endpoint (max_new_tokens={max_tok}, max_time={_mt}). "
        "First request after scale-to-zero can take several minutes to load the model.",
        flush=True,
    )
    assistant = call_hosted_tutor(
        user_content,
        max_new_tokens=max_tok,
        max_time=max_time_sec,
    )
    print(f"Done in {time.perf_counter() - t0:.1f}s (client wall time).", flush=True)

    parsed = _first_json_object(assistant)
    if parsed is not None and "explanation" in parsed and "question" in parsed:
        print(json.dumps(parsed, indent=2, ensure_ascii=True))
    else:
        print(assistant)
