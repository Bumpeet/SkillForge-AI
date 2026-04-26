"""
Hugging Face Inference Endpoints (Inference Toolkit) custom handler.

Deploy this file in the *root* of the Hub model repo (e.g. Bumpeet/qwen2.5-1.5b-adaptive-tutor-rl)
alongside the weights. The filename must be ``handler.py``.

Request body (JSON):
  - ``inputs`` (str, required): the fully formatted *user* message — your client must inject
    concept/mastery/difficulty/mistakes (same string you would pass as one ``user`` turn).
  - ``parameters`` (dict, optional): ``max_new_tokens`` (default 1024), ``max_time`` (seconds, default 180),
    ``temperature`` (0.2), ``do_sample`` (true), ``top_p``, ``repetition_penalty``, etc.

  ``max_time`` stops generation after that many *wall-clock* seconds (Transformers) so a run
  cannot hang unbounded. Increase for very slow hardware or long outputs.

This handler does *not* apply str.format on a template; it only wraps ``inputs`` in
``[{\"role\": \"user\", \"content\": ...}]`` and runs the model like ``infer.py``.

For OpenAI-compatible ``/v1`` APIs, use TGI/vLLM instead of the Inference Toolkit. See
``inference.py`` docstring for TEACHING_BACKEND options.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _default_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return torch.float16


def _to_device(
    batch: Dict[str, torch.Tensor], model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    # Works with single-GPU; device_map="auto" first parameter device.
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {k: v.to(dev) for k, v in batch.items()}


class EndpointHandler:
    def __init__(self, path: str = "", **kwargs: Any) -> None:
        if not path:
            raise ValueError("EndpointHandler requires a model directory path (Hub mount).")
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=_default_dtype(),
            device_map="auto",
        )
        self._model.eval()

    def __call__(self, data: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        user_text = data.get("inputs")
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError(
                "Request JSON must include non-empty string key 'inputs' (formatted user message)."
            )

        parameters = data.get("parameters") or {}
        if not isinstance(parameters, dict):
            parameters = {}

        max_new_tokens = int(parameters.get("max_new_tokens", 1024))
        temperature = float(parameters.get("temperature", 0.2))
        do_sample = bool(parameters.get("do_sample", True))
        top_p = parameters.get("top_p")
        max_time = parameters.get("max_time", 180.0)
        repetition_penalty = parameters.get("repetition_penalty")

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max(1, min(max_new_tokens, 4096)),
            "temperature": max(0.0, temperature),
            "do_sample": do_sample,
        }
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if max_time is not None and float(max_time) > 0:
            gen_kwargs["max_time"] = float(max_time)
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)

        cfg = self._model.config
        pad_id = getattr(cfg, "pad_token_id", None) or self._tokenizer.pad_token_id
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        eos = getattr(cfg, "eos_token_id", None) or self._tokenizer.eos_token_id
        if eos is not None:
            gen_kwargs["eos_token_id"] = eos

        messages = [{"role": "user", "content": user_text}]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        raw = self._tokenizer(prompt, return_tensors="pt")
        batch = _to_device(dict(raw), self._model)
        with torch.inference_mode():
            outputs = self._model.generate(**batch, **gen_kwargs)

        prompt_len = batch["input_ids"].shape[1]
        gen_ids = outputs[0, prompt_len:]
        assistant = self._tokenizer.decode(
            gen_ids, skip_special_tokens=True
        ).strip()

        return {"generated_text": assistant}
