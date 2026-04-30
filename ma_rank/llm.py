from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
import requests

PROJECT_ENV = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(PROJECT_ENV, override=True)


class LLMClient:
    def __init__(self, provider: str | None = None):
        self.provider = (provider or os.getenv("MA_RANK_LLM_PROVIDER", "gemini")).strip().lower()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.gemini_fallback_models = [
            model.strip()
            for model in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash-lite,gemini-2.0-flash").split(",")
            if model.strip()
        ]
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def enabled(self) -> bool:
        return self.provider in {"ollama", "gemini", "openai"}

    def extract_json(self, system: str, prompt: str) -> dict[str, Any]:
        if not self.enabled():
            raise RuntimeError("No LLM provider configured. Set MA_RANK_LLM_PROVIDER to 'gemini' or 'ollama'.")
        if self.provider == "ollama":
            text = self._ollama(system, prompt)
        elif self.provider == "gemini":
            text = self._gemini(system, prompt)
        elif self.provider == "openai":
            text = self._openai(system, prompt)
        else:
            raise RuntimeError(f"Unsupported LLM provider: {self.provider}")
        return _coerce_json(text)

    def _ollama(self, system: str, prompt: str) -> str:
        chat_payload = {
            "model": self.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
        }
        response = requests.post(f"{self.ollama_base_url}/api/chat", json=chat_payload, timeout=90)
        if response.status_code == 404:
            first_error = response.text
            generate_payload = {
                "model": self.ollama_model,
                "stream": False,
                "prompt": f"{system}\n\n{prompt}",
                "format": "json",
            }
            response = requests.post(f"{self.ollama_base_url}/api/generate", json=generate_payload, timeout=90)
            if response.status_code == 404:
                raise RuntimeError(
                    f"Ollama returned 404 for model '{self.ollama_model}'. "
                    f"Installed model names must match OLLAMA_MODEL exactly. "
                    f"First response: {first_error}. Second response: {response.text}"
                )
        response.raise_for_status()
        payload = response.json()
        if "response" in payload:
            return payload.get("response", "")
        return payload.get("message", {}).get("content", "")

    def _gemini(self, system: str, prompt: str) -> str:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=api_key)
        models = [self.gemini_model] + [m for m in self.gemini_fallback_models if m != self.gemini_model]
        last_error = None
        for model in models:
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=f"{system}\n\n{prompt}",
                        config=types.GenerateContentConfig(response_mime_type="application/json"),
                    )
                    if model != self.gemini_model:
                        print(f"Gemini fallback model used: {model}")
                    return response.text or ""
                except Exception as exc:
                    last_error = exc
                    retryable = _is_retryable_gemini_error(exc)
                    if not retryable:
                        raise
                    wait_seconds = 2 ** attempt
                    print(f"Gemini model {model} unavailable ({exc}); retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)
            print(f"Gemini model {model} failed after retries; trying next fallback if available.")
        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}") from last_error

    def _openai(self, system: str, prompt: str) -> str:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return response.choices[0].message.content or ""


def _coerce_json(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("LLM returned an empty response.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"LLM response did not contain a JSON object: {text[:300]}")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM JSON response must be an object.")
    return data


def _is_retryable_gemini_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(token in text for token in ["429", "500", "502", "503", "504", "unavailable", "high demand"])
