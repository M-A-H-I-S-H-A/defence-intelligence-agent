"""
Centralized LLM service wrapping the Groq API.
Every agent calls through this module so retries, error handling, and
structured JSON parsing behave consistently across the whole system.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


class LLMServiceError(Exception):
    """Raised when the LLM service fails after all retries."""


class LLMService:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMServiceError(
                "GROQ_API_KEY not found. Set it in your .env file."
            )
        self.client = Groq(api_key=key)
        self.model = model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        """
        Basic text completion. Returns the raw text response.
        Retries on transient failures with exponential backoff.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - intentional broad catch, we retry
                last_error = exc
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        raise LLMServiceError(
            f"LLM completion failed after {MAX_RETRIES} attempts: {last_error}"
        )

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """
        Completion that expects a JSON object back. Instructs the model to
        return ONLY JSON, then parses it. Strips markdown code fences if the
        model wraps the JSON anyway.
        """
        json_system_prompt = (
            f"{system_prompt}\n\n"
            "IMPORTANT: Respond with ONLY a valid JSON object. "
            "Do not include any preamble, explanation, or markdown formatting "
            "such as ```json fences. Return raw JSON only."
        )

        raw_text = self.complete(
            system_prompt=json_system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        cleaned = self._strip_json_fences(raw_text)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMServiceError(
                f"LLM did not return valid JSON. Raw response: {raw_text[:500]}"
            ) from exc

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text


# Singleton instance for convenience — agents can import this directly.
_llm_service_instance: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance