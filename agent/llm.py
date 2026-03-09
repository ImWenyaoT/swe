"""LLM interface – thin wrapper around the OpenAI Chat Completions API."""

from __future__ import annotations

import os
from typing import Protocol


class LLMProtocol(Protocol):
    """Anything that takes a prompt string and returns a completion string."""

    def complete(self, prompt: str) -> str:
        ...


class OpenAILLM:
    """
    Wrapper around the OpenAI Chat Completions API.

    Reads the API key from the ``OPENAI_API_KEY`` environment variable
    (or the ``api_key`` constructor argument).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No OpenAI API key provided. Pass api_key= or set the "
                "OPENAI_API_KEY environment variable."
            )
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            from openai import OpenAI  # type: ignore[import]

            client_kwargs: dict = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use OpenAILLM. "
                "Install it with:  pip install openai"
            ) from exc

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
