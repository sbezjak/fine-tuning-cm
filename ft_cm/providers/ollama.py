"""Ollama `/api/generate` adapter, carried from the sibling repos and trimmed.

Kept for the free local judge/eval lane (the `ollama` marker). The fine-tuned
model itself is served through the MLX provider, not Ollama, but a live Ollama
check is a cheap sanity backend and keeps the marker discipline consistent with
P0-P5.
"""

from __future__ import annotations

import httpx

from ft_cm.providers.base import Provider


class OllamaError(RuntimeError):
    """Ollama returned an error or an unexpected payload."""


class OllamaProvider(Provider):
    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.system = system

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.system:
            payload["system"] = self.system
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
        if "response" not in data:
            raise OllamaError(f"no 'response' field in Ollama reply: {data!r}")
        return data["response"]
