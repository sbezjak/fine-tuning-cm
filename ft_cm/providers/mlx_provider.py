"""MLX-LM backend: serves the base model, with or without a LoRA adapter.

This is the seam the before/after eval runs through. Constructing it with
`adapter_path=None` gives the untuned base model (the "before"); pointing
`adapter_path` at a trained adapter gives the tuned model (the "after"), and the
harness cannot tell them apart. The model + tokenizer load once per process and
are reused across the whole held-out slice.

Import of `mlx_lm` is deferred into `_ensure_loaded` so the module imports (and
the mocked test suite runs) on a machine without the training stack installed.
Deterministic by default: greedy decoding, so the held-out numbers are stable
across reruns.
"""

from __future__ import annotations

from pathlib import Path

from ft_cm.providers.base import Provider


class MLXProvider(Provider):
    def __init__(
        self,
        model: str,
        adapter_path: str | Path | None = None,
        system: str | None = None,
        max_tokens: int = 8,
    ) -> None:
        self.model_name = model
        self.adapter_path = str(adapter_path) if adapter_path is not None else None
        self.system = system
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load

        self._model, self._tokenizer = load(
            self.model_name,
            adapter_path=self.adapter_path,
        )

    def _render(self, prompt: str) -> str:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    async def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        from mlx_lm import generate as mlx_generate

        rendered = self._render(prompt)
        return mlx_generate(
            self._model,
            self._tokenizer,
            prompt=rendered,
            max_tokens=self.max_tokens,
            verbose=False,
        )
