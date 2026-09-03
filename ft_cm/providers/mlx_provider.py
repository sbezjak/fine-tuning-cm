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
        self._label_ids: dict[str, list[int]] | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load

        self._model, self._tokenizer = load(
            self.model_name,
            adapter_path=self.adapter_path,
        )

    def _messages(self, prompt: str) -> list[dict]:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _render(self, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            self._messages(prompt), tokenize=False, add_generation_prompt=True
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

    def decision_score(self, prompt: str):
        """Read the model's confidence at the DECISION TOKEN instead of decoding a word.

        One forward pass over the rendered prompt; the logits at the LAST position are
        the model's distribution over the next (answer) token. Softmax them, sum the
        probability mass over each label's token-id family, and hand the per-label mass
        to the pure restricted-binary readout in `logprob`. Returns a `LabelScore`
        (P(unsafe), per-label mass, off-label mass). No generation - so it is cheaper
        than `generate` and reads exactly the position greedy would have sampled from.
        """
        self._ensure_loaded()
        import mlx.core as mx

        from ft_cm.logprob import build_label_token_ids, scores_from_label_probs

        if self._label_ids is None:
            self._label_ids = build_label_token_ids(self._tokenizer)

        ids = self._tokenizer.apply_chat_template(
            self._messages(prompt), add_generation_prompt=True
        )
        logits = self._model(mx.array([ids]))
        probs = mx.softmax(logits[0, -1, :].astype(mx.float32), axis=-1)
        label_probs = {
            label: float(sum(probs[i].item() for i in id_list))
            for label, id_list in self._label_ids.items()
        }
        return scores_from_label_probs(label_probs)
