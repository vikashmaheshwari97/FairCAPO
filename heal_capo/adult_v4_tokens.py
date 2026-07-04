from __future__ import annotations

import warnings


class TokenCounter:
    """Use the cached model tokenizer, with a deterministic regex fallback."""

    def __init__(self, model_id: str):
        self.model_id = str(model_id or "")
        self._tokenizer = None
        self._attempted = False

    def _load(self):
        if self._attempted:
            return self._tokenizer
        self._attempted = True
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, local_files_only=True
            )
        except Exception as exc:
            warnings.warn(
                f"Could not load tokenizer {self.model_id!r}; using regex token count: {exc}",
                RuntimeWarning,
            )
            self._tokenizer = None
        return self._tokenizer

    def count(self, text: str) -> int:
        value = str(text or "")
        tokenizer = self._load()
        if tokenizer is not None:
            return len(tokenizer.encode(value, add_special_tokens=False))
        import re
        return len(re.findall(r"\w+|[^\w\s]", value, flags=re.UNICODE))
