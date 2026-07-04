from __future__ import annotations

import hashlib

from heal_capo.core import PromptCandidate


class RetrievalPromptCandidate(PromptCandidate):
    def __init__(self, source, retriever):
        super().__init__(
            instruction=source.instruction,
            examples=[],
            parent_ids=list(source.parent_ids),
            candidate_id=source.candidate_id,
            metadata=dict(source.metadata),
        )
        self.retriever = retriever
        self.trace = {}

    def render(self, text: str) -> str:
        selected = self.retriever.select(text)
        key = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        self.trace[key] = selected
        prompt = PromptCandidate(
            instruction=self.instruction,
            examples=[{"input": row["input"], "output": row["output"]} for row in selected],
            parent_ids=list(self.parent_ids),
            candidate_id=self.candidate_id,
            metadata=dict(self.metadata),
        )
        return prompt.render(text)
