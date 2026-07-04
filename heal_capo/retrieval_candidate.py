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
        sections = [str(self.instruction).strip()]
        if selected:
            sections.append(
                "Retrieved reference examples follow. Use them as local evidence, "
                "not as a class-prior vote."
            )
            for index, row in enumerate(selected, 1):
                sections.append(
                    f"Reference {index}:\n{row['input']}\n"
                    f"Reference label: {row['output']}"
                )
        sections.append(f"Current record:\n{text}\nAnswer:")
        return "\n\n".join(section for section in sections if section)
