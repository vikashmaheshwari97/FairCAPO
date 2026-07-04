from __future__ import annotations

from heal_capo.core import PromptPortfolio


class DeduplicatingPromptPortfolio(PromptPortfolio):
    def add(self, candidate, result=None) -> None:
        for index, existing in enumerate(self.candidates):
            if existing.candidate_id == candidate.candidate_id:
                self.candidates[index] = candidate
                break
        else:
            self.candidates.append(candidate)
        if result is not None:
            self.evaluations[candidate.candidate_id] = result
