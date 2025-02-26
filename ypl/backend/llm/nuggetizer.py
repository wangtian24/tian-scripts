"""Module for Yupp's nuggetizer implementation."""


from nuggetizer.core.types import Nugget, NuggetAssignMode, Request, ScoredNugget
from nuggetizer.models.async_nuggetizer import AsyncNuggetizer

from ypl.backend.prompts import (
    NUGGETIZER_ASSIGN_PROMPT,
    NUGGETIZER_ASSIGN_PROMPT_SUPPORT_GRADE_2,
    NUGGETIZER_ASSIGN_USER_PROMPT,
    NUGGETIZER_ASSIGN_USER_PROMPT_SUPPORT_GRADE_2,
    NUGGETIZER_CREATE_PROMPT,
    NUGGETIZER_CREATE_USER_PROMPT,
    NUGGETIZER_SCORE_PROMPT,
    NUGGETIZER_SCORE_USER_PROMPT,
)


class YuppNuggetizer(AsyncNuggetizer):
    """Yupp's implementation of AsyncNuggetizer with conversation-aware prompts."""

    def _create_nugget_prompt(self, request: Request, start: int, end: int, nuggets: list[str]) -> list[dict[str, str]]:
        """Create the prompt for nugget creation with conversation history."""
        context = "\n".join([f"[{i+1}] {doc.segment}" for i, doc in enumerate(request.documents[start:end])])
        messages = [
            {"role": "system", "content": NUGGETIZER_CREATE_PROMPT},
            {
                "role": "user",
                "content": NUGGETIZER_CREATE_USER_PROMPT.format(
                    max_nuggets=self.creator_max_nuggets,
                    query=request.query.text,
                    context=context,
                    nuggets=nuggets,
                    nugget_count=len(nuggets),
                ),
            },
        ]
        return messages

    def _create_score_prompt(self, query: str, nuggets: list[Nugget]) -> list[dict[str, str]]:
        """Create the prompt for scoring nuggets with conversation history."""
        messages = [
            {"role": "system", "content": NUGGETIZER_SCORE_PROMPT},
            {
                "role": "user",
                "content": NUGGETIZER_SCORE_USER_PROMPT.format(
                    query=query,
                    nuggets=[nugget.text for nugget in nuggets],
                ),
            },
        ]
        return messages

    def _create_assign_prompt(self, query: str, context: str, nuggets: list[ScoredNugget]) -> list[dict[str, str]]:
        """Create the prompt for assigning nuggets with conversation history."""
        messages = [
            {
                "role": "system",
                "content": NUGGETIZER_ASSIGN_PROMPT_SUPPORT_GRADE_2
                if self.assigner_mode == NuggetAssignMode.SUPPORT_GRADE_2
                else NUGGETIZER_ASSIGN_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    NUGGETIZER_ASSIGN_USER_PROMPT_SUPPORT_GRADE_2
                    if self.assigner_mode == NuggetAssignMode.SUPPORT_GRADE_2
                    else NUGGETIZER_ASSIGN_USER_PROMPT
                ).format(query=query, context=context, nuggets=[nugget.text for nugget in nuggets]),
            },
        ]
        return messages
