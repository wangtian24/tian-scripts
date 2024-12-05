from datetime import datetime

from sqlalchemy import Connection, delete, select, update
from sqlmodel import Session

from ypl.db.chats import PromptModifier


def _add_prompt_modifiers(connection: Connection, modifiers: list[tuple[str, str, str]]) -> None:
    with Session(connection) as session:
        for category, name, text in modifiers:
            # Check if modifier already exists
            existing_id = session.exec(
                select(PromptModifier.prompt_modifier_id).where(
                    PromptModifier.category == category, PromptModifier.name == name
                )
            ).fetchone()

            if existing_id:
                # Undelete if it exists
                session.exec(
                    update(PromptModifier)
                    .where(PromptModifier.prompt_modifier_id == existing_id[0])
                    .values(deleted_at=None)
                )
                continue

            prompt_modifier = PromptModifier(category=category, name=name, text=text)
            session.add(prompt_modifier)

        session.commit()


def add_initial_prompt_modifiers(connection: Connection) -> None:
    """Add initial prompt modifiers to the database."""

    # Source: https://docs.google.com/document/d/1ygAog8fvDY_nSd8u1Ae1pKrzjmhTlsNFvWsLvZHQNU8/edit?tab=t.0
    modifiers = [
        # Length/Conciseness
        ("length", "length_concise", "Limit the response to 50 words or fewer."),
        ("length", "length_detailed", "Provide a detailed explanation, with step-by-step instructions if applicable."),
        ("length", "length_medium", "Limit the response to 3-4 sentences with about 10 words each."),
        # Style/Tone
        ("style", "style_friendly", "Use a conversational, friendly tone."),
        ("style", "style_professional", "Maintain a neutral and professional tone."),
        ("style", "style_humorous", "Use a humorous and light-hearted tone."),
        ("style", "style_coaching", "Explain in a warm coaching tone"),
        # Complexity
        (
            "complexity",
            "complexity_high_school",
            "Respond as a high school teacher explaining a concept to her student",
        ),
        ("complexity", "complexity_college", "Assume the audience is college-educated."),
        ("complexity", "complexity_thorough", "Be thorough in your answer and explain in clear lucid steps."),
        ("complexity", "complexity_first_principles", "Explain by simplifying the concepts to first principles"),
        # Formatting
        ("formatting", "formatting_bullet_points", "Use bullet points and numbered lists"),
        ("formatting", "formatting_markdown_bullets", "Use markdown with bullet points in hierarchical form"),
        ("formatting", "formatting_structured_prose", "Write in highly structured prose."),
        ("formatting", "formatting_table", "Use a table when necessary to compare and contrast"),
        (
            "formatting",
            "formatting_mixed",
            "Alternate between small paragraphs of 2-3 sentences, lists/bullets, and tables wherever appropriate.",
        ),
    ]

    _add_prompt_modifiers(connection, modifiers)


def add_simplified_prompt_modifiers(connection: Connection) -> None:
    """Add simplified prompt modifiers to the database."""

    modifiers = [
        (
            "style",
            "style_concise",
            "Be as concise as possible; try to limit the response to 50 words or fewer.",
        ),
        (
            "style",
            "style_explanatory",
            "Make your response educational, providing examples and explanations as needed.",
        ),
        (
            "style",
            "style_formal",
            "Provide a clear and well-structured response, with a formal tone.",
        ),
    ]

    _add_prompt_modifiers(connection, modifiers)


def remove_prompt_modifiers(connection: Connection) -> None:
    with Session(connection) as session:
        session.exec(delete(PromptModifier))


def soft_remove_prompt_modifiers(connection: Connection) -> None:
    """Mark all prompt modifiers as deleted."""
    now = datetime.now()
    with Session(connection) as session:
        session.exec(update(PromptModifier).where(PromptModifier.deleted_at.is_(None)).values(deleted_at=now))
