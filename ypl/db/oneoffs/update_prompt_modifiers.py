from datetime import UTC, datetime

from sqlalchemy import Connection
from sqlalchemy.orm import Session

from ypl.db.chats import ModifierCategory, PromptModifier

modifiers = {
    "style_short": {
        "prev": {
            "name": "style_concise",
            "text": "Be as concise as possible; try to limit the response to 50 words or fewer.",
        },
        "new": {
            "name": "Shorter",
            "text": (
                "Generate a response that is concise, to the point, and free of unnecessary elaboration. "
                "Keep sentences brief while maintaining clarity and completeness. "
                "Omit redundant words and avoid excessive details."
            ),
        },
    },
    "style_formal": {
        "prev": {
            "name": "style_formal",
            "text": "Provide a clear and well-structured response, with a formal tone.",
        },
        "new": {
            "name": "More formal",
            "text": (
                "Generate a response with a neutral yet authoritative tone, ensuring clarity and professionalism. "
                "Use precise language, complete sentences, and avoid casual expressions, contractions, or slang. "
                "Ensure grammatical accuracy, and avoid personal opinions unless explicitly requested. "
                "Structure responses logically, using well-defined paragraphs."
            ),
        },
    },
    "style_explanatory": {
        "prev": {
            "name": "style_explanatory",
            "text": "Make your response educational, providing examples and explanations as needed.",
        },
        "new": {
            "name": "More structured",
            "text": (
                "Generate a response that is well-organized, logically structured, and prioritizes for readability. "
                "Present information in a clear format with distinct sections, using headings, bullet points, or "
                "numbered lists where appropriate."
            ),
        },
    },
    "style_casual": {
        "new": {
            "name": "More casual",
            "text": (
                "Generate a response which has a friendly, approachable, and conversational tone. "
                "Keep the response informal yet clear, using natural language and contractions. "
                "Feel free to add a touch of personality and humour while keeping the information accurate and useful. "
                "Use shorter sentences with everyday words (as applicable). "
                "No need for long explanations."
            ),
        },
    },
    "style_tabular": {
        "new": {
            "name": "Tabular",
            "text": (
                "If the information to be presented can be logically structured into rows and columns, generate a "
                "response in a well-organized table with clear headers and concise data points. "
                "If the information does lend itself to be structured as a table, start the response with "
                "“The response does not lend itself to be formatted in a tabular format. "
                "I will try to present it in a structured manner.” and follow it up in a structured manner."
            ),
        },
    },
    "style_summarize": {
        "new": {
            "name": "Summarize",
            "text": (
                "Generate a concise summary that captures the most important information while removing unnecessary "
                "details. "
                "Focus on key takeaways, core insights, and essential points. Avoid excessive details. "
                "Structure the response with the headline “Key Takeaways” and the rest as a bulleted list summarizing "
                "the key points. "
                "Each bullet to have a 2-3 word heading followed by a short paragraph."
            ),
        },
    },
}

now = datetime.now(UTC)


def upgrade_modifiers(connection: Connection) -> None:
    with Session(connection) as session:
        for modifier_data in modifiers.values():
            new_data = modifier_data["new"]

            if "prev" in modifier_data:
                # Update existing modifier
                existing_modifier = (
                    session.query(PromptModifier)
                    .filter(PromptModifier.name == modifier_data["prev"]["name"], PromptModifier.deleted_at.is_(None))
                    .first()
                )

                if existing_modifier:
                    existing_modifier.name = new_data["name"]
                    existing_modifier.text = new_data["text"]
                    existing_modifier.modified_at = now
                    existing_modifier.category = ModifierCategory.style
                    print(f"Updated modifier: {existing_modifier}")
            else:
                # Create new modifier
                new_modifier = PromptModifier(name=new_data["name"], text=new_data["text"])
                new_modifier.category = ModifierCategory.style
                session.add(new_modifier)
                print(f"Created new modifier: {new_modifier}")

        # Delete any modifiers not in the new data
        new_names = [modifier_data["new"]["name"] for modifier_data in modifiers.values()]
        existing_modifiers = (
            session.query(PromptModifier)
            .filter(PromptModifier.deleted_at.is_(None))
            .filter(PromptModifier.name.notin_(new_names))
            .all()
        )

        for modifier in existing_modifiers:
            modifier.deleted_at = now
            print(f"Marked as deleted (not in new data): {modifier}")

        session.commit()


def downgrade_modifiers(connection: Connection) -> None:
    with Session(connection) as session:
        for modifier_data in modifiers.values():
            new_data = modifier_data["new"]

            existing_modifier = (
                session.query(PromptModifier)
                .filter(PromptModifier.name == new_data["name"], PromptModifier.deleted_at.is_(None))
                .first()
            )

            if existing_modifier:
                existing_modifier.category = ModifierCategory.style
                if "prev" in modifier_data:
                    # Revert to previous values
                    existing_modifier.name = modifier_data["prev"]["name"]
                    existing_modifier.text = modifier_data["prev"]["text"]
                    existing_modifier.modified_at = now
                    existing_modifier.deleted_at = None
                    print(f"Reverted modifier: {existing_modifier}")
                else:
                    # If no previous version exists, mark as deleted
                    existing_modifier.deleted_at = now
                    print(f"Marked as deleted: {existing_modifier}")

        # Delete any modifiers not in the prev data
        prev_names = [modifier_data["prev"]["name"] for modifier_data in modifiers.values() if "prev" in modifier_data]
        existing_modifiers = (
            session.query(PromptModifier)
            .filter(PromptModifier.deleted_at.is_(None))
            .filter(PromptModifier.name.notin_(prev_names))
            .all()
        )

        for modifier in existing_modifiers:
            modifier.deleted_at = now
            print(f"Marked as deleted (not in prev data): {modifier}")

        session.commit()
