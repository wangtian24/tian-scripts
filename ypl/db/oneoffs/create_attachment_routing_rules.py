from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import text

from ypl.backend.db import get_engine

# To run this script, set the right environment variables in .env
# and run `python -m ypl.db.oneoffs.create_attachment_routing_rules`

rules = [
    {
        "is_active": True,
        "z_index": 900000,
        "source_category": "image",
        "destination": "Anthropic/claude-3-opus-20240229",
        "target": "ACCEPT",
        "probability": 1.0,
    },
    {
        "is_active": True,
        "z_index": 900000,
        "source_category": "image",
        "destination": "Google/gemini-1.5-pro-002",
        "target": "ACCEPT",
        "probability": 1.0,
    },
    {
        "is_active": True,
        "z_index": 900000,
        "source_category": "image",
        "destination": "OpenAI/gpt-4o-mini",
        "target": "ACCEPT",
        "probability": 1.0,
    },
    {
        "is_active": True,
        "z_index": 800000,
        "source_category": "image",
        "destination": "Google/gemini-1.5-pro-exp-0827",
        "target": "ACCEPT",
        "probability": 1.0,
    },
    {
        "is_active": True,
        "z_index": 900000,
        "source_category": "image",
        "destination": "Google/gemini-1.5-flash-8b",
        "target": "ACCEPT",
        "probability": 1.0,
    },
    {
        "is_active": True,
        "z_index": 900000,
        "source_category": "image",
        "destination": "Google/gemini-2.0-flash-exp",
        "target": "ACCEPT",
        "probability": 1.0,
    },
    {
        "is_active": True,
        "z_index": 800000,
        "source_category": "image",
        "destination": "OpenAI/gpt-4o-mini-2024-07-18",
        "target": "ACCEPT",
        "probability": 1.0,
    },
    {
        "is_active": True,
        "z_index": 800000,
        "source_category": "image",
        "destination": "OpenAI/gpt-4o-2024-08-06",
        "target": "ACCEPT",
        "probability": 1.0,
    },
    {
        "is_active": True,
        "z_index": 800000,
        "source_category": "image",
        "destination": "OpenAI/gpt-4o-2024-05-13",
        "target": "ACCEPT",
        "probability": 1.0,
    },
    {
        "is_active": True,
        "z_index": 900000,
        "source_category": "image",
        "destination": "OpenAI/gpt-4o",
        "target": "ACCEPT",
        "probability": 1.0,
    },
    {
        "is_active": True,
        "z_index": 900000,
        "source_category": "image",
        "destination": "Anthropic/claude-3-5-sonnet-20241022",
        "target": "ACCEPT",
        "probability": 1.0,
    },
    {
        "is_active": True,
        "z_index": 800000,
        "source_category": "image",
        "destination": "Anthropic/claude-3-5-sonnet-20240620",
        "target": "ACCEPT",
        "probability": 1.0,
    },
    {
        "is_active": True,
        "z_index": 600000,
        "source_category": "image",
        "destination": "Mistral AI/pixtral-12b-2409",
        "target": "ACCEPT",
        "probability": 1.0,
    },
]


def create_image_routing_rules() -> None:
    load_dotenv()
    with get_engine().connect() as conn:
        # Delete existing rules with raw SQL
        conn.execute(text("DELETE FROM routing_rules WHERE source_category = 'image'"))

        # Batch insert with raw SQL
        values = [{"routing_rule_id": str(uuid4()), **rule} for rule in rules]
        conn.execute(
            text(
                """
                INSERT INTO routing_rules
                (routing_rule_id, is_active, z_index, source_category, destination, target, probability)
                VALUES
                (:routing_rule_id, :is_active, :z_index, :source_category, :destination, :target, :probability)
            """
            ),
            values,
        )
        conn.commit()


if __name__ == "__main__":
    create_image_routing_rules()
