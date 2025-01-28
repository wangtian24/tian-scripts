from dotenv import load_dotenv
from sqlalchemy import text

from ypl.backend.db import get_engine

# To run this script, set the right environment variables in .env
# and run `python -m ypl.db.oneoffs.update_supported_mime_types`

# Models that support only images
IMAGE_ONLY_MODELS = [
    "gpt-4o-mini",
    "gpt-4o-2024-01-18",
    "claude-3-opus-20240229",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4o",
    "claude-3-opus-20240229",
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "pixtral-12b-2409",
]

# Models that support both images and PDFs
IMAGE_AND_PDF_MODELS = [
    "gemini-2.0-flash-exp",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "gemini-1.5-pro",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro-exp-0827",
    "gemini-1.5-pro-002",
    "gemini-1.5-pro-exp-0827",
]

# First update language models with image-only support
image_only_models = f"""
UPDATE language_models
SET supported_attachment_mime_types = ARRAY['image/*']
WHERE internal_name in (
    {','.join(f"'{model}'" for model in IMAGE_ONLY_MODELS)}
);
"""

# Then update models that support both image and PDF
image_and_pdf_models = f"""
UPDATE language_models
SET supported_attachment_mime_types = ARRAY['image/*', 'application/pdf']
WHERE internal_name in (
    {','.join(f"'{model}'" for model in IMAGE_AND_PDF_MODELS)}
);
"""


def update_language_models() -> None:
    load_dotenv()
    with get_engine().connect() as conn:
        conn.execute(text(image_only_models))
        conn.execute(text(image_and_pdf_models))
        conn.commit()


if __name__ == "__main__":
    update_language_models()
