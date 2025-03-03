import asyncio
import base64
import logging
from datetime import datetime
from io import BytesIO

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from PIL import ExifTags, Image
from ypl.backend.llm.provider.provider_clients import get_internal_provider_client
from ypl.backend.prompts import IMAGE_DESCRIPTION_PROMPT
from ypl.backend.utils.json import json_dumps
from ypl.db.attachments import TransientAttachment

EXIF_ORIENTATION_TAG = -1

IMAGE_DESC_MODEL = "gpt-4o"
IMAGE_DESC_MAX_TOKENS = 100


async def generate_image_description(file: TransientAttachment) -> dict[str, str]:
    start = datetime.now()
    image_bytes = file.file
    await asyncio.sleep(0)
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        messages: list[BaseMessage] = [
            SystemMessage(content=IMAGE_DESCRIPTION_PROMPT.format(file_name=file.filename)),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ]
            ),
        ]
        client = await get_internal_provider_client(IMAGE_DESC_MODEL, IMAGE_DESC_MAX_TOKENS)
        result = await client.ainvoke(messages)
        return {"description": str(result.content)}
    finally:
        logging.info(
            json_dumps(
                {
                    "message": "Attachments: Image description generated",
                    "duration_ms": datetime.now() - start,
                    "file_name": file.filename,
                }
            )
        )


def downsize_image(blob: bytes, size: tuple[int, int]) -> bytes:
    original_image = Image.open(BytesIO(blob))
    image = original_image.copy()
    image.thumbnail(size)
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes.getvalue()


def _get_exif_orientation_tag() -> int:
    global EXIF_ORIENTATION_TAG
    if EXIF_ORIENTATION_TAG == -1:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                EXIF_ORIENTATION_TAG = orientation
                break
    return EXIF_ORIENTATION_TAG


def maybe_rotate_image(image: Image.Image) -> Image.Image:
    tag = _get_exif_orientation_tag()
    try:
        exif = image._getexif()  # type: ignore
        if exif[tag] == 8:
            image = image.rotate(90, expand=True)
        elif exif[tag] == 3:
            image = image.rotate(180, expand=True)
        elif exif[tag] == 6:
            image = image.rotate(270, expand=True)
    except Exception:
        pass  # Image doesn't have EXIF data, or not rotated.
    return image
