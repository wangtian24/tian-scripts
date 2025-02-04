import logging
import os

from html2text import HTML2Text


def load_html_template(filename: str) -> str:
    """Load an email template from a file.

    Args:
        filename: Name of the template file to load
    Returns:
        The template contents as a string
    """
    # Points to ypl/backend/email/campaigns/templates
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    template_path = os.path.join(template_dir, filename)

    logging.info(f"Loading template from: {template_path}")

    try:
        with open(template_path, encoding="utf-8") as f:
            content = f.read()
            return content
    except FileNotFoundError as e:
        raise ValueError(f"Template file not found: {template_path}") from e


def load_html_wrapper() -> str:
    """Load the HTML template from file."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "html_wrapper.html")
    with open(template_path, encoding="utf-8") as f:
        return f.read()


def html_to_plaintext(html_content: str) -> str:
    converter = HTML2Text()
    # Prevent markdown conversion
    converter.ignore_emphasis = True
    converter.ignore_links = True
    converter.ignore_images = True
    converter.ignore_tables = True

    # Keep basic text structure
    converter.single_line_break = False
    converter.body_width = 0  # Disable wrapping to let email client handle it
    converter.wrap_list_items = True

    return str(converter.handle(html_content)).strip()
