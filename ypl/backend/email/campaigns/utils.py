import logging
import os


def load_html_template(filename: str, fallback_template: str) -> str:
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
    except FileNotFoundError:
        return fallback_template


def load_html_wrapper() -> str:
    """Load the HTML template from file."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "html_wrapper.html")
    with open(template_path, encoding="utf-8") as f:
        return f.read()
