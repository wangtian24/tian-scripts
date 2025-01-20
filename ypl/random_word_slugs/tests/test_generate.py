import pytest
from ypl.random_word_slugs.generate import (
    DEFAULT_NUMBER_OF_WORDS,
    RandomSlugConfigError,
    format_slug,
    generate_slug,
    get_total_unique_slugs,
)


def test_generate_slug_default() -> None:
    """Test generate_slug with default parameters"""
    slug = generate_slug()
    parts = slug.split("-")
    assert len(parts) == DEFAULT_NUMBER_OF_WORDS
    assert all(part.isalpha() for part in parts)


def test_generate_slug_custom_words() -> None:
    """Test generate_slug with custom number of words"""
    num_words = 5
    slug = generate_slug(num_words)
    parts = slug.split("-")
    assert len(parts) == num_words


def test_generate_slug_formats() -> None:
    """Test generate_slug with different format options"""
    slug = generate_slug(options={"format": "snake", "seed": 42})
    assert "_" in slug
    assert "-" not in slug

    slug = generate_slug(options={"format": "camel", "seed": 42})
    assert "_" not in slug
    assert "-" not in slug
    assert slug[0].islower()
    assert any(c.isupper() for c in slug[1:])


def test_generate_slug_seed() -> None:
    """Test generate_slug with seed for reproducibility"""
    slug1 = generate_slug(options={"seed": 42})
    slug2 = generate_slug(options={"seed": 42})
    assert slug1 == slug2


def test_format_slug() -> None:
    """Test format_slug with different formats"""
    words = ["happy", "jumping", "kangaroo"]

    assert format_slug(words, "kebab") == "happy-jumping-kangaroo"
    assert format_slug(words, "snake") == "happy_jumping_kangaroo"
    assert format_slug(words, "camel") == "happyJumpingKangaroo"
    assert format_slug(words) == "happy-jumping-kangaroo"  # default format


def test_invalid_options() -> None:
    """Test invalid options raise appropriate errors"""
    with pytest.raises(RandomSlugConfigError):
        generate_slug(options={"format": "invalid"})  # type: ignore

    with pytest.raises(RandomSlugConfigError):
        generate_slug(options={"parts_of_speech": ["invalid"]})  # type: ignore

    with pytest.raises(RandomSlugConfigError):
        generate_slug(options={"categories": {"invalid": []}})

    with pytest.raises(RandomSlugConfigError):
        generate_slug(options={"seed": {}})  # type: ignore


def test_get_total_unique_slugs() -> None:
    """Test get_total_unique_slugs calculation"""
    # Test with default options
    total = get_total_unique_slugs()
    assert total > 0

    # Test with custom number of words
    total_custom = get_total_unique_slugs(number_of_words=4)
    assert total_custom > 0

    # Test with custom categories
    total_categories = get_total_unique_slugs(options={"categories": {"adjective": ["color"], "noun": ["animal"]}})
    assert total_categories > 0
