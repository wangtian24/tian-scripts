from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select
from ypl.backend.db import get_async_session
from ypl.db.language_models import (
    LanguageModel,
    LanguageModelTaxonomy,
)


class TaxonomyTreeNode(BaseModel):
    taxo_label: str
    taxonomy_id: UUID
    taxo_path: str
    is_pickable: bool
    is_leaf_node: bool
    provider_count: int


def _join_path(components: list[str]) -> str:
    """
    Join path components with '/' until the last non-empty component.
    If A's path is a prefix of B, then A is an ancestor of B (parent and above)
    """
    # Find the last non-empty component
    last_non_empty_index = -1
    for i in range(len(components) - 1, -1, -1):
        if components[i]:
            last_non_empty_index = i
            break

    if last_non_empty_index == -1:
        return ""

    # Keep all components up to the last non-empty one, including empty ones in the middle
    result = "/".join(components[: last_non_empty_index + 1])

    # Add trailing slash unless all components are used
    if last_non_empty_index < len(components) - 1:
        result += "/"

    return result


async def _get_all_taxonomy() -> list[TaxonomyTreeNode]:
    async with get_async_session() as session:
        query = (
            select(LanguageModelTaxonomy, func.count(LanguageModel.language_model_id).label("provider_count"))  # type: ignore
            .outerjoin(LanguageModel)
            .where(
                LanguageModelTaxonomy.deleted_at.is_(None),  # type: ignore
            )
            .group_by(LanguageModelTaxonomy)  # type: ignore
        )
        results = (await session.exec(query)).all()
        return [
            TaxonomyTreeNode(
                **result[0].model_dump(exclude={"language_model_taxonomy_id"}),
                taxonomy_id=result[0].language_model_taxonomy_id,
                taxo_path=_join_path(
                    [
                        result[0].model_publisher or "",
                        result[0].model_family or "",
                        result[0].model_class or "",
                        result[0].model_version or "",
                        result[0].model_release or "",
                    ]
                ),
                provider_count=result[1],
            )
            for result in results
        ]


async def do_model_taxonomy_visualization() -> None:
    """Visualize model taxonomy tree"""
    nodes: list[TaxonomyTreeNode] = await _get_all_taxonomy()
    # Sort nodes by taxonomy_path
    nodes.sort(key=lambda node: node.taxo_path)

    # Print taxonomy tree
    stack: list[str] = []
    prev_path = ""

    print("=" * 100)
    print(f"{'Path':<35} {'Pickable':>10} {'Count':>8} {'Taxo Label':<30}")
    print("-" * 100)

    for node in nodes:
        if not node.taxo_path:
            continue

        # Determine the current path components
        current_path = node.taxo_path.rstrip("/")
        current_parts = current_path.split("/")

        # Compare with previous path to determine relationship
        if prev_path:
            prev_parts = prev_path.split("/")

            # Find common prefix length
            common_length = 0
            for i in range(min(len(prev_parts), len(current_parts))):
                if prev_parts[i] == current_parts[i]:
                    common_length += 1
                else:
                    break

            # Adjust stack based on relationship
            if common_length < len(stack):
                # Pop stack until we're at the right level
                stack = stack[:common_length]

            # If current path is deeper than previous, push to stack
            for i in range(common_length, len(current_parts)):
                stack.append(current_parts[i])
        else:
            # First item
            for part in current_parts:
                stack.append(part)

        # Display the path with proper indentation
        indent = "|   " * (len(stack) - 1)
        display_name = current_parts[-1] if current_parts else ""
        formatted_path = f"{indent}{display_name}"

        pickable = "*" if node.is_pickable else ""

        # Format provider count
        provider_count_display = ""
        if node.provider_count > 0:
            provider_count_display = str(node.provider_count)
        elif node.is_pickable and node.provider_count == 0:
            provider_count_display = "!! 0"

        print(f"{formatted_path:<35} {pickable:>10} {provider_count_display:>8} {node.taxo_label}")

        prev_path = current_path
