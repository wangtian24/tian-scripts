from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query

from ypl.backend.payment.payment import (
    CashoutsHistoryResponse,
    PointTransactionsHistoryResponse,
    adjust_points,
    get_cashouts,
    get_points_transactions,
)
from ypl.backend.utils.soul_utils import SoulPermission, validate_permissions

router = APIRouter()


async def validate_manage_points(
    x_creator_email: str | None = Header(None, alias="X-Creator-Email"),
) -> None:
    """Validate that the user has MANAGE_POINTS permission."""
    await validate_permissions([SoulPermission.MANAGE_CASHOUT], x_creator_email)


@router.get("/admin/points/transactions")
async def get_points_transactions_route(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(50, description="Maximum number of transactions to return", ge=1, le=100),
    offset: int = Query(0, description="Number of transactions to skip", ge=0),
) -> PointTransactionsHistoryResponse:
    """Get points transaction history for a user.

    Args:
        user_id: The ID of the user to get transactions for
        limit: Maximum number of transactions to return (default: 50, max: 100)
        offset: Number of transactions to skip for pagination (default: 0)

    Returns:
        List of point transactions with their details and a flag indicating if more rows exist
    """
    return await get_points_transactions(user_id=user_id, limit=limit, offset=offset)


@router.get("/admin/cashouts")
async def get_cashouts_route(
    limit: int = Query(50, description="Maximum number of cashouts to return", ge=1, le=100),
    offset: int = Query(0, description="Number of cashouts to skip", ge=0),
    user_id: str | None = Query(None, description="Filter cashouts for a specific user"),
) -> CashoutsHistoryResponse:
    """Get cashout transaction history.

    Args:
        limit: Maximum number of cashouts to return (default: 50, max: 100)
        offset: Number of cashouts to skip for pagination (default: 0)
        user_id: Optional user ID to filter cashouts for a specific user

    Returns:
        List of cashout transactions with their details and a flag indicating if more rows exist
    """
    return await get_cashouts(limit=limit, offset=offset, user_id=user_id)


@router.post("/admin/points/adjust", dependencies=[Depends(validate_manage_points)])
async def adjust_points_route(
    user_id: str = Query(..., description="User ID"),
    point_delta: int = Query(..., description="Points delta"),
    reason: str = Query(..., description="Reason for the adjustment"),
) -> UUID:
    """Adjust points for a user."""
    return await adjust_points(user_id=user_id, point_delta=point_delta, reason=reason)
