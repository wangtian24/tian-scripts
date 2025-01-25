from fastapi import APIRouter, Query

from ypl.backend.payment.payment import (
    CashoutsHistoryResponse,
    PointTransactionsHistoryResponse,
    get_cashouts,
    get_points_transactions,
)

router = APIRouter()


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
