import logging
from dataclasses import dataclass
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from sqlmodel import select

from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.point_transactions import PointsActionEnum, PointTransaction

router = APIRouter()


@dataclass
class PointTransactionResponse:
    transaction_id: str
    user_id: str
    point_delta: int
    action_type: PointsActionEnum
    action_details: dict
    created_at: datetime
    deleted_at: datetime | None


@dataclass
class PointTransactionsHistoryResponse:
    transactions: list[PointTransactionResponse]


@router.get("/points/transactions")
async def get_points_transactions(
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
        List of point transactions with their details
    """
    try:
        async with get_async_session() as session:
            stmt = (
                select(PointTransaction)
                .where(PointTransaction.user_id == user_id)
                .order_by(PointTransaction.created_at.desc())  # type: ignore
                .offset(offset)
                .limit(limit)
            )

            result = await session.execute(stmt)
            transactions = result.scalars().all()

            log_dict = {
                "message": "Points transactions found",
                "user_id": user_id,
                "transactions_count": len(transactions),
                "limit": limit,
                "offset": offset,
            }
            logging.info(json_dumps(log_dict))

            return PointTransactionsHistoryResponse(
                transactions=[
                    PointTransactionResponse(
                        transaction_id=str(tx.transaction_id),
                        user_id=tx.user_id,
                        point_delta=tx.point_delta,
                        action_type=tx.action_type,
                        action_details=tx.action_details,
                        created_at=tx.created_at,
                        deleted_at=tx.deleted_at,
                    )
                    for tx in transactions
                ]
            )

    except Exception as e:
        log_dict = {
            "message": "Error getting points transactions",
            "user_id": user_id,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
