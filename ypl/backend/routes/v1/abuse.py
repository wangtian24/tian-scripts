import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ypl.backend.abuse.utils import get_abuse_events
from ypl.backend.utils.json import json_dumps
from ypl.db.abuse import AbuseEventState, AbuseEventType

router = APIRouter(prefix="/admin")


@dataclass
class AbuseEventData:
    abuse_event_id: UUID
    created_at: datetime | None
    event_type: str
    event_details: dict
    user_id: str
    user_email: str
    user_name: str | None
    user_created_at: datetime | None
    state: str
    reviewed_at: datetime | None


@dataclass
class AbuseEventsResponse:
    events: list[AbuseEventData]
    has_more_rows: bool


@router.get("/abuse_events")
async def get_abuse_events_route(
    user_id: Annotated[str | None, Query(description="Optional User ID")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Number of events to return")] = 20,
    offset: Annotated[int, Query(ge=0, description="Number of events to skip")] = 0,
    event_type: Annotated[AbuseEventType | None, Query(description="Optional event type")] = None,
    state: Annotated[AbuseEventState | None, Query(description="Optional state")] = None,
) -> AbuseEventsResponse:
    try:
        events, has_more_rows = await get_abuse_events(
            user_id=user_id,
            limit=limit,
            offset=offset,
            event_type=event_type,
            state=state,
        )
        return AbuseEventsResponse(
            events=[
                AbuseEventData(
                    abuse_event_id=e.abuse_event_id,
                    created_at=e.created_at,
                    event_type=e.event_type.value,
                    event_details=e.event_details,
                    user_id=e.user.user_id,
                    user_email=e.user.email,
                    user_name=e.user.name,
                    user_created_at=e.user.created_at,
                    state=e.state.value,
                    reviewed_at=e.reviewed_at,
                )
                for e in events
            ],
            has_more_rows=has_more_rows,
        )
    except Exception as e:
        log_dict = {"message": f"Error listing abuse events: {str(e)}"}
        logging.exception(json_dumps(log_dict))
        # Avoid returning the error details, that might contain sensitive information.
        raise HTTPException(status_code=500, detail="Error listing abuse events; check logs for details.") from e
