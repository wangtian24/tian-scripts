import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ypl.backend.abuse.utils import get_abuse_events, review_abuse_event
from ypl.backend.utils.json import json_dumps
from ypl.db.abuse import AbuseEvent, AbuseEventState, AbuseEventType

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
    user_status: str
    state: str
    reviewed_at: datetime | None
    reviewed_by: str | None
    review_notes: str | None

    @classmethod
    def from_abuse_event(cls, event: AbuseEvent) -> "AbuseEventData":
        return cls(
            abuse_event_id=event.abuse_event_id,
            created_at=event.created_at,
            event_type=event.event_type.value,
            event_details=event.event_details,
            user_id=event.user.user_id,
            user_email=event.user.email,
            user_name=event.user.name,
            user_created_at=event.user.created_at,
            user_status=event.user.status.value,
            state=event.state.value,
            reviewed_at=event.reviewed_at,
            reviewed_by=event.reviewed_by,
            review_notes=event.review_notes,
        )


@dataclass
class AbuseEventsResponse:
    events: list[AbuseEventData]
    has_more_rows: bool


@router.get("/abuse_events/list")
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
            events=[AbuseEventData.from_abuse_event(e) for e in events],
            has_more_rows=has_more_rows,
        )
    except Exception as e:
        log_dict = {"message": f"Error listing abuse events: {str(e)}"}
        logging.exception(json_dumps(log_dict))
        # Avoid returning the error details, that might contain sensitive information.
        raise HTTPException(status_code=500, detail="Error listing abuse events; check logs for details.") from e


class AbuseEventReviewRequest(BaseModel):
    abuse_event_id: UUID
    reviewer: str
    notes: str | None = None


@router.post("/abuse_events/review")
async def review_abuse_event_route(request: AbuseEventReviewRequest) -> AbuseEventData:
    try:
        event = await review_abuse_event(request.abuse_event_id, request.reviewer, request.notes)
        if event is None:
            raise HTTPException(status_code=404, detail="Abuse event not found")
        return AbuseEventData.from_abuse_event(event)
    except Exception as e:
        logging.exception(json_dumps({"message": f"Error reviewing abuse event: {str(e)}"}))
        raise HTTPException(status_code=500, detail="Error reviewing abuse event; check logs for details.") from e
