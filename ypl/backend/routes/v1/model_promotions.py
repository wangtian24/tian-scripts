import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from ypl.backend.llm.promotions import (
    activate_promotion,
    create_promotion,
    deactivate_promotion,
    get_active_model_promotions,
    get_all_model_promotions,
)
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.soul_utils import SoulPermission, validate_permissions
from ypl.db.model_promotions import ModelPromotion

router = APIRouter()


class ModelPromotionCreationRequest(BaseModel):
    language_model_name: str
    promo_start_date: datetime | None = None  # the default is today
    promo_end_date: datetime | None = None  # the default is today + 7 days
    promo_strength: float


async def validate_manage_model_promotions(
    x_creator_email: str | None = Header(None, alias="X-Creator-Email"),
) -> None:
    """Validate that the user has MANAGE_MODEL_PERFORMANCE permission."""
    await validate_permissions([SoulPermission.MANAGE_MODEL_PERFORMANCE], x_creator_email)


@router.post("/admin/model_promotions/create", dependencies=[Depends(validate_manage_model_promotions)])
async def model_promotion_create(request: ModelPromotionCreationRequest) -> ModelPromotion:
    try:
        # Create the model promotion
        promotion = await create_promotion(
            model_name=request.language_model_name,
            start_date=request.promo_start_date,
            end_date=request.promo_end_date,
            promo_strength=request.promo_strength,
        )

        # Log the creation
        logging.info(
            json_dumps(
                {
                    "message": f"Created model promotion for: {request.language_model_name}",
                    "promotion_id": str(promotion.promotion_id),
                    "language_model_id": str(promotion.language_model_id),
                    "start_date": promotion.promo_start_date.isoformat() if promotion.promo_start_date else "NULL",
                    "end_date": promotion.promo_end_date.isoformat() if promotion.promo_end_date else "NULL",
                    "strength": promotion.promo_strength,
                }
            )
        )
        return promotion
    except Exception as e:
        log_dict = {"message": f"Error creating model promotion - {str(e)}"}
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/admin/model_promotions/all_promotions")
async def model_promotions_all_promotions() -> list[tuple[str, ModelPromotion]]:
    """Get all model promotions."""
    try:
        return await get_all_model_promotions()
    except Exception as e:
        log_dict = {"message": f"Error getting all model promotions - {str(e)}"}
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/admin/model_promotions/active_promotions")
async def model_promotions_active_promotions() -> list[tuple[str, ModelPromotion]]:
    """Get all currently active model promotions."""
    try:
        return get_active_model_promotions()
    except Exception as e:
        log_dict = {"message": f"Error getting active model promotions - {str(e)}"}
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/admin/model_promotions/activate", dependencies=[Depends(validate_manage_model_promotions)])
async def model_promotion_activate(promotion_id: UUID) -> str:
    try:
        await activate_promotion(promotion_id)
        logging.info(
            json_dumps({"message": f"Activated model promotion {promotion_id}", "promotion_id": str(promotion_id)})
        )
        return "success"
    except HTTPException:
        raise
    except Exception as e:
        log_dict = {"message": f"Error activating model promotion - {str(e)}"}
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/admin/model_promotions/deactivate", dependencies=[Depends(validate_manage_model_promotions)])
async def model_promotion_deactivate(promotion_id: UUID) -> str:
    try:
        await deactivate_promotion(promotion_id)
        logging.info(
            json_dumps({"message": f"Deactivated model promotion {promotion_id}", "promotion_id": str(promotion_id)})
        )
        return "success"
    except HTTPException:
        raise
    except Exception as e:
        log_dict = {"message": f"Error deactivating model promotion - {str(e)}"}
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
