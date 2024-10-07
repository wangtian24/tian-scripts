import logging
from datetime import datetime

from sqlmodel import Session, select
from ypl.backend.db import get_engine
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum


def verify_onboard_submitted_models() -> None:
    """
    Verify and onboard submitted models.
    This function should be run periodically to check and update the status of submitted models.
    """
    with Session(get_engine()) as session:
        query = select(LanguageModel).where(
            LanguageModel.status == LanguageModelStatusEnum.SUBMITTED,
            LanguageModel.deleted_at.is_(None),  # type: ignore
        )
        submitted_models = session.exec(query).all()

        for model in submitted_models:
            try:
                # TODO: Implement model verification logic
                # For now, we'll just do a hardcoded check and set the status to VERIFIED_PENDING_ACTIVATION
                if model.name.lower().startswith("google"):
                    model.status = LanguageModelStatusEnum.VERIFIED_PENDING_ACTIVATION
                    model.modified_at = datetime.utcnow()
                    logging.info(
                        f"Model {model.name} ({model.internal_name}) validated successfully and set to ACTIVE."
                    )
                else:
                    logging.info(
                        f"Model {model.name} ({model.internal_name}) not validated. Keeping status as SUBMITTED."
                    )
            except Exception as e:
                # TODO: Implement alerting
                logging.info(f"Model {model.name} ({model.internal_name}) validation failed: {str(e)}")

        session.commit()


if __name__ == "__main__":
    verify_onboard_submitted_models()
