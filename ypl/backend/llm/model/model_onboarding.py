import logging
from datetime import datetime

from huggingface_hub import HfApi, ModelCard
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
                api = HfApi()
                model_info = api.model_info(model.internal_name)
                downloads = model_info.downloads
                likes = model_info.likes

                card = ModelCard.load(model.internal_name)
                mmlu_pro_score = None
                if isinstance(card.data.eval_results, list):
                    for eval_result in card.data.eval_results:
                        if eval_result.dataset_name.startswith("MMLU-PRO") and eval_result.metric_name == "accuracy":
                            mmlu_pro_score = eval_result.metric_value
                            logging.info(f"MMLU-PRO score found: {mmlu_pro_score}")
                            break
                else:
                    logging.info(f"card.data.eval_results is not a list. Type: {type(card.data.eval_results)}")

                # TODO: Implement proper model verification logic
                # For now, we'll use the MMLU-Pro score for verification
                # TODO: Change these magic numbers to some configurable values as business rules
                if mmlu_pro_score is not None and mmlu_pro_score > 50 and downloads > 1000 and likes > 100:
                    model.status = LanguageModelStatusEnum.VERIFIED_PENDING_ACTIVATION
                    model.modified_at = datetime.utcnow()
                    logging.info(
                        f"Model {model.name} ({model.internal_name}) validated successfully "
                        f"and set to VERIFIED_PENDING_ACTIVATION. MMLU-Pro score: {mmlu_pro_score}, "
                        f"downloads: {downloads}, likes: {likes}"
                    )
                else:
                    logging.info(
                        f"Model {model.name} ({model.internal_name}) not validated. "
                        f"MMLU-Pro score: {mmlu_pro_score}, downloads: {downloads}, likes: {likes}. "
                        f"Keeping status as SUBMITTED."
                    )
            except Exception as e:
                # TODO: Implement alerting
                logging.info(f"Model {model.name} ({model.internal_name}) validation failed: {str(e)}")

        session.commit()


if __name__ == "__main__":
    verify_onboard_submitted_models()
