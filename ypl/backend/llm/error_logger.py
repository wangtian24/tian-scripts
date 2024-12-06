import logging

from sqlalchemy import select
from sqlmodel import Session

from ypl.backend.db import get_engine
from ypl.db.all_models import *  # noqa: F403
from ypl.db.language_models import LanguageModel, LanguageModelResponseStatus


class LanguageModelStatusLogger:
    def log(
        self,
        status: LanguageModelResponseStatus,
        *,
        internal_name: str | None = None,
    ) -> None:
        raise NotImplementedError


class DefaultLanguageModelStatusLogger(LanguageModelStatusLogger):
    def log(
        self,
        status: LanguageModelResponseStatus,
        *,
        internal_name: str | None = None,
    ) -> None:
        logging.info(f"Language model status: {status}")


class DatabaseLanguageModelStatusLogger(LanguageModelStatusLogger):
    def log(
        self,
        status: LanguageModelResponseStatus,
        *,
        internal_name: str | None = None,
    ) -> None:
        """If internal_name is specified, it will be used to replace the ID in `status`."""
        with Session(get_engine()) as session:
            if internal_name is not None:
                status.language_model_id = (
                    session.exec(
                        select(LanguageModel).where(LanguageModel.internal_name == internal_name)  # type: ignore
                    )
                    .one()
                    .language_model_id
                )

            session.add(status)
            session.commit()
