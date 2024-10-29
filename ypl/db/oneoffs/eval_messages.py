import uuid

from sqlalchemy import Connection, text
from tqdm import tqdm


def upgrade_eval_messages(connection: Connection) -> None:
    """Migrate evals to eval_messages."""

    # Get evals that have no message_evals.
    query = text(
        """
        SELECT e.eval_id, e.score_1, e.score_2, e.user_comment,
               m1.message_id as message_1_id, m2.message_id as message_2_id
        FROM evals e
        LEFT JOIN message_evals me ON e.eval_id = me.eval_id
        LEFT JOIN chat_messages m1 ON e.message_1_id = m1.message_id
        LEFT JOIN chat_messages m2 ON e.message_2_id = m2.message_id
        WHERE e.eval_type IN ('SELECTION', 'SLIDER_V0')
        AND me.message_eval_id IS NULL
        AND e.deleted_at IS NULL
    """
    )

    results = connection.execute(query).fetchall()

    # Create all message_evals
    for eval_row in tqdm(results, desc="Migrating evals"):
        message_ids = [eval_row.message_1_id, eval_row.message_2_id]
        scores = [eval_row.score_1, eval_row.score_2]

        for message_id, score in zip(message_ids, scores, strict=True):
            if message_id is not None:
                connection.execute(
                    text(
                        "INSERT INTO message_evals (message_eval_id, eval_id, message_id, score, user_comment) "
                        "VALUES (:message_eval_id, :eval_id, :message_id, :score, :user_comment)"
                    ),
                    {
                        "message_eval_id": uuid.uuid4(),
                        "eval_id": eval_row.eval_id,
                        "message_id": message_id,
                        "score": score,
                        "user_comment": eval_row.user_comment,
                    },
                )


def downgrade_eval_messages(connection: Connection) -> None:
    connection.execute(text("DELETE FROM message_evals"))
