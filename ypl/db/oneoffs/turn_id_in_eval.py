import pandas as pd
import requests
from sqlalchemy import Connection, text


def download_and_load_csv(bucket_name: str, blob_name: str) -> pd.DataFrame:
    url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
    response = requests.get(url)
    response.raise_for_status()

    local_path = f"/tmp/{blob_name.split('/')[-1]}"
    with open(local_path, "wb") as f:
        f.write(response.content)

    return pd.read_csv(local_path, names=["eval_id", "turn_id"])


def add_turn_id_to_evals(connection: Connection) -> None:
    """Add turn_id to evals."""

    # Get turn_id directly from chat_messages, if available.
    query = text(
        """
        UPDATE evals e
        SET turn_id = cm.turn_id
        FROM message_evals me
        JOIN chat_messages cm ON me.message_id = cm.message_id
        WHERE e.eval_id = me.eval_id
        AND e.turn_id IS NULL;
        """
    )
    connection.execute(query)

    # Early exit if all evals already have turn_id.
    check_query = text("SELECT COUNT(*) FROM evals WHERE turn_id IS NULL;")
    result = connection.execute(check_query).scalar()
    if result == 0:
        return

    # If not available, get turn_id from a recent backup of the database.
    create_temp_table = text(
        """
        CREATE TEMP TABLE temp_eval_to_turn (
            eval_id uuid,
            turn_id uuid
        );
        """
    )
    connection.execute(create_temp_table)

    prod_df = download_and_load_csv("yupp-open", "prod_evals_20241122.csv")
    staging_df = download_and_load_csv("yupp-open", "staging_evals_20241122.csv")
    combined_df = pd.concat([prod_df, staging_df], ignore_index=True)

    insert_query = text(
        """
        INSERT INTO temp_eval_to_turn (eval_id, turn_id)
        VALUES (:eval_id, :turn_id)
        """
    )
    records = combined_df.to_dict("records")
    connection.execute(insert_query, records)

    update_from_temp = text(
        """
        UPDATE evals e
        SET turn_id = te.turn_id
        FROM temp_eval_to_turn te
        WHERE e.eval_id = te.eval_id
        AND e.turn_id IS NULL;
        """
    )
    connection.execute(update_from_temp)

    drop_temp_table = text("DROP TABLE IF EXISTS temp_eval_to_turn;")
    connection.execute(drop_temp_table)
