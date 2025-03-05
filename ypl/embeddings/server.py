"""FastAPI server for embedding texts using a pre-trained model.

Run with:

    uvicorn ypl.embeddings.server:app --host

For more information, please see README.md.
"""
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

import google.cloud.logging
import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException, Response
from google.cloud import logging_v2, monitoring_v3
from google.cloud.monitoring_v3.types import TimeInterval
from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Constants for custom metric types.
REQUEST_COUNT_METRIC = "custom.googleapis.com/embedding_server/request_count"
REQUEST_DURATION_METRIC = "custom.googleapis.com/embedding_server/request_duration"
NUM_STRINGS_METRIC = "custom.googleapis.com/embedding_server/num_strings"
TOTAL_LENGTH_METRIC = "custom.googleapis.com/embedding_server/total_length"

GAUGE = 1
DELTA = 2
CUMULATIVE = 3

API_KEY: str
DEVICE: str
GCP_PROJECT: str
MODEL_NAME: str
PROJECT_NAME: str
logger: logging_v2.logger.Logger
monitoring_client: monitoring_v3.MetricServiceClient
model: SentenceTransformer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global API_KEY, DEVICE, GCP_PROJECT, MODEL_NAME, PROJECT_NAME, logger, model, monitoring_client

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")

    log_client = google.cloud.logging.Client()
    logger = log_client.logger("embedding-server", labels={"model_name": MODEL_NAME})

    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.log_text("API_KEY environment variable is not set. Aborting.")
        raise RuntimeError("API_KEY environment variable is not set. Aborting.")
    else:
        API_KEY = api_key
        logger.log_text(f"Starting with API key: {API_KEY[0:4]}...")

    # Retrieve the GCP project ID.
    gcp_project = os.getenv("GCP_PROJECT")
    if not gcp_project:
        logger.log_text("GCP_PROJECT environment variable is not set. Aborting.")
        raise RuntimeError("GCP_PROJECT environment variable is not set. Aborting.")
    GCP_PROJECT = gcp_project
    PROJECT_NAME = f"projects/{GCP_PROJECT}"

    # Initialize Google Cloud Monitoring client.
    monitoring_client = monitoring_v3.MetricServiceClient()

    # Load and initialize the sentence transformer model.
    model = SentenceTransformer(MODEL_NAME)
    model = model.to(DEVICE)
    logger.log_text(f"[{PROJECT_NAME}] Serving model {MODEL_NAME} on {DEVICE}.")

    # Hand control back to FastAPI so it can process requests
    yield


# Initialize logging and FastAPI app.
app = FastAPI(lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str]
    model_name: str | None = None


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


def record_metric(
    metric_type: str,
    value: float,
    value_type: str = "double",
    labels: dict | None = None,
    cumulative: bool = False,
    start_time: float | None = None,
    end_time: float | None = None,
) -> None:
    """Records a custom metric to Google Cloud Monitoring.

    Args:
        metric_type: The full metric type name.
        value: The metric value.
        value_type: The type of the metric value ('double' or 'int').
        labels: Optional dictionary of label keys and values.
        cumulative: Whether the metric is cumulative (True) or gauge (False).
        start_time: Optional start time for gauge metrics.
        end_time: Optional end time for gauge and required for cumulative metrics.
    """
    if labels is None:
        labels = {}

    series = monitoring_v3.TimeSeries()
    series.metric.type = metric_type
    for key, val in labels.items():
        series.metric.labels[key] = str(val)
    series.resource.type = "global"
    series.metric_kind = CUMULATIVE if cumulative else GAUGE

    point = monitoring_v3.Point()
    if value_type == "int":
        point.value.int64_value = int(value)
    else:
        point.value.double_value = float(value)

    # Set interval based on metric type and provided times
    if cumulative:
        if end_time is None:
            raise ValueError("end_time must be provided for cumulative metrics")
        end_seconds = int(end_time)
        end_nanos = int((end_time - end_seconds) * 10**9)
        point.interval = TimeInterval(end_time=Timestamp(seconds=end_seconds, nanos=end_nanos))
    else:  # Gauge
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time must be provided for gauge metrics")
        start_seconds = int(start_time)
        start_nanos = int((start_time - start_seconds) * 10**9)
        end_seconds = int(end_time)
        end_nanos = int((end_time - end_seconds) * 10**9)
        point.interval = TimeInterval(
            start_time=Timestamp(seconds=start_seconds, nanos=start_nanos),
            end_time=Timestamp(seconds=end_seconds, nanos=end_nanos),
        )
    series.points.append(point)

    try:
        monitoring_client.create_time_series(name=PROJECT_NAME, time_series=[series])
    except Exception as error:
        logger.log_text(f"Failed to record metric {metric_type}: {error}")


def increment_request_count() -> None:
    """Increments the request count metric."""
    record_metric(REQUEST_COUNT_METRIC, 1, value_type="int", cumulative=True, end_time=time.time())


def update_embed_metrics(start_time: float, request: EmbedRequest) -> None:
    """Updates metrics for an embed request.

    Records:
      - Total number of requests.
      - Duration of the request.
      - Number of strings in the request.
      - Total length of all strings in the request.
    """
    end_time = time.time()
    elapsed = end_time - start_time
    num_strings = len(request.texts)
    total_length = sum(len(text) for text in request.texts)

    increment_request_count()
    record_metric(REQUEST_DURATION_METRIC, elapsed, start_time=start_time, end_time=end_time)
    record_metric(NUM_STRINGS_METRIC, num_strings, value_type="int", start_time=start_time, end_time=end_time)
    record_metric(TOTAL_LENGTH_METRIC, total_length, value_type="int", start_time=start_time, end_time=end_time)


def extract_sequence_lengths(texts: list[str]) -> list[int]:
    """Extracts the sequence lengths of the given texts."""
    return [len(model.tokenizer.encode(text, add_special_tokens=True)) for text in texts]


def estimate_batch_size(
    sequence_lengths: list[int],
    default_batch_size: int = 32,
    max_memory: int = 805_306_368,
    model_name: str | None = None,
) -> int:
    if max_memory <= 0:
        raise ValueError("max_memory must be greater than 0")

    if default_batch_size < 1:
        raise ValueError("default_batch_size must be greater than 0")

    if not sequence_lengths:
        return default_batch_size

    if "BAAI/bge-m3" == model_name:
        # BGE-M3 model memory usage estimation, obtained by
        # fitting a linear model with design matrix:
        # X = np.column_stack([
        #     np.ones_like(B),  # 1
        #     B * T             # B*T
        # ])
        #
        # mem = -71_670 + 45_052 * batch_size * max_token_count

        result = default_batch_size
        result = (max_memory - 71_670) // (45_052 * max(sequence_lengths))
        return 1 if result < 1 else result

    return default_batch_size


# TODO(amin): Switch to ypl/backend/routes/api_auth.py.
@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest, response: Response, x_api_key: str = Header(None)) -> EmbedResponse:
    """Handles embed requests by computing embeddings for the provided texts.

    Also tracks custom metrics for each request.
    """
    start_time = time.time()
    if x_api_key != API_KEY:
        raise HTTPException(status_code=404, detail="Not Found")

    model_name = request.model_name if request.model_name else MODEL_NAME
    if model_name != MODEL_NAME:
        logger.log_text(f"Model {request.model_name} not found.")
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} was not found.")

    sequence_lengths = extract_sequence_lengths(request.texts)

    # Let's warn the user if the sequence length is too long.
    overlong = [len for len in sequence_lengths if len > model.tokenizer.model_max_length]

    # Add a warning header if overlong sequences exist
    if overlong:
        response.headers[
            "X-Warning"
        ] = f"Some inputs exceed max length ({model.tokenizer.model_max_length}): {overlong}"

    try:
        ebs = estimate_batch_size(sequence_lengths, max_memory=805_306_368, model_name=model_name)
        logger.log_text(f"Estimating batch size for {len(request.texts)} texts with lengths {sequence_lengths}.")
        print(f"Estimated batch size: {ebs}")
        embeddings = cast(
            np.ndarray,
            model.encode(
                request.texts,
                device=DEVICE,
                convert_to_numpy=True,
                batch_size=ebs,
            ),
        ).tolist()
        return EmbedResponse(embeddings=embeddings)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        update_embed_metrics(start_time, request)


@app.get("/status")
async def status(x_api_key: str = Header(None)) -> dict:
    """Returns the server status if the API key is valid."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=404, detail="Not Found")

    if model:
        return {"status": "OK"}
    else:
        raise HTTPException(status_code=503, detail="Model not ready")
