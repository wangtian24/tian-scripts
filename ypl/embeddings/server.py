import os
import time
from typing import cast

import google.cloud.logging
import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from google.cloud import monitoring_v3
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

# Initialize logging and FastAPI app.
app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = os.getenv("MODEL_NAME", "BAAI/bge-m3")

log_client = google.cloud.logging.Client()
logger = log_client.logger("embedding-server", labels={"model_name": model_name})

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.log_text("API_KEY environment variable is not set. Aborting.")
    raise RuntimeError("API_KEY environment variable is not set. Aborting.")
else:
    logger.log_text(f"Starting with API key: {API_KEY[0:4]}...")

# Retrieve the GCP project ID.
GCP_PROJECT = os.getenv("GCP_PROJECT")
if not GCP_PROJECT:
    logger.log_text("GCP_PROJECT environment variable is not set. Aborting.")
    raise RuntimeError("GCP_PROJECT environment variable is not set. Aborting.")
PROJECT_NAME = f"projects/{GCP_PROJECT}"

# Initialize Google Cloud Monitoring client.
monitoring_client = monitoring_v3.MetricServiceClient()

# Load and initialize the sentence transformer model.
model = SentenceTransformer(model_name)
model = model.to(device)
logger.log_text(f"[{PROJECT_NAME}] Serving model {model_name} on {device}.")


class EmbedRequest(BaseModel):
    texts: list[str]
    model_name: str = model_name


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


# TODO(amin): Switch to ypl/backend/routes/api_auth.py.
@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest, x_api_key: str = Header(None)) -> EmbedResponse:
    """Handles embed requests by computing embeddings for the provided texts.

    Also tracks custom metrics for each request.
    """
    start_time = time.time()
    if x_api_key != API_KEY:
        raise HTTPException(status_code=404, detail="Not Found")

    if request.model_name != model_name:
        logger.log_text(f"Model {request.model_name} not found.")
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} was not found.")
    try:
        embeddings = cast(np.ndarray, model.encode(request.texts, device=device, convert_to_numpy=True)).tolist()
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
