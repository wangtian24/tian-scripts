# Embeddings Server

This is a FastAPI-based server that provides an endpoint to embed a list of strings into vector embeddings.
## Requirements

- Python 3.11+
- Poetry
- Mamba

## Installation

1. Activate your mamba environment:
    ```bash
    mamba activate ys-dev
    ```

2. Install the required packages using Poetry:
    ```bash
    poetry install
    ```

## Running the Server

To start the server, run the following command from `yupp-mind/`:
```bash
./ypl/embeddings/entrypoint.sh service
```

If you prefer to run the server using Docker, which is the recommended approach, use the following command:
```bash
./ypl/embeddings/run_docker.sh
```

## API

### POST /embed

**Request:**
- `texts`: List of strings to embed.

**Response:**
- `embeddings`: List of vector embeddings (each embedding is a list of floats).

**Example:**

Request:
```bash
curl -X POST "http://0.0.0.0:8000/embed" -H "Content-Type: application/json" -d '{
    "texts": ["Hello, world!", "Yupp with Yapps is great!"]
}'
```

Response:
```json
{
    "embeddings": [
        [0.123, 0.456, ...],
        [0.789, 0.012, ...]
    ]
}
```

### GET /status

Check if the model is loaded and the service is ready.

**Example:**

Request:
```bash
curl -X GET "http://0.0.0.0:8000/status"
```

Response:
```json
{
    "status": "OK"
}
```