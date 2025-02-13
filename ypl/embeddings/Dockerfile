FROM python:3.11-slim AS base

# Set the working directory
WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency files first to leverage Docker cache
COPY pyproject.toml poetry.lock /app/

# Install Poetry
RUN pip install poetry

# Update the lock file and install dependencies
RUN poetry lock && poetry install --no-root

# Copy the rest of the application files.
COPY ypl /app/ypl
RUN chmod +x /app/ypl/embeddings/entrypoint.sh

# Expose the port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/ypl/embeddings/entrypoint.sh", "service"]
