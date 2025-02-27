FROM python:3.11-slim AS base

# Set the working directory
WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Download and install gcloud SDK
RUN wget https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    tar -xzf google-cloud-sdk.tar.gz && \
    rm google-cloud-sdk.tar.gz && \
    ./google-cloud-sdk/install.sh -q

# Add gcloud to PATH
ENV PATH="/app/google-cloud-sdk/bin/:$PATH"

# Copy the dependency files first to leverage Docker cache
COPY pyproject.toml poetry.lock /app/

# Install Poetry
RUN pip install poetry

# Update the lock file and install dependencies
RUN poetry lock && poetry install --no-root

# Copy the rest of the application files.
RUN mkdir -p /app/ypl   
COPY ./ypl/embeddings /app/ypl/embeddings
RUN chmod +x /app/ypl/embeddings/entrypoint.sh

# Expose the port
EXPOSE 80

# Set the entrypoint
ENTRYPOINT ["/app/ypl/embeddings/entrypoint.sh", "service"]
