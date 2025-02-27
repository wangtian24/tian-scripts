#!/bin/bash
set -e

if [ "$1" = "service" ]; then
    exec poetry run uvicorn ypl.embeddings.server:app --host 0.0.0.0 --port 80 --loop uvloop --workers 1
else
    echo "Error: Only service command is supported"
    exit 1
fi
