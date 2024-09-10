#!/bin/bash
set -e

if [ "$1" = "service" ]; then
    exec uvicorn backend.server:app --host 0.0.0.0 --port 8080
elif [ -n "$1" ]; then
    exec python /app/cli.py "$@"
else
    echo "Error: No command specified"
    exit 1
fi