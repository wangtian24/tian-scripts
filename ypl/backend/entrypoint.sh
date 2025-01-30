#!/bin/bash
set -e

if [ "$1" = "service" ]; then
    exec uvicorn ypl.backend.server:app --host 0.0.0.0 --port 8080 --loop uvloop --workers 2
elif [ -n "$1" ]; then
    exec python -m ypl.cli "$@"
else
    echo "Error: No command specified"
    exit 1
fi