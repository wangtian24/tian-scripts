#!/bin/bash
set -e

exec uvicorn ypl.webhooks.server:app --host 0.0.0.0 --port 8081 