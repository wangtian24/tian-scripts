#!/bin/bash
set -e

exec uvicorn ypl.partner_payments.server.main:app --host 0.0.0.0 --port ${PORT:-8080} 