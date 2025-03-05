#!/bin/bash
#
# NAME
#     service.sh - A script to launch the embedding server.
#
# SYNOPSIS
#     ./service.sh service [--port PORT]
#
# DESCRIPTION
#     This script is designed to run the embedding server using uvicorn.
#
#     By default, the script will run on port 80. You can optionally
#     specify a custom port by passing the "--port" parameter followed by
#     the desired port number.
#
# EXAMPLES
#     1) Run the service on the default port (80):
#          ./service.sh service
#
#     2) Run the service on port 8080:
#          ./service.sh service --port 8080
#
#     3) Any unsupported command or parameter will result in an error.
#
set -e

# Default port
PORT=80

if [ "$1" = "service" ]; then
    # Shift off the "service" positional parameter
    shift

    # Parse any optional parameters
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            --port)
                PORT="$2"
                shift
                shift
                ;;
            *)
                echo "Unknown parameter: $1"
                exit 1
                ;;
        esac
    done

    # Now run uvicorn with the desired port
    exec poetry run uvicorn ypl.embeddings.server:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --loop uvloop \
        --workers 1
else
    echo "Error: Only 'service' command is supported"
    exit 1
fi
