#!/bin/bash
# Start query service and wait for it to finish
docker compose up query-service --exit-code-from query-service

# If query service exited successfully (exit code 0), start other services
if [ $? -eq 0 ]; then
    echo "Query service completed successfully. Starting other services..."
    docker compose --profile post-query up -d
else
    echo "Query service failed. Other services will not be started."
    exit 1
fi