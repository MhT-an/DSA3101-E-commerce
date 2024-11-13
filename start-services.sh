#!/bin/bash

# Function to check if query service completed successfully
check_query_completion() {
    if [ -f "./shared-data/.ready" ]; then
        return 0
    else
        return 1
    fi
}

echo "Starting query service..."

# Run query service and capture its exit code
docker compose up query-service --exit-code-from query-service
QUERY_EXIT_CODE=$?

if [ $QUERY_EXIT_CODE -eq 0 ]; then
    echo "Query service container exited successfully"
    
    # Additional check for completion file
    if check_query_completion; then
        echo "Data processing verified complete. Starting notebook and app services..."
        docker compose --profile post-query up -d
        
        # Verify other services started
        if [ $? -eq 0 ]; then
            echo "All services are now running. You can check their status with 'docker compose ps'"
        else
            echo "Error: Failed to start notebook and app services"
            exit 1
        fi
    else
        echo "Error: Query service container exited but data processing may not have completed"
        echo "Check the query service logs for errors: docker compose logs query-service"
        exit 1
    fi
else
    echo "Error: Query service failed with exit code $QUERY_EXIT_CODE"
    echo "Check the query service logs: docker compose logs query-service"
    exit 1
fi