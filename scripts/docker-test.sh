#!/bin/bash
# Test Docker environment script

echo "🧪 Running Story Generator Tests in Docker..."

# Build and run tests
docker-compose --profile test up --build app-test

echo "✅ Tests completed!"