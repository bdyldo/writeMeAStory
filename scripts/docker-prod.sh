#!/bin/bash
# Production Docker environment script

echo "🚀 Starting Story Generator in Production Mode..."

# Build and start production environment
docker-compose --profile prod up --build app-prod

echo "✅ Production server started at http://localhost:8000"
echo "🛑 Press Ctrl+C to stop"