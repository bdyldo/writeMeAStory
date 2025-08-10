#!/bin/bash
# Development Docker environment startup script

echo "🐳 Starting Story Generator in Development Mode..."

# Build and start development environment
docker-compose --profile dev up --build app-dev

echo "✅ Development server started at http://localhost:8000"
echo "📁 Source code is mounted for hot reload"
echo "🛑 Press Ctrl+C to stop"