# Run backend using 'poetry run python -m app.main' when in the server directory
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import socketio
from app.core.socket_manager import sio
import uvicorn
from dotenv import load_dotenv
import os
from pathlib import Path

# initialize FastAPI application
app = FastAPI()

load_dotenv()

host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", 10000))

# Used for CORS configuration during dev stage
front_end_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
environment = os.getenv("STAGE", "PROD")

# Get paths for serving frontend, allows backend to serve static files
server_dir = Path(__file__).parent.parent  # Go up to server/
frontend_dist_path = server_dir.parent / "client" / "dist"

# Serve built frontend in production
if environment == "PROD" and frontend_dist_path.exists():
    print("🚀 Production mode: Serving frontend from backend")

    # Mount static assets (CSS, JS, images)
    app.mount(
        "/assets",
        StaticFiles(directory=str(frontend_dist_path / "assets")),
        name="assets",
    )

    # API routes for health and status check
    @app.get("/api/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "WriteMeAStory Backend",
            "environment": environment,
            "frontend_served": True,
        }

    @app.get("/api/status")
    async def status():
        return {
            "backend": "running",
            "frontend_path": str(frontend_dist_path),
            "frontend_exists": frontend_dist_path.exists(),
        }

    @app.get("/{path:path}")
    async def serve_frontend(path: str = ""):
        # Excludes requests that start with "api/", "socket.io/", or OpenAPI docs routes to avoid overriding backend routes.
        if path.startswith(("api/", "socket.io/", "docs", "redoc", "openapi.json")):
            return {"error": "API route not found", "path": path}

        # Serve index.html for all other routes to enable client-side routing
        index_file = frontend_dist_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        else:
            return {"error": "Frontend not built", "path": str(frontend_dist_path)}

    print("✅ Frontend routes configured")

else:
    print("🔧 Development mode")
    # configure CORS middleware to wrap around all request handlers.
    # ! This allows specific origins / methods / headers to access backend API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[front_end_url],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # For backend debugging in Dev using console
    @app.get("/api/health")
    async def health_check():
        return {"status": "healthy", "service": "WriteMeAStory Backend"}


# Link Socket.IO server which comes from /core/socket_manager.py with FastAPI
socket_app = socketio.ASGIApp(sio, app)

if __name__ == "__main__":
    # Run the application using Uvicorn server using localhost and port 8000
    uvicorn.run(
        "app.main:socket_app", host=host, port=port, reload=(environment != "PROD")
    )
