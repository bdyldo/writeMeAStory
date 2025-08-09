# Run backend using 'poetry run python -m app.main' when in the server directory
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import socketio
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables 
load_dotenv()

# ! socket_manager depends on dotenv()
# import socketio server instance, which is either cpu based or wrapped using gpu cloud app Modal
from .core.socket_manager import sio
import uvicorn

# initialize FastAPI application
app = FastAPI()

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

    # StaticFiles mount static assets to browser from a directory on disk via HTTP
    # This serving must always happen without controls that can be given when using FileResponse
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

    @app.api_route("/{path:path}", methods=["GET", "HEAD"])
    async def serve_frontend(path: str = ""):
        # Excludes requests that start with "api/", "socket.io/", or OpenAPI docs routes to avoid overriding backend routes.
        if path.startswith(("api/", "socket.io/", "docs", "redoc", "openapi.json")):
            return {"error": "API route not found", "path": path}

        # Serve index.html for all other routes to enable client-side routing
        index_file = frontend_dist_path / "index.html"
        if index_file.exists():
            # Sends a single specific file as the HTTP response. For serving one file, not a whole folder.
            # Can be dynamically returned under different conditions, better than StaticFiles
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
    # Dynamically determine the correct import path based on execution context
    # When run from server/ directory: use "app.main:socket_app"  
    # When run from root/ directory (Docker): use "server.app.main:socket_app"
    import sys
    import os
    
    # Check if we're running from the server directory or root directory
    current_dir = os.getcwd()
    if current_dir.endswith('/server') or 'server' not in sys.modules:
        app_path = "app.main:socket_app"  # Local execution from server/
    else:
        app_path = "server.app.main:socket_app"  # Docker execution from root/
    
    # Run the application using Uvicorn server using host
    uvicorn.run(
        app_path, host=host, port=port, reload=(environment != "PROD")
    )
