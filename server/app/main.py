# Run backend using 'poetry run python -m app.main' when in the server directory
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
from app.core.socket_manager import sio
import uvicorn
from dotenv import load_dotenv
import os

# initialize FastAPI application
app = FastAPI()

load_dotenv()

host = os.getenv("HOST", "127.0.0.1")
port = int(os.getenv("PORT", 8000))
front_end_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

# configure CORS middleware to wrap around all request handlers.
# ! This allows specific origins / methods / headers to access backend API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[front_end_url],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Link Socket.IO server which comes from /core/socket_manager.py with FastAPI
socket_app = socketio.ASGIApp(sio, app)

if __name__ == "__main__":
    # Run the application using Uvicorn server using localhost and port 8000
    uvicorn.run("app.main:socket_app", host=host, port=port, reload=True)