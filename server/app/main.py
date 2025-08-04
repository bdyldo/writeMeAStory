from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
from app.core.socket_manager import sio
import uvicorn

# initialize FastAPI application
app = FastAPI()

# configure CORS middleware to wrap around all request handlers.
# ! This allows specific origins / methods / headers to access backend API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Link Socket.IO server which comes from /core/socket_manager.py with FastAPI
socket_app = socketio.ASGIApp(sio, app)

if __name__ == "__main__":
    uvicorn.run("app.main:socket_app", host="0.0.0.0", port=8000, reload=True)