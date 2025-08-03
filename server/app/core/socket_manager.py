import socketio
import asyncio
from .story_generator import StoryGenerator

# AsyncServer Enables async support, necessary for non-blocking operations like model inference and streaming
# async_mode='asgi': Integrates with ASGI apps
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")

# Create an instance of StoryGenerator for token generation output
story_generator = StoryGenerator()

# Socket.IO event handlers
# ! Missing Reconnection events still


# @sio.event is a decorator registers an event handler for the corresponding event
# same as using sio.on('connect', connect) - event and function name should be the same
@sio.event
async def connect(sid, environ):
    print(f"Client {sid} connected")


@sio.event
async def disconnect(sid):
    print(f"Client {sid} disconnected")


@sio.event
async def generate_story(sid, data):
    # The Python-SocketIO server automatically deserializes sent through JSON from frontend
    # into a Python dictionary (dict) before passing it into your event handler
    # second param is the fallback value if the key is not found
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 200)
    temperature = data.get("temperature", 0.7)

    # ! Uses async for to yield tokens as they’re generated, not waiting for the whole story
    async for token in story_generator.generate_story_stream(
        prompt, max_tokens, temperature
    ):
        # Emit each token back to the client, sent through each 'story_token' event
        await sio.emit("story_token", {"content": token}, room=sid)

    # Finished generation, emit 'story_complete' event
    await sio.emit("story_complete", {}, room=sid)
