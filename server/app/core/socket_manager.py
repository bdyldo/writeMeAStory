import socketio
import asyncio
import os
from .story_generator import StoryGenerator
from .modal_story_generator import ModalStoryGenerator

# AsyncServer Enables async support, necessary for non-blocking operations like model inference and streaming
# async_mode='asgi': Integrates with ASGI apps
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")

# Choose generator based on environment variable
USE_MODAL = os.getenv("USE_MODAL", "false").lower() == "true"
if USE_MODAL:
    print("🚀 Using Modal GPU for story generation")
    story_generator = ModalStoryGenerator()
else:
    print("💻 Using local model for story generation")
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
    stream = data.get("stream", True)

    # ! Uses async for to yield tokens as they’re generated, not waiting for the whole story
    if stream:
        async for token in story_generator.generate_story_stream(
            prompt, max_tokens, temperature
        ):
            # Emit each token back to the client, sent through each 'story_token' event
            await sio.emit("story_token", {"content": token}, room=sid)
    else:
        # Collect all tokens from the async generator into a full text
        tokens = []
        async for token in story_generator.generate_story_stream(
            prompt, max_tokens, temperature
        ):
            tokens.append(token)
        generated_text = "".join(tokens)
        await sio.emit("story_token", {"content": generated_text}, room=sid)

    # Finished generation, emit 'story_complete' event
    await sio.emit("story_complete", {}, room=sid)
