# Run this using 'python3 -m tests.test_generator' inside server folder
import asyncio
from app.core.story_generator import StoryGenerator

async def test():
    generator = StoryGenerator()
    
    async for token in generator.generate_story_stream(
        "Once upon a time", 
        max_tokens=20, 
        temperature=0.7
    ):
        print(token, end="", flush=True)

asyncio.run(test())