# Run this using 'python3 -m tests.test_generator' inside server folder
# generate a small story in temperature 0.7
import asyncio
from core.story_generator import StoryGenerator

async def test():
    generator = StoryGenerator()
    
    async for token in generator.generate_story_stream(
        "Once upon a time", 
        max_tokens=200, 
        temperature=0.7
    ):
        print(token, end="", flush=True)

asyncio.run(test())