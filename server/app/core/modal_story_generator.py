"""
Modal-enabled story generator that calls GPU inference remotely
"""
import asyncio
import aiohttp
import modal
import os
from typing import AsyncGenerator


class ModalStoryGenerator:
    def __init__(self):
        """Initialize Modal client"""
        self.modal_app_name = "story-generator"
        
        # Try to get Modal endpoint URL from environment
        self.modal_url = os.getenv("MODAL_ENDPOINT_URL")
        
        if not self.modal_url:
            print("⚠️  MODAL_ENDPOINT_URL not set, will use Modal client")
            self.use_client = True
        else:
            print(f"✅ Using Modal HTTP endpoint: {self.modal_url}")
            self.use_client = False

    async def _call_modal_client(self, prompt: str, max_tokens: int, temperature: float):
        """Call Modal using the Python client (Fall back for HTTP))"""
        try:
            # Import Modal client
            app = modal.App.lookup(self.modal_app_name)
            StoryGenerator = app["StoryGenerator"]
            
            # Call the remote function
            result = StoryGenerator().generate_tokens.remote(
                prompt, max_tokens, temperature
            )
            return result
            
        except Exception as e:
            print(f"❌ Modal client error: {e}")
            return {
                "success": False, 
                "error": f"Modal client error: {str(e)}",
                "generated_text": ""
            }

    async def _call_modal_http(self, prompt: str, max_tokens: int, temperature: float):
        """Call Modal using HTTP endpoint"""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.modal_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}",
                            "generated_text": ""
                        }
                        
        # If exceeded 60 seconds for the request, we just abort it
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Request timed out",
                "generated_text": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"HTTP request error: {str(e)}",
                "generated_text": ""
            }

    async def generate_story_stream(self, prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
        """
        Generate story with streaming simulation
        This maintains the same interface as original StoryGenerator
        """
        try:
            print(f"🚀 Calling Modal for generation: prompt='{prompt[:50]}...', max_tokens={max_tokens}, temp={temperature}")
            
            # Call Modal service
            if self.use_client:
                result = await self._call_modal_client(prompt, max_tokens, temperature)
            else:
                result = await self._call_modal_http(prompt, max_tokens, temperature)
            
            if not result["success"]:
                yield f" [Error: {result['error']}]"
                return
                
            generated_text = result["generated_text"]
            print(f"✅ Generated {result.get('tokens_generated', 'unknown')} tokens")
            
            # Stream the text word by word to maintain the same UX
            words = generated_text.split()
            for i, word in enumerate(words):
                if i == 0:
                    yield word
                else:
                    yield " " + word
                
                # Delay text generation for streaming
                await asyncio.sleep(0.05)
                
        except Exception as e:
            print(f"❌ Modal generation error: {e}")
            yield f" [Error: {str(e)}]"