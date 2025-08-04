import torch
import asyncio
from transformers import AutoTokenizer
from pathlib import Path

class StoryGenerator:
    def __init__(self):
        # Get the directory where this file is located
        current_file = Path(__file__)  
        server_dir = current_file.parent.parent.parent  
        model_path = server_dir / "app" / "model" / "model.pt"
        tokenizer_path = server_dir / "app" / "model" / "tokenizer"
        
        print(f"Loading model from: {model_path}")
        
        self.model = torch.load(str(model_path))
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        print(f"✅ Model loaded with vocab size: {self.tokenizer.vocab_size}")
    
    async def generate_story_stream(self, prompt, max_tokens, temperature):
        try:
            # Convert prompt to tokens
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            
            # Generate tokens using model's generate method
            # ! Note: model.generate() returns just the NEW tokens, not including input
            generated_tokens = self.model.generate(
                input_ids, 
                max_tokens=max_tokens, 
                temperature=temperature
            )
            
            # Stream each token with a small delay
            for token_id in generated_tokens:
                # Decode single token
                token_text = self.tokenizer.decode([token_id.item()], skip_special_tokens=True)
                
                # Only yield non-empty tokens
                if token_text.strip():
                    yield token_text
                    
                # Small delay for streaming effect
                await asyncio.sleep(0.05)
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            yield f" [Error: {str(e)}]"
