import torch
import asyncio
from transformers import AutoTokenizer
from pathlib import Path
from app.model.transformer import RNNLanguageModel

class StoryGenerator:
    def __init__(self):
        current_file = Path(__file__)  
        server_dir = current_file.parent.parent.parent  
        model_path = server_dir / "app" / "model" / "model.pt"
        tokenizer_path = server_dir / "app" / "model" / "tokenizer"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"Loading model from: {model_path}")
        
        # Load tokenizer first to get vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        
        # Create model instance with same parameters as training
        self.model = RNNLanguageModel(
            embed_dim=64,      # Same as your training
            hidden_dim=128,    # Same as your training  
            vocab_size=self.tokenizer.vocab_size,
            num_head=8,        # Default or whatever you used
            key_dim=16,        # Same as your dk
            value_dim=16       # Same as your dv
        )
        
        # Load the saved state dictionary
        self.model.load_state_dict(torch.load(str(model_path), map_location=device))
        self.model.to(device)
        self.model.eval()
        
        print(f"✅ Model loaded with vocab size: {self.tokenizer.vocab_size}")
    
    async def generate_story_stream(self, prompt, max_tokens, temperature):
        try:
            # Convert prompt to tokens
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            input_ids = input_ids.to(self.model.device)  # Move to correct device
            
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
                #token_text = self.tokenizer.decode([token_id.item()], skip_special_tokens=True)
                
                # Only yield non-empty tokens
                if token_id.strip():
                    yield token_id
                    
                # Small delay for streaming effect
                await asyncio.sleep(0.05)
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            yield f" [Error: {str(e)}]"
