"""
Modal app for GPU-accelerated story generation.
Deploy with: modal deploy modal_app.py
"""

import modal
from pathlib import Path
from app.schemas.story import StoryRequest, StoryResponse

# Modal configuration
app = modal.App("story-generator")

# Create a custom Docker-Like environment with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch",
        "transformers", 
        "fastapi",
        "pydantic"
    ])
)

# Create a cloud based file system inside Modal to attach files needed GPU instances 
MODEL_VOLUME = modal.Volume.from_name("story-model", create_if_missing=True)

# Define a class-based deployment container that runs on a GPU with Transformer model and dependencies
@app.cls(
    image=image,
    gpu=modal.gpu.A10G(),  # Use A10G GPU - good balance of performance/cost
    volumes={"/model": MODEL_VOLUME},
    container_idle_timeout=300,  # Keep container warm for 300 seconds
)

class StoryGenerator:
    
    def __init__(self):
        """Initialize the model on Modal's GPU"""
        import torch
        import sys
        sys.path.append("/model")
        
        # Import the transformer module - check multiple possible locations
        import importlib.util
        import os
        
        # Check what files exist in the volume
        print("Files in /model:")
        for root, dirs, files in os.walk("/model"):
            for file in files:
                print(os.path.join(root, file))
        
        # Load transformer
        transformer_path = "/model/transformer.py"
        
        if not transformer_path:
            raise FileNotFoundError(f"Could not find transformer.py")
            
        print(f"Loading transformer from: {transformer_path}")
        
        spec = importlib.util.spec_from_file_location("transformer", transformer_path)
        transformer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transformer_module)
        RNNLanguageModel = transformer_module.RNNLanguageModel
        
        from transformers import AutoTokenizer
        
        model_path = Path("/model/model.pt")
        tokenizer_path = Path("/model/tokenizer")
        
        print(f"Loading model from: {model_path}")
        print(f"Loading tokenizer from: {tokenizer_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        
        # Create model with same parameters as training
        self.model = RNNLanguageModel(
            embed_dim=512, 
            hidden_dim=768,  
            vocab_size=self.tokenizer.vocab_size,
            num_head=6,  
            key_dim=128,  
            value_dim=128,  
        )
        
        # Load model weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(str(model_path), map_location=device))
        self.model.to(device)
        self.model.eval()
        
        print(f"✅ Model loaded on {device} with vocab size: {self.tokenizer.vocab_size}")

    @modal.method()
    def generate_tokens(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
        """Generate tokens for the given prompt"""
        import torch
        
        try:
            # Convert prompt to tokens
            input_ids = self.tokenizer.encode(
                prompt, add_special_tokens=False, return_tensors="pt"
            )
            input_ids = input_ids.to(next(self.model.parameters()).device)

            with torch.no_grad():
                output_tokens = self.model.generate(
                    input_ids, max_tokens=max_tokens, temperature=temperature
                )

            generated_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            return StoryResponse(
                success=True,
                generated_text=generated_text,
                tokens_generated=len(output_tokens)
            )
            
        except Exception as e:
            return StoryResponse(
                success=False,
                error=str(e),
                generated_text=""
            )

# Create a StoryGenerator instance
story_generator = StoryGenerator()

# Setting up the web endpoint for Modal to call using HTTP
@app.function(image=image)
@modal.web_endpoint(method="POST")
def generate_story_endpoint(request: StoryRequest) -> StoryResponse:
    """
    Web endpoint for story generation
    POST body: {"prompt": "Once upon a time", "max_tokens": 100, "temperature": 0.7}
    """
    if not request.prompt:
        return StoryResponse(success=False, error="No prompt provided", generated_text="")
    
    result = story_generator.generate_tokens.remote(
        request.prompt, 
        request.max_tokens, 
        request.temperature
    )
    return result