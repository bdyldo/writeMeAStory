"""
Modal app for GPU-accelerated story generation.
Deploy with: modal deploy modal_app.py
"""

import modal
from pathlib import Path

# Modal configuration
app = modal.App("story-generator")

# Create a custom image with your dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch",
        "transformers", 
        "fastapi",
        "pydantic"
    ])
)

# Mount your model files to Modal
MODEL_VOLUME = modal.Volume.from_name("story-model", create_if_missing=True)

@app.cls(
    image=image,
    gpu=modal.gpu.A10G(),  # Use A10G GPU - good balance of performance/cost
    volumes={"/model": MODEL_VOLUME},
    container_idle_timeout=300,  # Keep container warm for 5 minutes
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
        
        # Try different possible paths for the transformer file
        possible_paths = [
            "/model/app/model/transformer.py",
            "/model/transformer.py", 
            "/model/app.model.transformer.py"
        ]
        
        transformer_path = None
        for path in possible_paths:
            if os.path.exists(path):
                transformer_path = path
                break
                
        if not transformer_path:
            raise FileNotFoundError(f"Could not find transformer.py in any of: {possible_paths}")
            
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
        
        # Create model with same parameters as your training
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
            return {
                "success": True,
                "generated_text": generated_text,
                "tokens_generated": len(output_tokens)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "generated_text": ""
            }

# Expose the model as a web endpoint
@app.function(
    image=image,
    schedule=modal.Cron("0 8 * * *"),  # Optional: warm up daily at 8 AM
)
def keep_warm():
    """Optional function to keep the model warm"""
    generator = StoryGenerator()
    result = generator.generate_tokens.local("Hello", max_tokens=5, temperature=0.5)
    print(f"Warmup result: {result}")

# Create a singleton instance
story_generator = StoryGenerator()

# Web endpoint for easy access
@app.function(image=image)
@modal.web_endpoint(method="POST")
def generate_story_endpoint(request_data: dict):
    """
    Web endpoint for story generation
    POST body: {"prompt": "Once upon a time", "max_tokens": 100, "temperature": 0.7}
    """
    prompt = request_data.get("prompt", "")
    max_tokens = request_data.get("max_tokens", 100)
    temperature = request_data.get("temperature", 0.7)
    
    if not prompt:
        return {"success": False, "error": "No prompt provided"}
    
    result = story_generator.generate_tokens.remote(prompt, max_tokens, temperature)
    return result

if __name__ == "__main__":
    # For local testing
    print("Testing Modal app locally...")
    with app.run():
        generator = StoryGenerator()
        result = generator.generate_tokens.local(
            "Once upon a time there was a", 
            max_tokens=50, 
            temperature=0.7
        )
        print(f"Result: {result}")