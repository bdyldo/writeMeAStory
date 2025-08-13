"""
Modal app for GPU-accelerated story generation.
Deploy with: modal deploy modal_app.py
"""

import modal
from pathlib import Path

# Modal configuration
app = modal.App("story-generator")

# Create a custom Docker-Like environment with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    ["torch", "transformers", "fastapi", "pydantic"]
)

# Create a cloud based file system inside Modal to attach files needed GPU instances
MODEL_VOLUME = modal.Volume.from_name("story-model", create_if_missing=True)


# Define a class-based deployment container that runs on a GPU with Transformer model and dependencies
@app.cls(
    image=image,
    gpu="A10G",  # Use A10G GPU - good balance of performance/cost
    volumes={"/model": MODEL_VOLUME},
    scaledown_window=300,  # Keep container warm for 300 seconds
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

        # Load transformer_v2 from the correct path
        transformer_path = "/model/app/model/transformer_v2.py"

        if not os.path.exists(transformer_path):
            raise FileNotFoundError(
                f"Could not find transformer_v2.py at {transformer_path}"
            )

        print(f"Loading transformer_v2 from: {transformer_path}")

        spec = importlib.util.spec_from_file_location("transformer_v2", transformer_path)
        transformer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transformer_module)
        ProperTransformer = transformer_module.ProperTransformer

        from transformers import AutoTokenizer

        model_path = Path("/model/used_weight.pt")  
        tokenizer_path = Path("/model/tokenizer")

        print(f"Loading model from: {model_path}")
        print(f"Loading tokenizer from: {tokenizer_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        # Create model with same parameters as training (from transformer_v2.py)
        self.model = ProperTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=768,
            num_heads=12,
            num_layers=16,
            d_ff=3072,
            dropout=0.1
        )

        # Load model weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(str(model_path), map_location=device))
        self.model.to(device)
        self.model.eval()

        print(
            f"✅ Model loaded on {device} with vocab size: {self.tokenizer.vocab_size}"
        )

    @modal.method()
    def generate_tokens(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.7
    ):
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

            generated_text = self.tokenizer.decode(
                output_tokens, skip_special_tokens=True
            )
            return {
                "success": True,
                "generated_text": generated_text,
                "tokens_generated": len(output_tokens),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "generated_text": ""}


# Create a StoryGenerator instance
story_generator = StoryGenerator()


# Setting up the web endpoint for Modal to call using HTTP
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
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
