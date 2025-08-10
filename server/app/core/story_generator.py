import asyncio
from transformers import AutoTokenizer
from pathlib import Path
from ..model.transformer import RNNLanguageModel


class StoryGenerator:
    def __init__(self):
        current_file = Path(__file__)
        server_dir = current_file.parent.parent.parent
        model_path = server_dir / "app" / "model" / "model.pt"
        tokenizer_path = server_dir / "app" / "model" / "tokenizer"

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"Loading model from: {model_path}")

        # Load tokenizer first to get vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        # Create model instance with same parameters as training
        # ! # Must match training exactly
        self.model = RNNLanguageModel(
            embed_dim=512,
            hidden_dim=768,
            vocab_size=self.tokenizer.vocab_size,
            num_head=6,
            key_dim=128,
            value_dim=128,
        )

        # Load the saved state dictionary
        self.model.load_state_dict(torch.load(str(model_path), map_location=device))
        self.model.to(device)
        self.model.eval()

        print(f"✅ Model loaded with vocab size: {self.tokenizer.vocab_size}")

    async def generate_story_stream(self, prompt, max_tokens, temperature):
        try:
            # Convert prompt to tokens
            input_ids = self.tokenizer.encode(
                prompt, add_special_tokens=False, return_tensors="pt"
            )
            input_ids = input_ids.to(next(self.model.parameters()).device)

            import torch

            with torch.no_grad():
                output_tokens = self.model.generate(
                    input_ids, max_tokens=max_tokens, temperature=temperature
                )

            generated_text = self.tokenizer.decode(
                output_tokens, skip_special_tokens=True
            )

            # Now stream it character by character or word by word for UI effect
            words = generated_text.split()
            for i, word in enumerate(words):
                if i == 0:
                    yield word
                else:
                    yield " " + word  # Proper spacing

                await asyncio.sleep(0.05)

        except Exception as e:
            print(f"❌ Generation error: {e}")
            yield f" [Error: {str(e)}]"
