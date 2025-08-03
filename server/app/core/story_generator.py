from app.model.transformer import OptimizedLanguageModel, ModelConfig

class StoryGenerator:
    def __init__(self):
        # Create your model directly
        config = ModelConfig(
            embed_dim=256,
            hidden_dim=512,
            vocab_size=10000,  # Set your actual vocab size
            # ... other config
        )
        self.model = OptimizedLanguageModel(config)
        
        # Simple word tokenizer (or use your own)
        self.vocab = self._create_simple_vocab()
    
    def _create_simple_vocab(self):
        # Quick & dirty tokenizer - replace with yours
        return {"hello": 1, "world": 2, "story": 3, "the": 4}
    
    async def generate_story_stream(self, prompt, max_tokens, temperature):
        # Convert prompt to token IDs however you want
        token_ids = self._text_to_tokens(prompt)
        input_tensor = torch.tensor([token_ids])
        
        # Generate with YOUR model
        generated = self.model.generate(
            input_tensor, 
            max_new_tokens=max_tokens, 
            temperature=temperature
        )
        
        # Stream tokens back
        for token_id in generated:
            token_text = self._token_to_text(token_id.item())
            yield token_text