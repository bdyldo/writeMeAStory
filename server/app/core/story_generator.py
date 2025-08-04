import torch
import asyncio
from transformers import AutoTokenizer
from pathlib import Path
from app.model.transformer import RNNLanguageModel  

class StoryGenerator:
    def __init__(self):
        # Get tokenizer path
        current_file = Path(__file__)
        server_dir = current_file.parent.parent.parent
        tokenizer_path = server_dir / "app" / "model" / "tokenizer"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        print(f"📝 Tokenizer loaded with vocab size: {self.tokenizer.vocab_size}")
        
        # create model
        self.model = RNNLanguageModel(
            embed_dim=256,          
            hidden_dim=512,          
            vocab_size=self.tokenizer.vocab_size, 
            key_dim=64,              
            value_dim=64            
        )
        
        # loading trained weight from model.pt
        model_path = server_dir / "app" / "model" / "model.pt"
        if model_path.exists():
            print(f"🎯 Loading trained weights from: {model_path}")
            try:
                checkpoint = torch.load(str(model_path), map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif hasattr(checkpoint, 'state_dict'):  # If full model was saved
                    self.model = checkpoint
                else:
                    print("⚠️ Could not load weights, using random initialization")
            except Exception as e:
                print(f"⚠️ Could not load weights: {e}, using random initialization")
        else:
            print("⚠️ No trained weights found, using random initialization")
        
        # switch model to eval mode
        self.model.eval()
        print(f"✅ Model ready with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    async def generate_story_stream(self, prompt, max_tokens, temperature):
        try:
            print(f"🎭 Generating story for: '{prompt[:50]}...'")
            
            # Convert prompt to tokens using model's approach
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            
            # Use transformer model to generate tokens
            generated_tokens = self.model.generate(
                input_ids, 
                max_tokens=max_tokens, 
                temperature=temperature
            )
            
            print(f"🔤 Generated {len(generated_tokens)} tokens")
            
            # Stream each token
            for i, token_id in enumerate(generated_tokens):
                # Decode single token
                token_text = self.tokenizer.decode([token_id.item()], skip_special_tokens=True)
                
                # Only yield non-empty tokens
                if token_text.strip():
                    print(f"📤 Token {i}: '{token_text}'")  # Debug output
                    yield token_text
                    
                # Small delay for streaming effect
                await asyncio.sleep(0.1)
                
            print("✅ Story generation completed")
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            yield f" [Error: {str(e)}]"