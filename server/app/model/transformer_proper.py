import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sin/cos calculation
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        
        # Register as buffer (not parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to embeddings
        Args:
            x: (seq_len, batch_size, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention (improved from your SelfAttention)"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Single linear layer for efficiency (instead of separate Q,K,V)
        self.w_qkv = nn.Linear(d_model, 3 * d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (seq_len, seq_len) causal mask
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V in one go
        qkv = self.w_qkv(x)  # (batch_size, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, d_k)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attn_output)


class FeedForward(nn.Module):
    """Position-wise feed-forward network (missing from your model)"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block (missing from your model)"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization (Pre-LN style - more stable)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: causal mask
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Pre-LayerNorm + Residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Pre-LayerNorm + Residual connection  
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class ProperTransformer(nn.Module):
    """Proper GPT-style decoder-only transformer"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks stack
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Generate causal mask to prevent attending to future tokens"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: (batch_size, seq_len)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Token embeddings
        x = self.token_embedding(tokens)  # (batch_size, seq_len, d_model)
        x = x * math.sqrt(self.d_model)  # Scale embeddings
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        x = self.dropout(x)
        
        # Generate causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def generate(self, tokens: Tensor, max_tokens: int = 50, temperature: float = 1.0) -> Tensor:
        """Generate text using the transformer"""
        self.eval()
        device = tokens.device
        
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Forward pass
                logits = self.forward(tokens)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample next token
                if temperature == 0.0:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                generated_tokens.append(next_token.item())
                
                # Update tokens for next iteration - fix dimension mismatch
                next_token = next_token.view(1, 1)  # Ensure correct shape
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # Prevent infinite sequences
                if tokens.shape[1] > 1000:
                    break
        
        return torch.tensor(generated_tokens)


# Example usage and comparison
def create_proper_transformer(vocab_size: int = 50257) -> ProperTransformer:
    """Create a proper transformer with similar capacity to your hybrid model"""
    return ProperTransformer(
        vocab_size=vocab_size,
        d_model=512,        # Hidden dimension
        num_heads=8,        # Multi-head attention
        num_layers=6,       # Number of transformer blocks
        d_ff=2048,          # Feed-forward hidden dimension (usually 4x d_model)
        max_seq_len=1024,   # Maximum sequence length
        dropout=0.1         # Dropout rate
    )

# Training components
import json
import time
import argparse
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

# Initialize device
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

class SentenceDataset:
    """Dataset loader - fixed for proper batching"""
    def __init__(self, data_path):
        with open(data_path) as f:
            data = json.load(f)
            # Ensure each sequence is properly shaped
            data = [torch.tensor(seq, dtype=torch.long) for seq in data]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return tensor as (seq_len,) - DataLoader will add batch dimension
        return self.data[idx]


def train_transformer(
    model, train_data, valid_data, loss_fn, optimizer, 
    num_sequences, batch_size, scheduler=None, accumulation_steps=1,
):
    """Training loop for proper transformer"""
    model.train()
    max_grad_norm = 1.0
    
    train_batch_losses = []
    valid_batch_losses = []
    train_batch_loss = 0.0
    
    start_time = time.time()
    val_frequency = 0.1
    val_index = max(int(num_sequences * val_frequency) // batch_size, 1)
    
    optimizer.zero_grad()
    accumulated_loss = 0.0
    
    for idx, sequence in enumerate(train_data):
        time_elapsed = round((time.time() - start_time) / 60, 6)
        sequence = sequence.to(device)
        
        if idx >= num_sequences // batch_size:
            break
        
        # Ensure proper batch dimension
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)  # (seq_len,) -> (1, seq_len)
        
        # Forward pass
        logits = model(sequence)
        
        # Prepare targets (shift right for next-token prediction)
        targets = sequence[:, 1:]
        logits = logits[:, :-1, :]
        
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        
        # Compute loss
        loss = loss_fn(logits_flat, targets_flat) / accumulation_steps
        accumulated_loss += loss.item()
        loss.backward()
        
        # Update weights
        if (idx + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
            train_batch_loss += accumulated_loss * accumulation_steps
            accumulated_loss = 0.0
        
        # Validation
        if idx % val_index == 0:
            avg_train_loss = round(train_batch_loss / val_index if idx != 0 else train_batch_loss, 6)
            train_batch_losses.append(avg_train_loss)
            train_batch_loss = 0.0
            
            print(f"Batch: {idx} | Seq Length: {sequence.shape[1]} | "
                  f"Time: {time_elapsed}min | Train Loss: {avg_train_loss}")
            
            # Validation
            valid_loss = round(validate_transformer(model, valid_data, loss_fn), 6)
            valid_batch_losses.append(valid_loss)
            print(f"Validation Loss: {valid_loss}")
            
            if scheduler and len(valid_batch_losses) > 1:
                scheduler.step(valid_loss)
                print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            model.train()
    
    return train_batch_losses, valid_batch_losses


@torch.no_grad()
def validate_transformer(model, dataset, loss_fn, num_batches=5):
    """Validation for transformer"""
    model.eval()
    total_loss = 0.0
    actual_batches = 0
    
    for i, sequence in enumerate(dataset):
        if i >= num_batches:
            break
        
        sequence = sequence.to(device)
        
        # Ensure proper batch dimension
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)  # (seq_len,) -> (1, seq_len)
            
        logits = model(sequence)
        
        targets = sequence[:, 1:]
        logits = logits[:, :-1, :]
        
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        
        loss = loss_fn(logits_flat, targets_flat)
        total_loss += loss.item()
        actual_batches += 1
    
    return total_loss / max(actual_batches, 1)


@torch.no_grad()
def complete_transformer(model, tokenizer, prefix: str, num_tokens=64, temperature=0.0):
    """Text completion using proper transformer"""
    model.eval()
    input_tokens = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")
    input_tokens = input_tokens.to(device)
    output = model.generate(input_tokens, max_tokens=num_tokens, temperature=temperature)
    return tokenizer.decode(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Proper Transformer")
    
    # Required arguments 
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True) 
    parser.add_argument("--num_sequences", type=int, required=True)
    
    # Model arguments 
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension (replaces hidden_dim)")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of transformer layers") 
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward dimension")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--pretrained_weights", type=str, help="Path to pre-trained weights (.pt file)")
    parser.add_argument("--save_model_path", type=str, default="transformer_proper.pt", help="Where to save trained model")
    
    # Output arguments
    parser.add_argument("--train_losses_out", type=str)
    parser.add_argument("--val_losses_out", type=str)
    parser.add_argument("--metrics_out", type=str)
    
    args = parser.parse_args()
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token
    vocab_size = tokenizer.vocab_size
    print(f"Using GPT-2 tokenizer with vocab size: {vocab_size}")
    
    # Create model and move to GPU
    model = ProperTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=0.1
    ).to(device)
    
    # Load pre-trained weights if provided
    if args.pretrained_weights:
        print(f"Loading pre-trained weights from: {args.pretrained_weights}")
        model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))
        print("✅ Pre-trained weights loaded successfully!")
    
    print(f"Model: Proper Transformer")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Load data
    print("Loading data...")
    train_data = SentenceDataset(args.train_data)
    valid_data = SentenceDataset(args.val_data)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False
    )
    
    # Training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3)
    
    start = time.time()
    train_losses, valid_losses = train_transformer(
        model, train_dataloader, valid_dataloader, loss_fn, optimizer,
        args.num_sequences, args.batch_size, scheduler, accumulation_steps=2
    )
    training_time = time.time() - start
    
    # Results
    results = {
        "Train Losses": train_losses,
        "Valid Losses": valid_losses, 
        "Final Train Loss": train_losses[-1] if train_losses else 0,
        "Final Valid Loss": valid_losses[-1] if valid_losses else 0,
        "Time": training_time
    }
    
    print(f"Final Train Loss: {results['Final Train Loss']}")
    print(f"Final Valid Loss: {results['Final Valid Loss']}")
    print(f"Training Time: {training_time:.1f}s")
    
    # Save outputs
    if args.train_losses_out:
        with open(args.train_losses_out, 'w') as f:
            for loss in train_losses:
                f.write(f"{loss}\n")
    
    if args.val_losses_out:
        with open(args.val_losses_out, 'w') as f:
            for loss in valid_losses:
                f.write(f"{loss}\n")
    
    if args.metrics_out:
        with open(args.metrics_out, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), args.save_model_path)
    print(f"💾 Model saved to: {args.save_model_path}")
    
    # Test generation
    test_prompts = ["Once upon a time there was a "]
    for prompt in test_prompts:
        completion = complete_transformer(model, tokenizer, prompt, num_tokens=32, temperature=0.6)
        print(f"Test: {prompt}")
        print(f"Output: {completion}")