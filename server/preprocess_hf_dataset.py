#!/usr/bin/env python3
"""
Download HuggingFace dataset, tokenize with GPT-2, and prepare for transformer training
"""

import json
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import argparse

def download_and_tokenize_dataset(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1", 
    max_length: int = 512,
    num_samples: int = 100000,
    output_train: str = "pretrain_data.json",
    output_val: str = "pretrain_val.json"
):
    """Download HF dataset and tokenize for transformer training"""
    
    print(f"📦 Loading dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config)
    
    print("🔤 Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token
    
    def tokenize_and_chunk(examples):
        """Tokenize text and chunk into fixed-length sequences - optimized for quality"""
        sequences = []
        
        for text in examples["text"]:
            if not text or not text.strip():
                continue
                
            text = text.strip()
            
            # Filter out low-quality text
            if len(text) < 100:  # Too short
                continue
            if len(text) > 100000:  # Too long, likely corrupted
                continue
            if text.count('\n') / len(text) > 0.1:  # Too many line breaks
                continue
                
            try:
                # Process in reasonable chunks (don't truncate good content)
                words = text.split()
                chunk_size = 8000  # Process ~8k words at a time
                
                for i in range(0, len(words), chunk_size):
                    word_chunk = ' '.join(words[i:i + chunk_size])
                    
                    tokens = tokenizer.encode(word_chunk, add_special_tokens=False)
                    
                    # Create overlapping sequences for better context
                    stride = max_length // 4  # 25% overlap
                    for j in range(0, len(tokens) - max_length + 1, stride):
                        sequence = tokens[j:j + max_length]
                        if len(sequence) == max_length:
                            sequences.append(sequence)
                            
                        # Stop if we have enough sequences from this batch
                        if len(sequences) >= 1000:
                            break
                    
                    if len(sequences) >= 1000:
                        break
                        
            except Exception as e:
                continue
        
        return {"input_ids": sequences}
    
    print("🔄 Tokenizing and chunking dataset...")
    
    # Process training data only (validation set is too small)
    train_data = dataset["train"].map(
        tokenize_and_chunk,
        batched=True,
        batch_size=100,  # Smaller batches to avoid memory issues
        remove_columns=dataset["train"].column_names
    )
    
    # Flatten the nested lists
    all_sequences = []
    for batch in train_data:
        all_sequences.extend(batch["input_ids"])
    
    # Limit and split into train/val
    all_sequences = all_sequences[:num_samples]
    
    # Split: 90% train, 10% validation
    split_idx = int(len(all_sequences) * 0.9)
    train_sequences = all_sequences[:split_idx]
    val_sequences = all_sequences[split_idx:]
    
    print(f"📊 Processed {len(train_sequences)} training sequences")
    print(f"📊 Processed {len(val_sequences)} validation sequences")
    print(f"📊 Sequence length: {max_length} tokens")
    print(f"📊 Vocabulary size: {tokenizer.vocab_size}")
    
    # Save in format expected by your transformer
    print(f"💾 Saving training data to: {output_train}")
    with open(output_train, 'w') as f:
        json.dump(train_sequences, f)
    
    print(f"💾 Saving validation data to: {output_val}")
    with open(output_val, 'w') as f:
        json.dump(val_sequences, f)
    
    print("✅ Dataset preprocessing complete!")
    
    return {
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "vocab_size": tokenizer.vocab_size,
        "sequence_length": max_length
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HF dataset for transformer training")
    
    parser.add_argument("--dataset", type=str, default="wikitext", help="HuggingFace dataset name")
    parser.add_argument("--config", type=str, default="wikitext-103-raw-v1", help="Dataset configuration")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of training samples")
    parser.add_argument("--output_train", type=str, default="pretrain_data.json", help="Output training file")
    parser.add_argument("--output_val", type=str, default="pretrain_val.json", help="Output validation file")
    
    args = parser.parse_args()
    
    stats = download_and_tokenize_dataset(
        dataset_name=args.dataset,
        dataset_config=args.config,
        max_length=args.max_length,
        num_samples=args.num_samples,
        output_train=args.output_train,
        output_val=args.output_val
    )
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value:,}")