#!/usr/bin/env python3
"""
Fixed HuggingFace dataset preprocessing 
"""

import json
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

def create_training_data(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1", 
    max_length: int = 512,
    num_samples: int = 100000,
    output_train: str = "pretrain_data.json",
    output_val: str = "pretrain_val.json"
):
    """Simple, working approach to create training data"""
    
    print(f"📦 Loading dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config)
    
    print("🔤 Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("🔄 Processing training data...")
    
    # Concatenate all non-empty texts
    all_texts = []
    for text in tqdm(dataset["train"]["text"]):
        if text and text.strip() and len(text.strip()) > 50:  # Only non-empty, substantial text
            all_texts.append(text.strip())
    
    print(f"📄 Found {len(all_texts)} valid texts")
    
    # Join all texts together
    combined_text = " ".join(all_texts)
    print(f"📝 Combined text length: {len(combined_text)} characters")
    
    # Tokenize the entire text
    print("🔤 Tokenizing combined text...")
    all_tokens = tokenizer.encode(combined_text, add_special_tokens=False)
    print(f"🔢 Total tokens: {len(all_tokens)}")
    
    # Split into fixed-length sequences
    print(f"✂️  Creating sequences of length {max_length}...")
    sequences = []
    
    for i in range(0, len(all_tokens) - max_length + 1, max_length):
        sequence = all_tokens[i:i + max_length]
        if len(sequence) == max_length:
            sequences.append(sequence)
            
        if len(sequences) >= num_samples:
            break
    
    print(f"📊 Created {len(sequences)} sequences")
    
    # Split into train/validation
    split_idx = int(len(sequences) * 0.9)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    print(f"📊 Train sequences: {len(train_sequences)}")
    print(f"📊 Validation sequences: {len(val_sequences)}")
    
    # Verify data structure
    print("\n🔍 Verifying data structure...")
    print(f"First sequence type: {type(train_sequences[0])}")
    print(f"First sequence length: {len(train_sequences[0])}")
    print(f"First few tokens: {train_sequences[0][:10]}")
    
    # Test decoding
    sample_text = tokenizer.decode(train_sequences[0][:50])
    print(f"Sample decoded text: '{sample_text}'")
    
    # Save data
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed preprocessing for HF dataset")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--output_train", type=str, default="pretrain_data.json")
    parser.add_argument("--output_val", type=str, default="pretrain_val.json")
    
    args = parser.parse_args()
    
    stats = create_training_data(
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