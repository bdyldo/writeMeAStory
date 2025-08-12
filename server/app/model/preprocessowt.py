# preprocess_openwebtext2.py
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

DATASET_ID = "segyges/OpenWebText2"   # HF dataset name
BLOCK_SIZE = 1024                     # match your model context length
VAL_FRAC = 0.005                       # ~0.5% for validation

OUT_TRAIN = "large_general_text.json"
OUT_VAL   = "large_general_val.json"

def pack_blocks(token_lists, block_size):
    flat = []
    blocks = []
    for ids in token_lists:
        flat.extend(ids)
        while len(flat) >= block_size:
            blocks.append(flat[:block_size])
            flat = flat[block_size:]
    return blocks

def main():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(DATASET_ID, split="train")
    split = ds.train_test_split(test_size=VAL_FRAC, seed=42)

    for name, subset, outfile in [
        ("train", split["train"], OUT_TRAIN),
        ("val", split["test"], OUT_VAL)
    ]:
        sequences = []
        batch_ids = []
        for ex in tqdm(subset, desc=f"Tokenizing {name}"):
            ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
            if ids:
                ids.append(tok.eos_token_id)
                batch_ids.append(ids)
            if len(batch_ids) >= 2048:
                sequences.extend(pack_blocks(batch_ids, BLOCK_SIZE))
                batch_ids = []
        if batch_ids:
            sequences.extend(pack_blocks(batch_ids, BLOCK_SIZE))

        with open(outfile, "w") as f:
            json.dump(sequences, f)
        print(f"{name}: {len(sequences)} sequences saved to {outfile}")

if __name__ == "__main__":
    main()
