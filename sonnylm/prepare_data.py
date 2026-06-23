"""
Download the SonnyLM dataset from Hugging Face, format it as ChatML JSONL,
and train a BPE tokenizer. Produces:

    data/train.jsonl   (ChatML user/assistant turns)
    data/eval.jsonl    (the test split)
    data/tokenizer.json

Run this once before train.py.
"""

import json
import os

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

HF_DATASET = "AnyaAl/sonnylm_generic"


def main():
    ds = load_dataset(HF_DATASET)
    print(f"Downloaded: {len(ds['train']):,} train, {len(ds['test']):,} test samples")

    os.makedirs("data", exist_ok=True)
    texts = []

    for split, path in [("train", "data/train.jsonl"), ("test", "data/eval.jsonl")]:
        with open(path, "w") as f:
            for row in ds[split]:
                text = (
                    f'<|im_start|>user\n{row["input"]}<|im_end|>\n'
                    f'<|im_start|>assistant\n{row["output"]}<|im_end|>'
                )
                f.write(json.dumps({"text": text, "category": row["category"]}) + "\n")
                texts.append(text)
        print(f"  {path}: {len(ds[split]):,} samples")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=4096,
        special_tokens=["<pad>", "<|im_start|>", "<|im_end|>"],
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save("data/tokenizer.json")
    print(f"  Tokenizer: {tokenizer.get_vocab_size()} tokens")

    with open("data/train.jsonl") as f:
        sample = json.loads(f.readline())
    print(f'\nSample ({sample["category"]}):\n{sample["text"]}')


if __name__ == "__main__":
    main()
