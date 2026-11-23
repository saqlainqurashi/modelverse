
import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "train.jsonl"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

with DATASET_PATH.open("r", encoding="utf-8") as file:
    examples = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

print("Total training examples:", len(examples))

token_lengths = []

for index, example in enumerate(examples, start=1):
    messages = example["messages"]

    # Full conversation, including expected assistant response
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    token_lengths.append(len(token_ids))

    print(f"Example {index}: {len(token_ids)} tokens")

    if index == 1:
        print("\n=== FIRST TRAINING EXAMPLE ===")
        print("Question:", messages[1]["content"])
        print("Expected answer:", messages[2]["content"])
        print("\nFirst 20 token IDs:")
        print(token_ids[:20])

        print("\nFormatted conversation:")
        print(tokenizer.decode(token_ids))

print("\n=== DATASET STATISTICS ===")
print("Minimum tokens:", min(token_lengths))
print("Maximum tokens:", max(token_lengths))
print("Average tokens:", round(
    sum(token_lengths) / len(token_lengths), 2
))
