
import json
import random
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"
SOURCE_FILE = DATASET_DIR / "instructions.jsonl"

with SOURCE_FILE.open("r", encoding="utf-8") as file:
    examples = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

if len(examples) != 12:
    raise ValueError(
        "This exercise expects exactly 12 examples"
    )

# Fixed seed makes the split reproducible
random.Random(42).shuffle(examples)

train_data = examples[:8]
validation_data = examples[8:10]
test_data = examples[10:12]


def save_jsonl(filename, records):
    path = DATASET_DIR / filename

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


save_jsonl("train.jsonl", train_data)
save_jsonl("validation.jsonl", validation_data)
save_jsonl("test.jsonl", test_data)

print("Dataset split completed!")
print("Training examples:", len(train_data))
print("Validation examples:", len(validation_data))
print("Test examples:", len(test_data))
