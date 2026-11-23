
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"

TRAIN_FILE = DATASET_DIR / "train.jsonl"
ADDITIONAL_FILE = DATASET_DIR / "additional_instructions.jsonl"

VALIDATION_FILE = DATASET_DIR / "validation.jsonl"
TEST_FILE = DATASET_DIR / "test.jsonl"

OUTPUT_FILE = DATASET_DIR / "train_expanded.jsonl"


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path.name}, "
                    f"line {line_number}"
                ) from error

            # Validate conversational structure
            if not isinstance(record, dict):
                raise ValueError("Invalid record")

            messages = record.get("messages")

            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError(
                    f"Invalid messages in {path.name}, "
                    f"line {line_number}"
                )

            for expected_role, message in zip(
                ["system", "user", "assistant"], messages
            ):
                if not isinstance(message, dict):
                    raise ValueError("Invalid message")

                if message.get("role") != expected_role:
                    raise ValueError("Incorrect message role")

                content = message.get("content")

                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Missing message content")

            records.append(record)

    return records


def normalize_question(question):
    # Normalize case, whitespace and trailing punctuation
    question = question.casefold().strip()
    question = re.sub(r"\s+", " ", question)
    question = question.rstrip("?!.")

    return question


def get_question(record):
    return normalize_question(
        record["messages"][1]["content"]
    )


print("Loading datasets...")

train_records = load_jsonl(TRAIN_FILE)
additional_records = load_jsonl(ADDITIONAL_FILE)
validation_records = load_jsonl(VALIDATION_FILE)
test_records = load_jsonl(TEST_FILE)

print("Original training:", len(train_records))
print("Additional examples:", len(additional_records))
print("Validation:", len(validation_records))
print("Test:", len(test_records))

# Never include held-out questions in training
protected_questions = {
    get_question(record)
    for record in validation_records + test_records
}

combined_records = train_records + additional_records

seen_questions = set()
clean_records = []

duplicates = 0
leakage = 0

for record in combined_records:
    question = get_question(record)

    if question in protected_questions:
        leakage += 1
        continue

    if question in seen_questions:
        duplicates += 1
        continue

    seen_questions.add(question)
    clean_records.append(record)

print("\n=== DATASET REPORT ===")
print("Combined examples:", len(combined_records))
print("Duplicate questions removed:", duplicates)
print("Held-out overlaps removed:", leakage)
print("Final training examples:", len(clean_records))

if not clean_records:
    raise ValueError("No training examples remain")

# Save new training dataset
with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    for record in clean_records:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

print("\nExpanded dataset saved successfully!")
print("Output:", OUTPUT_FILE)
