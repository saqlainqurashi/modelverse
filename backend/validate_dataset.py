
import json
from pathlib import Path

DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "instructions.jsonl"
)

REQUIRED_ROLES = ["system", "user", "assistant"]


def validate_dataset(path):
    seen_questions = set()
    valid_examples = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(
                    f"Empty line at {line_number}"
                )

            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from error

            if not isinstance(example, dict):
                raise ValueError("Example must be an object")

            messages = example.get("messages")

            if not isinstance(messages, list):
                raise ValueError(
                    f"Missing messages at line {line_number}"
                )

            if len(messages) != 3:
                raise ValueError(
                    f"Expected 3 messages at line {line_number}"
                )

            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise ValueError("Invalid message object")

                if message.get("role") != REQUIRED_ROLES[index]:
                    raise ValueError(
                        f"Incorrect role at line {line_number}"
                    )

                content = message.get("content")

                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        f"Empty content at line {line_number}"
                    )

            question = messages[1]["content"].strip().casefold()

            if question in seen_questions:
                raise ValueError(
                    f"Duplicate question at line {line_number}"
                )

            seen_questions.add(question)
            valid_examples += 1

    if valid_examples == 0:
        raise ValueError("Dataset contains no examples")

    print("Dataset validation successful!")
    print("Total valid examples:", valid_examples)
    print("Duplicate questions: 0")
    print("Invalid records: 0")


if __name__ == "__main__":
    validate_dataset(DATASET_PATH)
