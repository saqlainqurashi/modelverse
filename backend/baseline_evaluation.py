
import json
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "datasets" / "test.jsonl"
OUTPUT_FILE = BASE_DIR / "experiments" / "baseline_results.json"

SYSTEM_PROMPT = (
    "You are an AI engineering tutor. "
    "Explain concepts clearly with practical examples."
)

# 1. Load the test dataset
with TEST_FILE.open("r", encoding="utf-8-sig") as file:
    test_examples = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

print("Total test examples:", len(test_examples))

# 2. Load pretrained Qwen
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading original Qwen model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.to("cpu")
model.eval()

print("Model loaded successfully!")

# 3. Evaluate each question
results = []

for index, example in enumerate(test_examples, start=1):

    messages = example["messages"]

    question = messages[1]["content"]
    expected_answer = messages[2]["content"]

    # Only pass system and user messages to the model
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    print(f"\nEvaluating question {index}...")
    print("Question:", question)

    # Generate the model's response
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Remove the input prompt tokens
    new_tokens = output_ids[0][
        inputs["input_ids"].shape[-1]:
    ]

    generated_answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    print("Model answer:", generated_answer)

    results.append({
        "question": question,
        "expected_answer": expected_answer,
        "model_answer": generated_answer
    })

# 4. Save results
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

if OUTPUT_FILE.exists():
    raise FileExistsError(
        "Baseline already exists. Preserve the original "
        "results before running again."
    )

output = {
    "model": MODEL_NAME,
    "evaluation_type": "baseline",
    "generation_settings": {
        "max_new_tokens": 120,
        "do_sample": False
    },
    "system_prompt": SYSTEM_PROMPT,
    "total_examples": len(results),
    "results": results
}

with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    json.dump(output, file, indent=2, ensure_ascii=False)

print("\nBaseline evaluation completed!")
print("Results saved to:", OUTPUT_FILE)
