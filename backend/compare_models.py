
import csv
import json
import math
import torch

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =========================================
# 1. Paths and configuration
# =========================================

ROOT = Path(__file__).resolve().parent.parent

BASELINE_FILE = ROOT / "experiments" / "baseline_results.json"
HISTORY_FILE = ROOT / "experiments" / "full_training_history.csv"
TEST_FILE = ROOT / "datasets" / "test.jsonl"

ADAPTER_DIR = (
    ROOT / "models" / "lora-full-training" / "best_adapter"
)

REPORT_FILE = (
    ROOT / "experiments" / "model_comparison.json"
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# =========================================
# 2. Check files
# =========================================

required = [
    BASELINE_FILE,
    HISTORY_FILE,
    TEST_FILE,
    ADAPTER_DIR / "adapter_config.json",
    ADAPTER_DIR / "adapter_model.safetensors",
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

if REPORT_FILE.exists():
    raise FileExistsError(
        "Comparison report already exists. "
        "Preserve it before creating another."
    )

# =========================================
# 3. Analyze training history
# =========================================

print("\n=== TRAINING HISTORY ===")

with HISTORY_FILE.open("r", encoding="utf-8") as file:
    history = list(csv.DictReader(file))

if not history:
    raise ValueError("Training history is empty")

best_row = min(
    history,
    key=lambda row: float(row["validation_loss"]),
)

for row in history:
    print(
        f"Epoch {row['epoch']} | "
        f"Train loss: {float(row['train_loss']):.4f} | "
        f"Validation loss: "
        f"{float(row['validation_loss']):.4f}"
    )

print("\nBest validation epoch:", best_row["epoch"])
print(
    "Best recorded validation loss:",
    best_row["validation_loss"],
)

# =========================================
# 4. Load original baseline
# =========================================

with BASELINE_FILE.open("r", encoding="utf-8") as file:
    baseline = json.load(file)

if baseline.get("model") != MODEL_NAME:
    raise ValueError("Baseline uses a different base model")

baseline_results = baseline["results"]

with TEST_FILE.open("r", encoding="utf-8-sig") as file:
    test_records = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

if len(baseline_results) != len(test_records):
    raise ValueError(
        "Baseline and test dataset sizes differ"
    )

# Verify identical questions and references
for saved, record in zip(baseline_results, test_records):
    if saved["question"] != record["messages"][1]["content"]:
        raise ValueError("Test questions changed")

    if (
        saved["expected_answer"]
        != record["messages"][2]["content"]
    ):
        raise ValueError("Test reference answers changed")

# Preserve original evaluation configuration
system_prompt = baseline["system_prompt"]
generation = baseline["generation_settings"]

max_new_tokens = generation["max_new_tokens"]
do_sample = generation["do_sample"]

if do_sample is not False:
    raise ValueError(
        "This comparison expects deterministic "
        "baseline generation"
    )

# =========================================
# 5. Load Qwen and trained adapter
# =========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading original Qwen model on CPU...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

base_model.to("cpu")
base_model.eval()

print("Loading best LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_DIR),
    is_trainable=False,
)

model.eval()

# =========================================
# 6. Generate adapted answers
# =========================================

def generate_answer(question):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


# =========================================
# 7. Compare results
# =========================================

comparisons = []

for index, item in enumerate(baseline_results, start=1):
    question = item["question"]
    expected = item["expected_answer"]
    original = item["model_answer"]

    print(f"\nEvaluating question {index}...")
    print("Question:", question)

    fine_tuned = generate_answer(question)

    print("\nOriginal Qwen:")
    print(original)

    print("\nFine-tuned Qwen:")
    print(fine_tuned)

    comparisons.append({
        "question": question,
        "reference_answer": expected,
        "base_answer": original,
        "lora_answer": fine_tuned,
        "answers_identical": original == fine_tuned,
        "base_word_count": len(original.split()),
        "lora_word_count": len(fine_tuned.split()),
    })

# =========================================
# 8. Save comparison report
# =========================================

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

report = {
    "model": MODEL_NAME,
    "adapter": str(ADAPTER_DIR),
    "best_validation_epoch": int(best_row["epoch"]),
    "best_validation_loss": float(
        best_row["validation_loss"]
    ),
    "generation_settings": generation,
    "total_questions": len(comparisons),
    "results": comparisons,
}

with REPORT_FILE.open("w", encoding="utf-8") as file:
    json.dump(
        report,
        file,
        indent=2,
        ensure_ascii=False,
    )

print("\n================================")
print("MODEL COMPARISON COMPLETED")
print("================================")
print("Total questions:", len(comparisons))
print(
    "Identical answers:",
    sum(item["answers_identical"] for item in comparisons)
)
print("Report saved:", REPORT_FILE)
