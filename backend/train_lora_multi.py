
import csv
import json
import math
import random
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

# =====================================
# 1. CONFIGURATION
# =====================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "datasets" / "train_expanded.jsonl"
VAL_FILE = ROOT / "datasets" / "validation.jsonl"

OUTPUT_DIR = ROOT / "models" / "lora-full-training"
HISTORY_FILE = ROOT / "experiments" / "full_training_history.csv"

EPOCHS = 4
MAX_TRAIN_EXAMPLES = 4  # Set to None for full dataset
MAX_LENGTH = 192
LEARNING_RATE = 2e-4
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

# Prevent accidentally overwriting an experiment
if OUTPUT_DIR.exists() or HISTORY_FILE.exists():
    raise FileExistsError(
        "Experiment output already exists. "
        "Use a new output name to preserve past results."
    )

print("Training device: CPU")

# =====================================
# 2. LOAD DATASET
# =====================================

def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


train_examples = load_jsonl(TRAIN_FILE)
val_examples = load_jsonl(VAL_FILE)

# Check exact question overlap
def questions(examples):
    return {
        " ".join(e["messages"][1]["content"].casefold().split())
        .rstrip("?.!")
        for e in examples
    }


if questions(train_examples) & questions(val_examples):
    raise ValueError("Training/validation question overlap")

print("Training examples available:", len(train_examples))
print("Validation examples:", len(val_examples))

if MAX_TRAIN_EXAMPLES is not None:
    train_examples = train_examples[:MAX_TRAIN_EXAMPLES]

# =====================================
# 3. LOAD MODEL AND TOKENIZER
# =====================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

model.to("cpu")
model.config.use_cache = False

# =====================================
# 4. ATTACH LORA
# =====================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

print("\nLoRA configuration ready!")
model.print_trainable_parameters()

# =====================================
# 5. PREPARE TOKENIZED DATA
# =====================================

def prepare_example(example):
    messages = example["messages"]

    if [m["role"] for m in messages] != [
        "system", "user", "assistant"
    ]:
        raise ValueError("Invalid conversation structure")

    # Build Qwen prompt including assistant header
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Construct assistant response with EOS marker
    answer_text = (
        messages[-1]["content"]
        + tokenizer.eos_token
    )

    prompt_ids = tokenizer.encode(
        prompt_text,
        add_special_tokens=False,
    )

    answer_ids = tokenizer.encode(
        answer_text,
        add_special_tokens=False,
    )

    input_ids = prompt_ids + answer_ids

    # Skip oversized examples rather than
    # silently truncating the target response
    if len(input_ids) > MAX_LENGTH:
        return None

    if not answer_ids:
        return None

    # Mask prompt; supervise assistant response
    labels = (
        [-100] * len(prompt_ids)
        + answer_ids
    )

    return {
        "input_ids": torch.tensor(
            [input_ids], dtype=torch.long
        ),
        "attention_mask": torch.ones(
            (1, len(input_ids)), dtype=torch.long
        ),
        "labels": torch.tensor(
            [labels], dtype=torch.long
        ),
    }


train_data = [
    item
    for example in train_examples
    if (item := prepare_example(example)) is not None
]

val_data = [
    item
    for example in val_examples
    if (item := prepare_example(example)) is not None
]

print("\nUsable training examples:", len(train_data))
print("Usable validation examples:", len(val_data))

if not train_data or not val_data:
    raise ValueError(
        "No usable examples. Review MAX_LENGTH."
    )

# =====================================
# 6. CREATE OPTIMIZER
# =====================================

trainable_parameters = [
    p for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE,
)

# =====================================
# 7. VALIDATION FUNCTION
# =====================================

def evaluate():
    model.eval()
    weighted_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_data:
            loss = model(**batch).loss

            # Causal LM internally shifts labels
            token_count = (
                batch["labels"][:, 1:] != -100
            ).sum().item()

            weighted_loss += loss.item() * token_count
            total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("No validation target tokens")

    return weighted_loss / total_tokens


# =====================================
# 8. TRAINING LOOP
# =====================================

print("\nStarting multi-epoch LoRA training!")

best_val_loss = float("inf")
history = []

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(EPOCHS):

    print(f"\n========== EPOCH {epoch + 1}/{EPOCHS} ==========")

    model.train()
    random.shuffle(train_data)

    train_weighted_loss = 0.0
    train_token_count = 0

    for step, batch in enumerate(train_data, start=1):
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
            error_if_nonfinite=True,
        )

        # Update adapter parameters
        optimizer.step()

        token_count = (
            batch["labels"][:, 1:] != -100
        ).sum().item()

        train_weighted_loss += loss.item() * token_count
        train_token_count += token_count

        print(
            f"Step {step}/{len(train_data)} | "
            f"Loss: {loss.item():.4f}",
            flush=True,
        )

    train_loss = train_weighted_loss / train_token_count

    # Run validation after the epoch
    val_loss = evaluate()

    if not math.isfinite(val_loss):
        raise RuntimeError("Non-finite validation loss")

    print(f"\nEpoch {epoch + 1} results:")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": val_loss,
    })

    # Save best validation checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        best_dir = OUTPUT_DIR / "best_adapter"
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)

        print("New best adapter saved!")

    # Save latest adapter
    latest_dir = OUTPUT_DIR / "latest_adapter"
    model.save_pretrained(latest_dir)
    tokenizer.save_pretrained(latest_dir)

    # Write training history after each epoch
    with HISTORY_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "validation_loss",
            ],
        )

        writer.writeheader()
        writer.writerows(history)

    # Restore training mode for the next epoch
    model.train()

print("\n================================")
print("TRAINING COMPLETED")
print("================================")

print(f"Best validation loss: {best_val_loss:.4f}")
print("Best adapter:", OUTPUT_DIR / "best_adapter")
print("Latest adapter:", OUTPUT_DIR / "latest_adapter")
print("Training history:", HISTORY_FILE)
