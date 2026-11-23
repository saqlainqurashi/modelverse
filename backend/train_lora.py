
import json
import math
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

# -----------------------------------------
# 1. Configuration
# -----------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_FILE = BASE_DIR / "datasets" / "train_expanded.jsonl"
OUTPUT_DIR = BASE_DIR / "models" / "lora-smoke-test"

MAX_LENGTH = 128
LEARNING_RATE = 1e-3
TRAINING_STEPS = 1

torch.manual_seed(42)

print("Device: CPU")

# -----------------------------------------
# 2. Load tokenizer
# -----------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -----------------------------------------
# 3. Load model
# -----------------------------------------

print("Loading Qwen on CPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

model.to("cpu")

# Disable key/value caching during training
model.config.use_cache = False

# -----------------------------------------
# 4. Attach LoRA adapters
# -----------------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

print("\nLoRA adapters attached!")
model.print_trainable_parameters()

# Reduce activation memory during training
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# -----------------------------------------
# 5. Read training examples
# -----------------------------------------

with DATASET_FILE.open("r", encoding="utf-8-sig") as file:
    examples = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

if not examples:
    raise ValueError("Training dataset is empty")

print("\nAvailable training examples:", len(examples))

# -----------------------------------------
# 6. Prepare one training example
# -----------------------------------------


def prepare_example(example):
    messages = example["messages"]

    if [m["role"] for m in messages] != [
        "system", "user", "assistant"
    ]:
        raise ValueError("Unexpected message roles")

    # System and user messages only
    prompt_messages = messages[:-1]

    # Expected assistant answer
    assistant_answer = messages[-1]["content"]

    # Format the prompt with Qwen's chat template
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize prompt separately
    prompt_ids = tokenizer.encode(
        prompt_text,
        add_special_tokens=False,
    )

    # Tokenize expected answer and end-of-turn marker
    answer_ids = tokenizer.encode(
        assistant_answer + tokenizer.eos_token,
        add_special_tokens=False,
    )

    # Combine prompt and answer tokens
    full_ids = prompt_ids + answer_ids

    if len(full_ids) > MAX_LENGTH:
        return None

    if not answer_ids:
        return None

    # Only assistant response tokens contribute to loss
    labels = (
        [-100] * len(prompt_ids)
        + answer_ids
    )

    return {
        "input_ids": torch.tensor(
            [full_ids], dtype=torch.long
        ),
        "attention_mask": torch.ones(
            (1, len(full_ids)), dtype=torch.long
        ),
        "labels": torch.tensor(
            [labels], dtype=torch.long
        ),
        "question": messages[1]["content"],
        "answer": assistant_answer,
        "length": len(full_ids),
    }



batch = None

for example in examples:
    candidate = prepare_example(example)

    if candidate is not None:
        batch = candidate
        break

if batch is None:
    raise ValueError(
        "No example fits MAX_LENGTH=128. "
        "Increase MAX_LENGTH or shorten an example."
    )

print("\n=== SELECTED EXAMPLE ===")
print("Question:", batch["question"])
print("Expected answer:", batch["answer"])
print("Sequence length:", batch["length"])

supervised_tokens = (
    batch["labels"] != -100
).sum().item()

print("Supervised tokens:", supervised_tokens)

inputs = {
    "input_ids": batch["input_ids"],
    "attention_mask": batch["attention_mask"],
    "labels": batch["labels"],
}

# -----------------------------------------
# 7. Optimizer
# -----------------------------------------

trainable_parameters = [
    p for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE,
)

# Save one adapter matrix before training
tracked_name, tracked_parameter = next(
    (name, p)
    for name, p in model.named_parameters()
    if "lora_B" in name and p.requires_grad
)

before_weights = tracked_parameter.detach().clone()

# -----------------------------------------
# 8. Measure loss before training
# -----------------------------------------

model.eval()

with torch.no_grad():
    initial_loss = model(**inputs).loss.item()

print(f"\nInitial loss: {initial_loss:.6f}")

# -----------------------------------------
# 9. Training loop
# -----------------------------------------

print("\nStarting LoRA training...")

model.train()

for step in range(TRAINING_STEPS):

    optimizer.zero_grad(set_to_none=True)

    # Forward pass
    outputs = model(**inputs)
    loss = outputs.loss

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss")

    # Backward pass
    loss.backward()

    # Check gradient availability
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters,
        max_norm=1.0,
        error_if_nonfinite=True,
    )

    # Update LoRA parameters
    optimizer.step()

    print(
        f"Step {step + 1}/{TRAINING_STEPS} | "
        f"Loss: {loss.item():.6f} | "
        f"Grad norm: {gradient_norm.item():.6f}"
    )

# -----------------------------------------
# 10. Verify parameter updates
# -----------------------------------------

weight_change = (
    tracked_parameter.detach() - before_weights
).abs().max().item()

print("\n=== ADAPTER UPDATE CHECK ===")
print("Tracked parameter:", tracked_name)
print(f"Maximum weight change: {weight_change:.10f}")

if weight_change <= 0:
    raise RuntimeError(
        "LoRA weights did not change during training"
    )

# -----------------------------------------
# 11. Measure loss after training
# -----------------------------------------

model.eval()

with torch.no_grad():
    final_loss = model(**inputs).loss.item()

if not math.isfinite(final_loss):
    raise RuntimeError("Final loss is not finite")

print(f"\nInitial loss: {initial_loss:.6f}")
print(f"Final loss:   {final_loss:.6f}")

# -----------------------------------------
# 12. Save adapter
# -----------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nLoRA training smoke test completed!")
print("Adapter saved to:", OUTPUT_DIR)
