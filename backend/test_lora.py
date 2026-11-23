
import torch

from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Step 1: Load pretrained Qwen
print("Loading Qwen model on CPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

model.to("cpu")

print("Base model loaded successfully!")

# Step 2: Count original model parameters
original_parameters = sum(
    p.numel() for p in model.parameters()
)

print(f"\nOriginal parameters: {original_parameters:,}")

# Step 3: Define LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Step 4: Attach adapters
print("\nAttaching LoRA adapters...")

model = get_peft_model(model, lora_config)

print("LoRA adapters attached successfully!")

# Step 5: Count trainable parameters
model.print_trainable_parameters()

# Step 6: Inspect LoRA matrices
print("\n=== LORA MATRICES ===")

for name, parameter in model.named_parameters():
    if "lora_A" in name or "lora_B" in name:
        print(name, tuple(parameter.shape))

# Step 7: Verify frozen vs trainable parameters
base_trainable = 0
adapter_trainable = 0

for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        if "lora_" in name:
            adapter_trainable += parameter.numel()
        else:
            base_trainable += parameter.numel()

print("\n=== PARAMETER SUMMARY ===")
print("Original parameters:", f"{original_parameters:,}")
print("Trainable adapter parameters:", f"{adapter_trainable:,}")
print("Other trainable parameters:", f"{base_trainable:,}")

assert base_trainable == 0, (
    "Unexpected non-LoRA parameters are trainable"
)

assert adapter_trainable > 0, (
    "No trainable LoRA parameters found"
)

print("\nSUCCESS: Only LoRA adapters are trainable!")
