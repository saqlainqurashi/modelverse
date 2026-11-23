
import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading Qwen model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.to("cpu")
model.eval()

print("\nModel loaded successfully!")

# Step 1: Inspect configuration
config = model.config

print("\n=== MODEL CONFIGURATION ===")
print("Hidden size:", config.hidden_size)
print("Transformer layers:", config.num_hidden_layers)
print("Attention heads:", config.num_attention_heads)
print("Key/value heads:", config.num_key_value_heads)

# Step 2: Inspect projection layers
print("\n=== ATTENTION PROJECTIONS ===")

projection_names = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)

projections = []

for name, module in model.named_modules():
    if name.endswith(projection_names):
        projections.append((name, module))

print("Total projection modules:", len(projections))

# Show the first 12
for name, module in projections[:12]:
    print(name)
    print("  Weight shape:", tuple(module.weight.shape))

# Step 3: Count parameters
print("\n=== MODEL PARAMETERS ===")

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(f"Total parameters: {total_parameters:,}")

# Step 4: Inspect first attention layer
print("\n=== FIRST ATTENTION LAYER ===")

for name, module in model.named_modules():
    if name.endswith("layers.0.self_attn"):
        print(module)
        break
