
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 1. Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. Load pretrained model
print("Loading Qwen model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

model.to("cpu")
model.eval()

# 3. Prepare input text
text = "Machine learning is powerful"

inputs = tokenizer(
    text,
    return_tensors="pt",
    add_special_tokens=False,
)

input_ids = inputs["input_ids"]

print("\nOriginal text:", text)
print("Token IDs:", input_ids.tolist())
print("Token count:", input_ids.shape[1])

# 4. Get the model's embedding layer
embedding_layer = model.get_input_embeddings()

print("\nEmbedding layer:")
print(embedding_layer)

print("Embedding matrix shape:")
print(tuple(embedding_layer.weight.shape))

# 5. Convert input IDs into embedding vectors
with torch.inference_mode():
    embeddings = embedding_layer(input_ids)

print("\nInput IDs shape:", tuple(input_ids.shape))
print("Embeddings shape:", tuple(embeddings.shape))

# 6. Inspect one embedding vector
first_vector = embeddings[0, 0]

print("\nFirst token:")
print(tokenizer.convert_ids_to_tokens(
    [input_ids[0, 0].item()]
))

print("\nFirst 10 embedding values:")
print(first_vector[:10].tolist())

print("\nEmbedding dimensions:")
print(first_vector.shape[0])

# 7. Count all model parameters
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nTotal model parameters:")
print(f"{total_parameters:,}")
