import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model on CPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

model.to("cpu")
model.eval()

print("Model loaded successfully!")

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Keep answers concise.",
    },
    {
        "role": "user",
        "content": "Explain the difference between supervised and unsupervised learning.",
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True
)

print("Generating response...")

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\nModel response:")
print(response)
