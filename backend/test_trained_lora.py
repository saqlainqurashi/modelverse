
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "lora-smoke-test"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading original Qwen model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)
base_model.to("cpu")
base_model.eval()

question = "What is supervised learning?"

messages = [
    {
        "role": "system",
        "content": (
            "You are an AI engineering tutor. "
            "Explain concepts clearly with practical examples."
        ),
    },
    {"role": "user", "content": question},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)


def generate_answer(model):
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][
        inputs["input_ids"].shape[-1]:
    ]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )


print("\n=== ORIGINAL QWEN ===")
print(generate_answer(base_model))

print("\nLoading trained LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_PATH),
    is_trainable=False,
)

model.eval()

print("\n=== QWEN + TRAINED LORA ===")
print(generate_answer(model))

print("\nComparison completed successfully!")
