
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Machine learning is powerful"

# Convert text into token IDs
token_ids = tokenizer.encode(
    text,
    add_special_tokens=False
)

# Convert each token ID into a token
tokens = tokenizer.convert_ids_to_tokens(token_ids)

print("Original text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Total tokens:", len(token_ids))

# Convert token IDs back into text
decoded_text = tokenizer.decode(token_ids)

print("Decoded text:", decoded_text)
