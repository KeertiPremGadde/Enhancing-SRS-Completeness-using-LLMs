import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration

# === Load model and tokenizer paths ===
tokenizer_path = "/home/gadde/Thesis/models/pretrained/led-base-16384-updated"  # update this path
model_path = "/home/gadde/Thesis/models/pretrained/led-base-16384-updated"          # update this path

# === Load tokenizer and model ===
tokenizer = LEDTokenizer.from_pretrained(tokenizer_path)
model = LEDForConditionalGeneration.from_pretrained(model_path)

# === Compare vocab sizes ===
tokenizer_vocab_size = len(tokenizer)
model_vocab_size = model.get_input_embeddings().num_embeddings

print("=== Tokenizer vs Model Embedding Size ===")
print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
print(f"Model vocab size:     {model_vocab_size}")

if tokenizer_vocab_size != model_vocab_size:
    print("❌ Mismatch detected — Model embedding size does not match tokenizer vocab.")
else:
    print("✅ Vocab size match confirmed.")

# === Check special token IDs ===
special_tokens = ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']
print("\n=== Special Token ID Check ===")
for token in special_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"{token} → Token ID: {token_id}")
    if token_id >= model_vocab_size:
        print(f"❌ Token ID {token_id} exceeds model vocab size!")
    elif token_id == tokenizer.unk_token_id:
        print(f"❌ Token {token} is unknown (UNK) — not added to tokenizer?")
    else:
        print(f"✅ Token {token} is valid and within range.")

# === Optional: Inspect sample label input ===
sample_text = "[SEC] 1. Introduction [SUBSEC] 1.1 Purpose [SUBSUBSEC] 1.1.1 Detail"
encoded = tokenizer(
    sample_text,
    max_length=1024,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)
max_token_id = encoded['input_ids'].max().item()
print(f"\n=== Max Token ID in Sample Text ===")
print(f"Max token ID: {max_token_id}")
print(f"Model vocab size: {model_vocab_size}")
if max_token_id >= model_vocab_size:
    print("❌ Sample token ID exceeds model vocab — training will crash.")
else:
    print("✅ Sample is safe for training.")

# === DONE ===
