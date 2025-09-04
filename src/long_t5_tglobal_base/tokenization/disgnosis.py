from transformers import T5Tokenizer, LongT5ForConditionalGeneration
import os

# === Load Tokenizer and Model ===
MODEL_PATH = "/home/gadde/Thesis/models/pretrained/long-t5-tglobal-base-updated"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = LongT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
embed = model.get_input_embeddings().weight

# === Get All Special Tokens ===
all_specials = tokenizer.all_special_tokens
print(f"\n🔍 Total special tokens found: {len(all_specials)}\n")

# === Print token ID and embedding row index ===
rows = []
for token in all_specials:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id >= embed.shape[0]:
        status = "❌ Outside embedding range"
    else:
        status = "✅ Inside embedding range"
    rows.append((token, token_id, status))

# === Nicely formatted output ===
print(f"{'Token':<20} {'Token ID':<10} {'Embedding Row Status'}")
print("=" * 50)
for token, tid, status in sorted(rows, key=lambda x: x[1]):
    print(f"{token:<20} {tid:<10} {status}")
