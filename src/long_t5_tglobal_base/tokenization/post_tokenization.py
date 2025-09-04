# import os
# import json
# import torch
# from transformers import T5TokenizerFast
# from tqdm import tqdm

# # === CONFIG ===
# MODEL_PATH = "/home/gadde/Thesis/models/pretrained/long-t5-tglobal-base-updated"
# INPUT_DIR = "/home/gadde/Thesis/data/complete/complete_srs_00001.json"  # change if needed
# OUTPUT_DIR = "/home/gadde/Thesis/src/long-t5-tglobal-base/tokenization/post_analysis"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# MAX_INPUT_LEN = 1024
# MAX_TARGET_LEN = 1024  # If doing supervised training

# # === LOAD TOKENIZER ===
# tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH)

# # === SPECIAL TOKEN DEBUG ===
# SPECIAL_TOKEN_IDS = {t: tokenizer.convert_tokens_to_ids(t) for t in ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]", "[/SEC]", "[/SUBSEC]", "[/SUBSUBSEC]"]}
# print("Special Token IDs:", SPECIAL_TOKEN_IDS)

# # === FUNCTION: Tokenize and Save ===
# def tokenize_example(text, example_id):
#     encoded = tokenizer(
#         text,
#         max_length=MAX_INPUT_LEN,
#         padding='max_length',
#         truncation=True,
#         return_tensors="pt"
#     )

#     # Save as .pt
#     torch.save(encoded, os.path.join(OUTPUT_DIR, f"{example_id}.pt"))

#     # Save as readable .json for inspection
#     readable = {
#         "input_ids": encoded["input_ids"].tolist(),
#         "attention_mask": encoded["attention_mask"].tolist(),
#         "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
#     }
#     with open(os.path.join(OUTPUT_DIR, f"{example_id}.json"), "w") as f:
#         json.dump(readable, f, indent=2)

# # === MAIN PROCESS ===
# all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt") or f.endswith(".json")]

# for fname in tqdm(all_files, desc="Tokenizing SRS files"):
#     fpath = os.path.join(INPUT_DIR, fname)

#     # Load content
#     if fname.endswith(".json"):
#         with open(fpath) as f:
#             content = json.load(f)
#         text = content.get("text", "")  # Adjust key if needed
#     else:
#         with open(fpath) as f:
#             text = f.read()

#     # Tokenize and Save
#     example_id = os.path.splitext(fname)[0]
#     tokenize_example(text, example_id)

# print("\n✅ Tokenization complete. Results saved to:", OUTPUT_DIR)  # Final confirmation



































import os
import json
import torch
from transformers import T5TokenizerFast

# === CONFIG ===
MODEL_PATH = "/home/gadde/Thesis/models/pretrained/long-t5-tglobal-base-updated"
OUTPUT_DIR = "/home/gadde/Thesis/src/long_t5_tglobal_base/tokenization/post_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_INPUT_LEN = 1024
EXAMPLE_ID = "direct_input_example"

# === LOAD TOKENIZER ===
tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH)

# === SPECIAL TOKEN DEBUG ===
SPECIAL_TOKEN_IDS = {
    t: tokenizer.convert_tokens_to_ids(t) 
    for t in ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]", "[/SEC]", "[/SUBSEC]", "[/SUBSUBSEC]"]
}
print("✅ Special Token IDs:", SPECIAL_TOKEN_IDS)

# Check if any token is mapped to <unk>
unk_id = tokenizer.unk_token_id
for tok, tid in SPECIAL_TOKEN_IDS.items():
    assert tid != unk_id, f"❌ Token {tok} is mapped to [UNK]!"

# Check split behavior (should not split into sub-parts)
split_behavior = {tok: tokenizer.tokenize(tok) for tok in SPECIAL_TOKEN_IDS}
print("🧪 Token split behavior:", split_behavior)


# === DIRECT INPUT TEXT ===
input_data = {
    "text": (
        "[SEC] 1. Introduction\n"
        "[SUBSEC] 1.1 Purpose\n"
        "The ATCGS automates test case generation.\n"
        "[SUBSEC] 1.2 Scope\n"
        "It integrates with CI/CD tools for seamless automation.\n"
        "[SUBSEC] 1.3 Product Perspective\n"
        "[SUBSUBSEC] 1.3.1 System Interfaces\n"
        "REST APIs enable integration.\n"
        "[SUBSUBSEC] 1.3.2 User Interfaces\n"
        "A web UI and CLI are provided.\n"
        "[/SUBSEC]\n[/SEC]"
    )
}
text = input_data["text"]

# === TOKENIZE & SAVE ===
encoded = tokenizer(
    text,
    max_length=MAX_INPUT_LEN,
    padding='max_length',
    truncation=True,
    return_tensors="pt"
)

# === Token count and truncation check ===
non_pad_count = (encoded["attention_mask"][0] == 1).sum().item()
print("🔢 Non-padding token count:", non_pad_count)

# Save .pt
torch.save(encoded, os.path.join(OUTPUT_DIR, f"{EXAMPLE_ID}.pt"))

# Save .json for inspection
readable = {
    "input_ids": encoded["input_ids"].tolist(),
    "attention_mask": encoded["attention_mask"].tolist(),
    "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
}
with open(os.path.join(OUTPUT_DIR, f"{EXAMPLE_ID}.json"), "w") as f:
    json.dump(readable, f, indent=2)

# === Round-trip decoding to inspect post-tokenization structure ===
decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
print("\n🔁 Decoded Output:\n", decoded)

print(f"\n✅ Tokenized output saved to {OUTPUT_DIR} as '{EXAMPLE_ID}.pt' and '{EXAMPLE_ID}.json'")