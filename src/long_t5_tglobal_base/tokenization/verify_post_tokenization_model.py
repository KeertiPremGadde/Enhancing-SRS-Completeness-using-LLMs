import torch
from transformers import LongT5ForConditionalGeneration, T5TokenizerFast
import os

# === CONFIG ===
MODEL_PATH = "/home/gadde/Thesis/models/pretrained/long-t5-tglobal-base-updated"
EXAMPLE_PT_PATH = "/home/gadde/Thesis/src/long_t5_tglobal_base/tokenization/post_analysis/direct_input_example.pt"

# === LOAD MODEL & TOKENIZER ===
tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH)
model = LongT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
#model.tie_weights()  # <--- ADD THIS LINE IMMEDIATELY
model.lm_head.weight = model.get_input_embeddings().weight
model.eval()

# === DEBUG CHECKPOINT ===
print("\n🔍 Sanity Debug Checkpoint")

# 1. ✅ Confirm token IDs for custom tokens
specials = ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]", "[/SEC]", "[/SUBSEC]", "[/SUBSUBSEC]"]
token_ids = tokenizer.convert_tokens_to_ids(specials)
print("🧠 Custom token IDs:", dict(zip(specials, token_ids)))

# 2. ✅ Confirm none of them are mapped to <unk>
unk_id = tokenizer.unk_token_id
assert all(tid != unk_id for tid in token_ids), "❌ One or more special tokens are mapped to [UNK]!"

# 3. ✅ Confirm LM head and input embeddings are memory-tied
assert model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr(), \
    "❌ LM head is not memory-tied to input embeddings!"
print("✅ Embedding + LM head are memory-tied")

# 4. ✅ Cosine similarity check between initialized vectors and eos/pad
embed = model.get_input_embeddings().weight
eos_embed = embed[tokenizer.eos_token_id]
pad_embed = embed[tokenizer.pad_token_id]
for token, tid in zip(specials, token_ids):
    vec = embed[tid]
    sim_to_eos = torch.nn.functional.cosine_similarity(vec.unsqueeze(0), eos_embed.unsqueeze(0)).item()
    sim_to_pad = torch.nn.functional.cosine_similarity(vec.unsqueeze(0), pad_embed.unsqueeze(0)).item()
    print(f"🔗 {token} cosine similarity — EOS: {sim_to_eos:.4f}, PAD: {sim_to_pad:.4f}")


print("✅ Model and tokenizer loaded.")

# === LOAD TOKENIZED INPUT ===
inputs = torch.load(EXAMPLE_PT_PATH)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
labels = input_ids.clone()  # simple autoencoding task

print("✅ Tokenized input loaded.")

# === FORWARD PASS ===
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
print(f"✅ Forward Pass Successful. Loss = {loss.item():.4f}")

# === BACKWARD PASS (Gradient Flow Check) ===
model.train()
loss.backward()

grads_exist = all(
    param.grad is not None for param in model.parameters() if param.requires_grad
)
print(f"✅ Backward Pass Successful. Gradients Exist: {grads_exist}")

# === LM HEAD ALIGNMENT CHECK ===
# is_tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
# print(f"✅ LM Head Weight Tied with Input Embedding: {is_tied}")

# === Fix weight tying if needed ===
# if not torch.equal(model.get_input_embeddings().weight, model.get_output_embeddings().weight):
#     print("\n🔧 Tying LM Head to Input Embeddings...")
#     #model.get_output_embeddings().weight = model.get_input_embeddings().weight

# is_now_tied = torch.equal(model.get_input_embeddings().weight, model.get_output_embeddings().weight)
# print(f"✅ LM Head Weight Tied After Fix: {is_now_tied}")

# Fix weight tying (proper way)
#model.tie_weights()

# 🔒 Ensure memory-level tying
assert model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr(), \
    "❌ LM head is not properly memory-tied to input embeddings!"
print("✅ Assertion Passed: LM head is memory-tied to input embeddings.")

#model.save_pretrained(MODEL_PATH)  # same or new path
#model.save_pretrained(MODEL_PATH + "-verified")