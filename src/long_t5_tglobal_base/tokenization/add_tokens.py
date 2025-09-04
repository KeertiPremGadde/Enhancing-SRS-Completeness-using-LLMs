# import torch
# import os
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from datetime import datetime
# from transformers import T5TokenizerFast, LongT5ForConditionalGeneration

# # === CONFIG ===
# MODEL_PATH = "/home/gadde/Thesis/models/pretrained/long-t5-tglobal-base-updated"
# LOG_DIR = "/home/gadde/Thesis/src/long-t5-tglobal-base/tokenization/add_tokens"
# os.makedirs(LOG_DIR, exist_ok=True)
# LOG_PATH = os.path.join(LOG_DIR, "add_token_report.json")

# SPECIAL_TOKENS = ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]", "[/SEC]", "[/SUBSEC]", "[/SUBSUBSEC]"]
# SPECIAL_TOKEN_IDS = {
#     "[SEC]": 32100, "[SUBSEC]": 32101, "[SUBSUBSEC]": 32102,
#     "[/SEC]": 32103, "[/SUBSEC]": 32104, "[/SUBSUBSEC]": 32105
# }

# # === LOAD ===
# tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH)
# model = LongT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
# embed = model.get_input_embeddings().weight
# original_embeddings = embed.detach().clone()
# #lm_head = model.get_output_embeddings().weight
# #original_lm_head = lm_head.detach().clone()
# original_embedding_size = embed.size(0)

# print(f"Original embedding shape: {original_embeddings.shape}")
# print(f"Post-resize embedding shape: {embed.shape}")

# # === PATCH TOKENIZER ===
# for token, tid in SPECIAL_TOKEN_IDS.items():
#     if tokenizer.convert_tokens_to_ids(token) != tid:
#         tokenizer.add_tokens([token], special_tokens=True)
#         tokenizer._tokenizer.add_special_tokens([token])

# existing = tokenizer.additional_special_tokens
# missing = [t for t in SPECIAL_TOKENS if t not in existing]
# tokenizer.additional_special_tokens = existing + missing

# # === VALIDATE ===
# token_id_validation = {
#     t: {
#         "expected_id": i,
#         "actual_id": tokenizer.convert_tokens_to_ids(t),
#         "match": tokenizer.convert_tokens_to_ids(t) == i
#     } for t, i in SPECIAL_TOKEN_IDS.items()
# }

# # # === RESIZE EMBEDDING IF NECESSARY ===
# # if max(SPECIAL_TOKEN_IDS.values()) >= original_embedding_size:
#     # model.resize_token_embeddings(original_embedding_size)

# required_vocab_size = max(SPECIAL_TOKEN_IDS.values()) + 1
# if required_vocab_size > embed.size(0):
#     model.resize_token_embeddings(required_vocab_size)

# # === PRE-INIT CHECK ===
# buffer_embedding_match = {
#     token: {
#         "token_id": tid,
#         "expected_buffer_id": tid,
#         "matches_initial_buffer": torch.allclose(embed[tid], original_embeddings[tid])
#     } for token, tid in SPECIAL_TOKEN_IDS.items()
# }

# # === EMBED INIT ===
# eos_embed = embed[tokenizer.eos_token_id].detach().clone()
# pad_embed = embed[tokenizer.pad_token_id].detach().clone()

# def add_noise(vec, scale=0.01):
#     return vec + scale * torch.randn_like(vec)

# ratios = {
#     "[SEC]": add_noise(0.1 * eos_embed),
#     "[SUBSEC]": add_noise(0.2 * eos_embed),
#     "[SUBSUBSEC]": add_noise(0.3 * eos_embed),
#     "[/SEC]": add_noise(0.9 * eos_embed + 0.1 * pad_embed),
#     "[/SUBSEC]": add_noise(0.8 * eos_embed + 0.2 * pad_embed),
#     "[/SUBSUBSEC]": add_noise(0.7 * eos_embed + 0.3 * pad_embed),
# }

# embed_init_log = {}
# with torch.no_grad():
#     for token, tid in SPECIAL_TOKEN_IDS.items():
#         vec = ratios[token]
#         embed[tid].copy_(vec)
#         embed_init_log[token] = {
#             "id": tid,
#             "norm": vec.norm().item(),
#             "cosine_to_eos": F.cosine_similarity(vec.unsqueeze(0), eos_embed.unsqueeze(0)).item(),
#             "cosine_to_pad": F.cosine_similarity(vec.unsqueeze(0), pad_embed.unsqueeze(0)).item()
#         }

# # === BUFFER SLOT CHECKS ===
# buffer_start = tokenizer.vocab_size
# buffer_end = original_embedding_size
# unchanged = [i for i in range(buffer_start, buffer_end) if i not in SPECIAL_TOKEN_IDS.values() and torch.allclose(embed[i], original_embeddings[i])]
# changed = [i for i in range(buffer_start, buffer_end) if not torch.allclose(embed[i], original_embeddings[i])]

# # === EXPLICIT WEIGHT TYING ===
# model.get_output_embeddings().weight = model.get_input_embeddings().weight
# weights_tied = embed.data_ptr() == model.get_output_embeddings().weight.data_ptr()

# # ✅ Now refresh and clone lm_head properly
# lm_head = model.get_output_embeddings().weight
# original_lm_head = lm_head.detach().clone()

# # === SAVE PATCHED special_tokens_map.json ===
# map_path = os.path.join(MODEL_PATH, "special_tokens_map.json")
# extra_ids = [f"<extra_id_{i}>" for i in range(100)]
# special_map = json.load(open(map_path)) if os.path.exists(map_path) else {}
# special_map["additional_special_tokens"] = SPECIAL_TOKENS + [x for x in extra_ids if x not in SPECIAL_TOKENS]
# json.dump(special_map, open(map_path, "w"), indent=2)

# # === SAVE ARTIFACTS ===
# tokenizer.save_pretrained(MODEL_PATH)
# # Explicit weight tying (again before save)
# model.get_output_embeddings().weight = model.get_input_embeddings().weight
# model.save_pretrained(MODEL_PATH)
# torch.save(original_embeddings, os.path.join(LOG_DIR, "original_embeddings.pt"))
# torch.save(embed, os.path.join(LOG_DIR, "final_embeddings.pt"))
# torch.save(original_lm_head, os.path.join(LOG_DIR, "original_lm_head.pt"))
# torch.save(model.get_output_embeddings().weight, os.path.join(LOG_DIR, "final_lm_head.pt"))

# # === FINAL CHANGE CHECK ===
# changed_rows = (embed - original_embeddings).abs().sum(dim=1) > 1e-6
# changed_indices = changed_rows.nonzero(as_tuple=True)[0].tolist()

# final_lm_head = model.get_output_embeddings().weight.detach().clone()
# lm_head_diff_indices = [
#     i for i in range(original_embedding_size)
#     if not torch.allclose(original_lm_head[i], final_lm_head[i])
# ]
# lm_head_unexpected = [i for i in lm_head_diff_indices if i not in SPECIAL_TOKEN_IDS.values()]

# # === LOG ===
# log = {
#     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     "model_path": MODEL_PATH,
#     "special_tokens": SPECIAL_TOKENS,
#     "special_token_ids": SPECIAL_TOKEN_IDS,
#     "token_id_validation": token_id_validation,
#     "pre_embedding_init_check": buffer_embedding_match,
#     "original": {
#         "vocab_size": 32100,
#         "embedding_shape": [original_embedding_size, embed.size(1)],
#         "lm_head_shape": list(original_lm_head.shape),
#         "weight_tying": weights_tied
#     },
#     "final": {
#         "embedding_shape": list(embed.shape),
#         "weights_tied": weights_tied
#     },
#     "initialized_embeddings": embed_init_log,
#     "unchanged_buffer_slots": unchanged,
#     "changed_buffer_slots": changed,
#     "final_check": {
#         "changed_indices": changed_indices,
#         "unexpected_changes": [i for i in changed_indices if i not in SPECIAL_TOKEN_IDS.values()]
#     },
#     "lm_head_check": {
#         "original_shape": list(original_lm_head.shape),
#         "final_shape": list(final_lm_head.shape),
#         "changed_indices": lm_head_diff_indices,
#         "unexpected_changes": lm_head_unexpected,
#         "matches_input_embedding": weights_tied
#     }
# }

# with open(LOG_PATH, "w") as f:
#     json.dump(log, f, indent=2)

# # === COSINE SIM HEATMAP ===
# token_ids = list(SPECIAL_TOKEN_IDS.values())
# with torch.no_grad():
#     sim_matrix = F.cosine_similarity(
#         embed[token_ids].unsqueeze(1),
#         embed[token_ids].unsqueeze(0),
#         dim=2
#     ).cpu().numpy()

# plt.figure(figsize=(7, 6))
# plt.imshow(sim_matrix, cmap='coolwarm')
# plt.title("Special Token Cosine Similarity")
# plt.colorbar()
# plt.xticks(range(len(SPECIAL_TOKENS)), SPECIAL_TOKENS, rotation=45, ha='right')
# plt.yticks(range(len(SPECIAL_TOKENS)), SPECIAL_TOKENS)
# for i in range(len(SPECIAL_TOKENS)):
#     for j in range(len(SPECIAL_TOKENS)):
#         plt.text(j, i, f"{sim_matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)

# plt.tight_layout()
# plt.savefig(os.path.join(LOG_DIR, "token_similarity_heatmap.png"))

# pd.DataFrame(sim_matrix, index=SPECIAL_TOKENS, columns=SPECIAL_TOKENS).to_csv(
#     os.path.join(LOG_DIR, "token_similarity_matrix.csv")
# )



# import torch
# import os
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from datetime import datetime
# from transformers import T5TokenizerFast, LongT5ForConditionalGeneration

# # === CONFIG ===
# MODEL_PATH = "/home/gadde/Thesis/models/pretrained/long-t5-tglobal-base-updated"
# LOG_DIR = "/home/gadde/Thesis/src/long-t5-tglobal-base/tokenization/add_tokens"
# os.makedirs(LOG_DIR, exist_ok=True)
# LOG_PATH = os.path.join(LOG_DIR, "add_token_report.json")

# SPECIAL_TOKENS = ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]", "[/SEC]", "[/SUBSEC]", "[/SUBSUBSEC]"]

# # === LOAD ===
# tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH)
# model = LongT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
# embed = model.get_input_embeddings().weight
# original_embeddings = embed.detach().clone()
# original_embedding_size = embed.size(0)

# # === ADD TOKENS ===
# special_tokens_dict = {'additional_special_tokens': SPECIAL_TOKENS}
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# model.resize_token_embeddings(len(tokenizer))

# # === EMBED INIT BASE ===
# with torch.no_grad():
#     base_mean = embed.mean(dim=0)
#     eos_embed = embed[tokenizer.eos_token_id].detach().clone()
#     pad_embed = embed[tokenizer.pad_token_id].detach().clone()

#     init_ratios = {
#         "[SEC]": 1.0,
#         "[SUBSEC]": 0.95,
#         "[SUBSUBSEC]": 0.9,
#         "[/SEC]": 0.5,
#         "[/SUBSEC]": 0.45,
#         "[/SUBSUBSEC]": 0.4,
#     }

#     embed_init_log = {}
#     for token, ratio in init_ratios.items():
#         tid = tokenizer.convert_tokens_to_ids(token)
#         vec = base_mean * ratio
#         embed[tid].copy_(vec)
#         embed_init_log[token] = {
#             "id": tid,
#             "norm": vec.norm().item(),
#             "cosine_to_eos": F.cosine_similarity(vec.unsqueeze(0), eos_embed.unsqueeze(0)).item(),
#             "cosine_to_pad": F.cosine_similarity(vec.unsqueeze(0), pad_embed.unsqueeze(0)).item()
#         }

# # === BUFFER SLOT CHECKS ===
# buffer_start = original_embedding_size
# buffer_end = embed.size(0)
# unchanged = [i for i in range(buffer_start, buffer_end)
#              if torch.allclose(embed[i], original_embeddings[min(i, original_embedding_size - 1)])]
# changed = [i for i in range(buffer_start, buffer_end)
#            if not torch.allclose(embed[i], original_embeddings[min(i, original_embedding_size - 1)])]

# # === WEIGHT TYING ===
# model.get_output_embeddings().weight = model.get_input_embeddings().weight
# weights_tied = embed.data_ptr() == model.get_output_embeddings().weight.data_ptr()

# # === LM HEAD SNAPSHOT ===
# lm_head = model.get_output_embeddings().weight
# original_lm_head = lm_head.detach().clone()
# final_lm_head = model.get_output_embeddings().weight.detach().clone()

# # === CHANGE CHECK ===
# changed_rows = (embed - original_embeddings).abs().sum(dim=1) > 1e-6
# changed_indices = changed_rows.nonzero(as_tuple=True)[0].tolist()

# # lm_head_diff_indices = [i for i in range(original_embedding_size)
# #                         if not torch.allclose(original_lm_head[i], final_lm_head[i])]
# min_len = min(original_lm_head.size(0), final_lm_head.size(0))
# lm_head_diff_indices = [
#     i for i in range(min_len)
#     if not torch.allclose(original_lm_head[i], final_lm_head[i])
# ]
# lm_head_unexpected = [i for i in lm_head_diff_indices if tokenizer.convert_ids_to_tokens(i) not in SPECIAL_TOKENS]

# # === SAVE FINAL MODEL + TOKENS ===
# tokenizer.save_pretrained(MODEL_PATH)
# model.save_pretrained(MODEL_PATH)
# torch.save(original_embeddings, os.path.join(LOG_DIR, "original_embeddings.pt"))
# torch.save(embed, os.path.join(LOG_DIR, "final_embeddings.pt"))
# torch.save(original_lm_head, os.path.join(LOG_DIR, "original_lm_head.pt"))
# torch.save(final_lm_head, os.path.join(LOG_DIR, "final_lm_head.pt"))

# # === LOGGING ===
# log = {
#     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     "model_path": MODEL_PATH,
#     "special_tokens": SPECIAL_TOKENS,
#     "init_ratios": init_ratios,
#     "num_added_tokens": num_added_toks,
#     "original": {
#         "embedding_shape": list(original_embeddings.shape),
#         "vocab_size": original_embedding_size
#     },
#     "final": {
#         "embedding_shape": list(embed.shape),
#         "weights_tied": weights_tied
#     },
#     "initialized_embeddings": embed_init_log,
#     "unchanged_buffer_slots": unchanged,
#     "changed_buffer_slots": changed,
#     "final_check": {
#         "changed_indices": changed_indices,
#         "unexpected_changes": [i for i in changed_indices if tokenizer.convert_ids_to_tokens(i) not in SPECIAL_TOKENS]
#     },
#     "lm_head_check": {
#         "original_shape": list(original_lm_head.shape),
#         "final_shape": list(final_lm_head.shape),
#         "changed_indices": lm_head_diff_indices,
#         "unexpected_changes": lm_head_unexpected,
#         "matches_input_embedding": weights_tied
#     }
# }
# with open(LOG_PATH, "w") as f:
#     json.dump(log, f, indent=2)

# # === COSINE SIM HEATMAP ===
# token_ids = [tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS]
# with torch.no_grad():
#     sim_matrix = F.cosine_similarity(
#         embed[token_ids].unsqueeze(1),
#         embed[token_ids].unsqueeze(0),
#         dim=2
#     ).cpu().numpy()

# plt.figure(figsize=(7, 6))
# plt.imshow(sim_matrix, cmap='coolwarm')
# plt.title("Special Token Cosine Similarity")
# plt.colorbar()
# plt.xticks(range(len(SPECIAL_TOKENS)), SPECIAL_TOKENS, rotation=45, ha='right')
# plt.yticks(range(len(SPECIAL_TOKENS)), SPECIAL_TOKENS)
# for i in range(len(SPECIAL_TOKENS)):
#     for j in range(len(SPECIAL_TOKENS)):
#         plt.text(j, i, f"{sim_matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
# plt.tight_layout()
# plt.savefig(os.path.join(LOG_DIR, "token_similarity_heatmap.png"))
# pd.DataFrame(sim_matrix, index=SPECIAL_TOKENS, columns=SPECIAL_TOKENS).to_csv(
#     os.path.join(LOG_DIR, "token_similarity_matrix.csv")
# )

# # === Norm Distribution Plot ===
# def plot_embedding_norms(tokenizer, model, special_tokens):
#     with torch.no_grad():
#         embeddings = model.get_input_embeddings().weight
#         norms = torch.norm(embeddings, dim=1).cpu().numpy()

#         plt.figure(figsize=(12, 6))
#         plt.hist(norms, bins=100, alpha=0.75, label='All token embeddings')
#         for token in special_tokens:
#             token_id = tokenizer.convert_tokens_to_ids(token)
#             token_norm = norms[token_id]
#             plt.axvline(token_norm, linestyle='--', label=f'{token} (norm: {token_norm:.4f})')
#         plt.title('Embedding Norms: Special Tokens vs. All Tokens')
#         plt.xlabel('Norm value')
#         plt.ylabel('Frequency')
#         plt.legend()
#         plt.savefig(os.path.join(LOG_DIR, "embedding_norm_distribution.png"))
#         plt.close()

# plot_embedding_norms(tokenizer, model, SPECIAL_TOKENS)
























import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from transformers import LongT5ForConditionalGeneration
from sklearn.decomposition import PCA
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import T5TokenizerFast

# === CONFIG ===
MODEL_PATH = "/home/gadde/Thesis/models/pretrained/long-t5-tglobal-base-updated"
LOG_DIR = "/home/gadde/Thesis/src/long_t5_tglobal_base/tokenization/add_tokens"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "add_token_report.json")

SPECIAL_TOKENS = ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]", "[/SEC]", "[/SUBSEC]", "[/SUBSUBSEC]"]

# === LOAD ===
tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH)
model = LongT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
embed = model.get_input_embeddings().weight
original_embeddings = embed.detach().clone()
original_embedding_size = embed.size(0)

# === PRESERVE ORIGINAL <extra_id_*> TOKENS ===
existing_extra_ids = [tok for tok in tokenizer.additional_special_tokens if tok.startswith("<extra_id_")]
combined_tokens = SPECIAL_TOKENS + [tok for tok in existing_extra_ids if tok not in SPECIAL_TOKENS]

# === ADD SPECIAL TOKENS ===
#tokenizer.add_special_tokens({"additional_special_tokens": combined_tokens})
num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": combined_tokens})

# 🧹 Remove old tokenizer.json to regenerate cleanly
tokenizer_json_path = os.path.join(MODEL_PATH, "tokenizer.json")
if os.path.exists(tokenizer_json_path):
    os.remove(tokenizer_json_path)

# ✅ Write tokenizer_config.json explicitly (before saving tokenizer)
tokenizer_config_path = os.path.join(MODEL_PATH, "tokenizer_config.json")
tokenizer_config_data = {
    "tokenizer_class": "T5TokenizerFast",
    "clean_up_tokenization_spaces": True,
    "model_max_length": 1000000000000000019884624838656
}
with open(tokenizer_config_path, "w") as f:
    json.dump(tokenizer_config_data, f, indent=2)


tokenizer.save_pretrained(MODEL_PATH)  # This writes special_tokens_map.json
model.resize_token_embeddings(len(tokenizer))  # Resize for newly added tokens

# === RE-TIE LM HEAD (AFTER RESIZE!) ===
#model.lm_head = torch.nn.Linear(model.model_dim, model.shared.num_embeddings, bias=False)
#model.lm_head.weight = model.get_input_embeddings().weight  # Explicit tying

# Manually re-tie LM head to input embedding (this works for T5 variants)
model.lm_head.weight = model.get_input_embeddings().weight

#model.save_pretrained(MODEL_PATH)

# === Verify tying worked ===
assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(), \
    "❌ LM head is NOT memory-tied to input embeddings!"



#model.tie_weights()  # Proper memory-level tying
#assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(), \
#    "❌ LM head is NOT memory-tied to input embeddings!"
weights_tied = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()

# === INIT SPECIAL TOKEN EMBEDDINGS ===
eos_embed = embed[tokenizer.eos_token_id].detach().clone()
pad_embed = embed[tokenizer.pad_token_id].detach().clone()

#def add_noise(vec, scale=0.01):
#    return vec + scale * torch.randn_like(vec)

# def rescale_to_mean_norm(vec, target_norm=280):
#     return vec * (target_norm / vec.norm())


def rescale_to_mean_norm(vec, target_norm, eps=1e-8):
    return vec * (target_norm / (vec.norm() + eps))

# ratios = {
#     "[SEC]": 0.9,
#     "[SUBSEC]": 0.8,
#     "[SUBSUBSEC]": 0.7,
#     "[/SEC]": 0.6,
#     "[/SUBSEC]": 0.5,
#     "[/SUBSUBSEC]": 0.4,
# }

# embed_init_log = {}
# with torch.no_grad():
#     for token, ratio in ratios.items():
#         tid = tokenizer.convert_tokens_to_ids(token)
#         #vec = rescale_to_mean_norm(ratio * eos_embed + (1 - ratio) * pad_embed)
#         #vec = rescale_to_mean_norm(add_noise(ratio * eos_embed + (1 - ratio) * pad_embed, scale=0.01))


#         base_vec = ratio * eos_embed + (1 - ratio) * pad_embed
#         # Optional noise, uncomment if desired
#         # base_vec = add_noise(base_vec, scale=0.01)

#         target_norm = eos_embed.norm()
#         vec = base_vec / base_vec.norm() * target_norm  # Preserves cosine similarity!


#         embed[tid].copy_(vec)
#         embed_init_log[token] = {
#             "id": tid,
#             "norm": vec.norm().item(),
#             "cosine_to_eos": F.cosine_similarity(vec.unsqueeze(0), eos_embed.unsqueeze(0)).item(),
#             "cosine_to_pad": F.cosine_similarity(vec.unsqueeze(0), pad_embed.unsqueeze(0)).item()
#         }

# === INIT SPECIAL TOKEN EMBEDDINGS USING MEAN-BASED STRATEGY ===
with torch.no_grad():
    base_vec = embed[:original_embedding_size].mean(dim=0).detach()
    target_norm = base_vec.norm()

    ratios = {
        "[SEC]": 1.0,
        "[SUBSEC]": 0.95,
        "[SUBSUBSEC]": 0.90,
        "[/SEC]": 0.85,
        "[/SUBSEC]": 0.80,
        "[/SUBSUBSEC]": 0.75,
    }

    embed_init_log = {}
    for token, ratio in ratios.items():
        tid = tokenizer.convert_tokens_to_ids(token)
        vec = rescale_to_mean_norm(ratio * base_vec, target_norm=target_norm)
        embed[tid].copy_(vec)
        embed_init_log[token] = {
            "id": tid,
            "norm": vec.norm().item(),
            "cosine_to_mean": F.cosine_similarity(vec.unsqueeze(0), base_vec.unsqueeze(0)).item()
        }

model.save_pretrained(MODEL_PATH)


# === VALIDATION ===
token_id_validation = {
    t: {
        "actual_id": tokenizer.convert_tokens_to_ids(t),
        "in_tokenizer": tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id
    } for t in SPECIAL_TOKENS
}

buffer_embedding_match = {
    token: {
        "token_id": tokenizer.convert_tokens_to_ids(token),
        "matches_initial": torch.allclose(
            embed[tokenizer.convert_tokens_to_ids(token)],
            original_embeddings[min(tokenizer.convert_tokens_to_ids(token), original_embedding_size - 1)]
        )
    } for token in SPECIAL_TOKENS
}

# === SAVE PATCHED TOKENIZER MAP ===
# map_path = os.path.join(MODEL_PATH, "special_tokens_map.json")
# special_map = json.load(open(map_path)) if os.path.exists(map_path) else {}
# special_map["additional_special_tokens"] = combined_tokens
# json.dump(special_map, open(map_path, "w"), indent=2)

# === DIAGNOSTIC ===
original_lm_head = model.lm_head.weight.detach().clone()
final_lm_head = model.get_input_embeddings().weight.detach().clone()

changed_rows = (embed - original_embeddings).abs().sum(dim=1) > 1e-6
changed_indices = changed_rows.nonzero(as_tuple=True)[0].tolist()

lm_head_diff_indices = [
    i for i in range(min(original_lm_head.shape[0], final_lm_head.shape[0]))
    if not torch.allclose(original_lm_head[i], final_lm_head[i])
]
lm_head_unexpected = [i for i in lm_head_diff_indices if tokenizer.convert_ids_to_tokens(i) not in SPECIAL_TOKENS]


model.save_pretrained(MODEL_PATH)

# === SAVE FINAL STATE ===
#tokenizer.save_pretrained(MODEL_PATH)
fast_tokenizer_path = os.path.join(MODEL_PATH, "tokenizer.json")

# Save tokenizer properly — whether it's slow or fast
#tokenizer.config.tokenizer_class = "T5TokenizerFast"  # ✅ Ensure correct class again
#tokenizer.save_pretrained(MODEL_PATH)

if os.path.exists(fast_tokenizer_path):
    tok = Tokenizer.from_file(fast_tokenizer_path)
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tok)
    wrapped.add_special_tokens({"additional_special_tokens": combined_tokens}) #failsafe step, not a bug
    wrapped.save_pretrained(MODEL_PATH)

# ✅ Inspect special_tokens_map.json content after saving
print("\n📄 Contents of special_tokens_map.json:")
with open(os.path.join(MODEL_PATH, "special_tokens_map.json")) as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
    
torch.save(original_embeddings, os.path.join(LOG_DIR, "original_embeddings.pt"))
torch.save(embed, os.path.join(LOG_DIR, "final_embeddings.pt"))
torch.save(original_lm_head, os.path.join(LOG_DIR, "original_lm_head.pt"))
torch.save(final_lm_head, os.path.join(LOG_DIR, "final_lm_head.pt"))

# === LOG REPORT ===
log = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_path": MODEL_PATH,
    "special_tokens": SPECIAL_TOKENS,
    "added_token_list": combined_tokens,
    "existing_extra_ids": existing_extra_ids,
    "init_ratios": ratios,
    "num_added_tokens": num_added_toks,
    "token_id_validation": token_id_validation,
    "pre_embedding_init_check": buffer_embedding_match,
    "original": {
        "embedding_shape": list(original_embeddings.shape),
        "vocab_size": original_embedding_size
    },
    "final": {
        "embedding_shape": list(embed.shape),
        "weights_tied": weights_tied
    },
    "initialized_embeddings": embed_init_log,
    "final_check": {
        "changed_indices": changed_indices,
        "unexpected_changes": [i for i in changed_indices if tokenizer.convert_ids_to_tokens(i) not in SPECIAL_TOKENS]
    },
    "lm_head_check": {
        "original_shape": list(original_lm_head.shape),
        "final_shape": list(final_lm_head.shape),
        "changed_indices": lm_head_diff_indices,
        "unexpected_changes": lm_head_unexpected,
        "matches_input_embedding": weights_tied
    }
}

token_id_ranges = {
    "min_token_id": min([tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS]),
    "max_token_id": max([tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS])
}
log["vocabulary_range"] = token_id_ranges

log["decoded_tokens"] = {
    tokenizer.convert_tokens_to_ids(t): tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(t))
    for t in SPECIAL_TOKENS
}

with open(LOG_PATH, "w") as f:
    json.dump(log, f, indent=2)

# === PLOTS ===
token_ids = [tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS]
with torch.no_grad():
    sim_matrix = F.cosine_similarity(embed[token_ids].unsqueeze(1), embed[token_ids].unsqueeze(0), dim=2).cpu().numpy()

plt.figure(figsize=(7, 6))
plt.imshow(sim_matrix, cmap='coolwarm')
plt.title("Special Token Cosine Similarity")
plt.colorbar()
plt.xticks(range(len(SPECIAL_TOKENS)), SPECIAL_TOKENS, rotation=45, ha='right')
plt.yticks(range(len(SPECIAL_TOKENS)), SPECIAL_TOKENS)
for i in range(len(SPECIAL_TOKENS)):
    for j in range(len(SPECIAL_TOKENS)):
        plt.text(j, i, f"{sim_matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "token_similarity_heatmap.png"))
pd.DataFrame(sim_matrix, index=SPECIAL_TOKENS, columns=SPECIAL_TOKENS).to_csv(
    os.path.join(LOG_DIR, "token_similarity_matrix.csv")
)

# Norm distribution
with torch.no_grad():
    norms = torch.norm(embed, dim=1).cpu().numpy()
    plt.figure(figsize=(12, 6))
    plt.hist(norms, bins=100, alpha=0.75, label='All token embeddings')
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_norm = norms[token_id]
        plt.axvline(token_norm, linestyle='--', label=f'{token} (norm: {token_norm:.4f})')
    plt.title('Embedding Norms: Special Tokens vs. All Tokens')
    plt.xlabel('Norm value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, "embedding_norm_distribution.png"))
    plt.close()

# PCA
with torch.no_grad():
    selected = embed[token_ids].cpu().numpy()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(selected)
    plt.figure(figsize=(7, 6))
    for i, token in enumerate(SPECIAL_TOKENS):
        plt.scatter(reduced[i, 0], reduced[i, 1], label=token)
        plt.text(reduced[i, 0], reduced[i, 1], token, fontsize=9, ha='center')
    plt.title("PCA Projection of Special Token Embeddings")
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, "embedding_pca_plot.png"))
    plt.close()

print(f"✅ Done. Special tokens added and properly initialized.")
print(f"🔁 LM Head and Input Embeddings are tied: {weights_tied}")
print(f"📁 Logs, plots, and model saved in: {LOG_DIR}")


# === Force tokenizer.json to be regenerated from correct class ===
print("\n🧹 Regenerating tokenizer.json with correct class (T5TokenizerFast)...")
tokenizer_json_path = os.path.join(MODEL_PATH, "tokenizer.json")
if os.path.exists(tokenizer_json_path):
    os.remove(tokenizer_json_path)

# Reload using correct class
tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
print("✅ tokenizer.json regenerated using T5TokenizerFast.\n")


# === Sanity check: test token IDs ===
print("\n🔍 Sanity Check: Special Token IDs")
tok = T5TokenizerFast.from_pretrained(MODEL_PATH)
print(tok.convert_tokens_to_ids(["[SEC]", "[SUBSEC]", "[SUBSUBSEC]", "[/SEC]", "[/SUBSEC]", "[/SUBSUBSEC]"]))
assert tok.convert_tokens_to_ids("[SEC]") == tokenizer.convert_tokens_to_ids("[SEC]")
print("🧠 Tokenizer class used in tokenizer_config.json:", tokenizer_config_data["tokenizer_class"])







# import torch
# import os
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from datetime import datetime
# from transformers import T5TokenizerFast, LongT5ForConditionalGeneration

# # === CONFIG ===
# MODEL_PATH = "/home/gadde/Thesis/models/pretrained/long-t5-tglobal-base-updated"
# LOG_DIR = "/home/gadde/Thesis/src/long-t5-tglobal-base/tokenization/add_tokens"
# os.makedirs(LOG_DIR, exist_ok=True)
# LOG_PATH = os.path.join(LOG_DIR, "add_token_report.json")

# SPECIAL_TOKENS = ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]", "[/SEC]", "[/SUBSEC]", "[/SUBSUBSEC]"]
# SPECIAL_TOKEN_IDS = {
#     "[SEC]": 32100, "[SUBSEC]": 32101, "[SUBSUBSEC]": 32102,
#     "[/SEC]": 32103, "[/SUBSEC]": 32104, "[/SUBSUBSEC]": 32105
# }

# # === LOAD ===
# tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH)
# model = LongT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
# embed = model.get_input_embeddings().weight
# original_embeddings = embed.detach().clone()
# original_embedding_size = embed.size(0)

# print(f"Original embedding shape: {original_embeddings.shape}")
# print(f"Post-resize embedding shape: {embed.shape}")

# # === PATCH TOKENIZER ===
# for token, tid in SPECIAL_TOKEN_IDS.items():
#     if tokenizer.convert_tokens_to_ids(token) != tid:
#         tokenizer.add_tokens([token], special_tokens=True)
#         tokenizer._tokenizer.add_special_tokens([token])

# existing = tokenizer.additional_special_tokens
# missing = [t for t in SPECIAL_TOKENS if t not in existing]
# tokenizer.additional_special_tokens = existing + missing

# # === VALIDATE ===
# token_id_validation = {
#     t: {
#         "expected_id": i,
#         "actual_id": tokenizer.convert_tokens_to_ids(t),
#         "match": tokenizer.convert_tokens_to_ids(t) == i
#     } for t, i in SPECIAL_TOKEN_IDS.items()
# }

# # === RESIZE EMBEDDING IF NECESSARY ===
# required_vocab_size = max(SPECIAL_TOKEN_IDS.values()) + 1
# if required_vocab_size > embed.size(0):
#     model.resize_token_embeddings(required_vocab_size)

# # === PRE-INIT CHECK ===
# buffer_embedding_match = {
#     token: {
#         "token_id": tid,
#         "expected_buffer_id": tid,
#         "matches_initial_buffer": torch.allclose(embed[tid], original_embeddings[tid])
#     } for token, tid in SPECIAL_TOKEN_IDS.items()
# }

# # === EMBED INIT ===
# eos_embed = embed[tokenizer.eos_token_id].detach().clone()
# pad_embed = embed[tokenizer.pad_token_id].detach().clone()

# def add_noise(vec, scale=0.01):
#     return vec + scale * torch.randn_like(vec)

# ratios = {
#     "[SEC]": add_noise(0.1 * eos_embed),
#     "[SUBSEC]": add_noise(0.2 * eos_embed),
#     "[SUBSUBSEC]": add_noise(0.3 * eos_embed),
#     "[/SEC]": add_noise(0.9 * eos_embed + 0.1 * pad_embed),
#     "[/SUBSEC]": add_noise(0.8 * eos_embed + 0.2 * pad_embed),
#     "[/SUBSUBSEC]": add_noise(0.7 * eos_embed + 0.3 * pad_embed),
# }

# embed_init_log = {}
# with torch.no_grad():
#     for token, tid in SPECIAL_TOKEN_IDS.items():
#         vec = ratios[token]
#         embed[tid].copy_(vec)
#         embed_init_log[token] = {
#             "id": tid,
#             "norm": vec.norm().item(),
#             "cosine_to_eos": F.cosine_similarity(vec.unsqueeze(0), eos_embed.unsqueeze(0)).item(),
#             "cosine_to_pad": F.cosine_similarity(vec.unsqueeze(0), pad_embed.unsqueeze(0)).item()
#         }

# # === BUFFER SLOT CHECKS ===
# buffer_start = tokenizer.vocab_size
# buffer_end = original_embedding_size
# unchanged = [i for i in range(buffer_start, buffer_end) if i not in SPECIAL_TOKEN_IDS.values() and torch.allclose(embed[i], original_embeddings[i])]
# changed = [i for i in range(buffer_start, buffer_end) if not torch.allclose(embed[i], original_embeddings[i])]

# # ✅ Now refresh and clone lm_head properly
# lm_head = model.get_output_embeddings().weight
# original_lm_head = lm_head.detach().clone()

# # === SAVE PATCHED special_tokens_map.json ===
# map_path = os.path.join(MODEL_PATH, "special_tokens_map.json")
# extra_ids = [f"<extra_id_{i}>" for i in range(100)]
# special_map = json.load(open(map_path)) if os.path.exists(map_path) else {}
# special_map["additional_special_tokens"] = SPECIAL_TOKENS + [x for x in extra_ids if x not in SPECIAL_TOKENS]
# json.dump(special_map, open(map_path, "w"), indent=2)

# # === SAVE ARTIFACTS ===
# tokenizer.save_pretrained(MODEL_PATH)

# # === FINAL WEIGHT TYING BEFORE SAVE ===
# model.get_output_embeddings().weight = model.get_input_embeddings().weight
# model.save_pretrained(MODEL_PATH)

# # Final check if weights are tied
# weights_tied = embed.data_ptr() == model.get_output_embeddings().weight.data_ptr()

# torch.save(original_embeddings, os.path.join(LOG_DIR, "original_embeddings.pt"))
# torch.save(embed, os.path.join(LOG_DIR, "final_embeddings.pt"))
# torch.save(original_lm_head, os.path.join(LOG_DIR, "original_lm_head.pt"))
# torch.save(model.get_output_embeddings().weight, os.path.join(LOG_DIR, "final_lm_head.pt"))

# # === FINAL CHANGE CHECK ===
# changed_rows = (embed - original_embeddings).abs().sum(dim=1) > 1e-6
# changed_indices = changed_rows.nonzero(as_tuple=True)[0].tolist()

# final_lm_head = model.get_output_embeddings().weight.detach().clone()
# lm_head_diff_indices = [
#     i for i in range(original_embedding_size)
#     if not torch.allclose(original_lm_head[i], final_lm_head[i])
# ]
# lm_head_unexpected = [i for i in lm_head_diff_indices if i not in SPECIAL_TOKEN_IDS.values()]

# # === LOG ===
# log = {
#     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     "model_path": MODEL_PATH,
#     "special_tokens": SPECIAL_TOKENS,
#     "special_token_ids": SPECIAL_TOKEN_IDS,
#     "token_id_validation": token_id_validation,
#     "pre_embedding_init_check": buffer_embedding_match,
#     "original": {
#         "vocab_size": 32100,
#         "embedding_shape": [original_embedding_size, embed.size(1)],
#         "lm_head_shape": list(original_lm_head.shape),
#         "weight_tying": weights_tied
#     },
#     "final": {
#         "embedding_shape": list(embed.shape),
#         "weights_tied": weights_tied
#     },
#     "initialized_embeddings": embed_init_log,
#     "unchanged_buffer_slots": unchanged,
#     "changed_buffer_slots": changed,
#     "final_check": {
#         "changed_indices": changed_indices,
#         "unexpected_changes": [i for i in changed_indices if i not in SPECIAL_TOKEN_IDS.values()]
#     },
#     "lm_head_check": {
#         "original_shape": list(original_lm_head.shape),
#         "final_shape": list(final_lm_head.shape),
#         "changed_indices": lm_head_diff_indices,
#         "unexpected_changes": lm_head_unexpected,
#         "matches_input_embedding": weights_tied
#     }
# }

# with open(LOG_PATH, "w") as f:
#     json.dump(log, f, indent=2)

# # === COSINE SIM HEATMAP ===
# token_ids = list(SPECIAL_TOKEN_IDS.values())
# with torch.no_grad():
#     sim_matrix = F.cosine_similarity(
#         embed[token_ids].unsqueeze(1),
#         embed[token_ids].unsqueeze(0),
#         dim=2
#     ).cpu().numpy()

# plt.figure(figsize=(7, 6))
# plt.imshow(sim_matrix, cmap='coolwarm')
# plt.title("Special Token Cosine Similarity")
# plt.colorbar()
# plt.xticks(range(len(SPECIAL_TOKENS)), SPECIAL_TOKENS, rotation=45, ha='right')
# plt.yticks(range(len(SPECIAL_TOKENS)), SPECIAL_TOKENS)
# for i in range(len(SPECIAL_TOKENS)):
#     for j in range(len(SPECIAL_TOKENS)):
#         plt.text(j, i, f"{sim_matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)

# plt.tight_layout()
# plt.savefig(os.path.join(LOG_DIR, "token_similarity_heatmap.png"))

# pd.DataFrame(sim_matrix, index=SPECIAL_TOKENS, columns=SPECIAL_TOKENS).to_csv(
#     os.path.join(LOG_DIR, "token_similarity_matrix.csv")
# )
