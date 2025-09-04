import os
import json
import torch
import numpy as np
from datetime import datetime
from transformers import T5Tokenizer, LongT5ForConditionalGeneration

OUTPUT_DIR = "/home/gadde/Thesis/src/long_t5_tglobal_base/tokenization/pre_analysis"
MODEL_PATH = "/home/gadde/Thesis/models/pretrained/long-t5-tglobal-base-updated"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_json(data, filename):
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        json.dump(data, f, indent=2)

def save_tensor(tensor, name):
    torch.save(tensor, os.path.join(OUTPUT_DIR, f"{name}.pt"))
    np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), tensor.detach().cpu().numpy())

def inspect_token(tokenizer, embeddings, token_str):
    if token_str not in tokenizer.get_vocab():
        return {
            "token": token_str,
            "exists": False
        }

    token_id = tokenizer.convert_tokens_to_ids(token_str)
    vector = embeddings[token_id].detach().cpu().numpy()

    return {
        "token": token_str,
        "exists": True,
        "id": token_id,
        "embedding": {
            "mean": float(np.mean(vector)),
            "std": float(np.std(vector)),
            "min": float(np.min(vector)),
            "max": float(np.max(vector)),
            "first_5_values": vector[:5].tolist()
        }
    }

def run_analysis():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = LongT5ForConditionalGeneration.from_pretrained(MODEL_PATH)

    embedding_weight = model.get_input_embeddings().weight
    lm_head_weight = model.lm_head.weight

    assert embedding_weight.shape == lm_head_weight.shape, "Mismatch between shared and lm_head weights!"

    analysis = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": MODEL_PATH,
        "vocabulary": {
            "vocab_size": len(tokenizer),
            "special_tokens": tokenizer.special_tokens_map,
            "all_special_tokens": tokenizer.all_special_tokens,
            "last_token_id": len(tokenizer) - 1,
        },
        "embeddings": {
            "input_shape": list(embedding_weight.shape),
            "output_shape": list(lm_head_weight.shape),
            "weight_stats": {
                "mean": float(embedding_weight.mean()),
                "std": float(embedding_weight.std()),
                "min": float(embedding_weight.min()),
                "max": float(embedding_weight.max())
            }
        },
        "architecture": {
            "hidden_size": model.config.d_model,
            "vocab_size": model.config.vocab_size,
            "max_length": model.config.n_positions if hasattr(model.config, "n_positions") else "not defined",
            "model_type": model.config.model_type,
            "weight_tying": model.config.tie_word_embeddings if hasattr(model.config, "tie_word_embeddings") else True
        },
        "special_token_embeddings": {},
        "custom_token_check": {},
    }
    
    # ✅ INSERT HERE — 1. Last 50 tokens in vocab
    vocab_items = list(tokenizer.get_vocab().items())
    vocab_items_sorted = sorted(vocab_items, key=lambda x: x[1])  # Sort by ID
    analysis["vocabulary"]["last_50_tokens"] = vocab_items_sorted[-50:]

    # ✅ INSERT HERE — 2. Extra token vectors analysis
    extra_token_vectors = []
    for i in range(100):
        token = f"<extra_id_{i}>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        vector = embedding_weight[token_id].detach().cpu().numpy()
        extra_token_vectors.append(vector)

    extra_token_matrix = np.stack(extra_token_vectors)
    analysis["extra_token_embeddings"] = {
        "mean_of_means": float(np.mean(extra_token_matrix)),
        "std_of_means": float(np.std(np.mean(extra_token_matrix, axis=1))),
        "embedding_shape": list(extra_token_matrix.shape)
    }

    # ✅ INSERT HERE — 3. Norm stats for all tokens
    embedding_norms = embedding_weight.norm(p=2, dim=1).detach().cpu().numpy()
    analysis["embeddings"]["norm_stats"] = {
        "mean_norm": float(np.mean(embedding_norms)),
        "std_norm": float(np.std(embedding_norms)),
        "min_norm": float(np.min(embedding_norms)),
        "max_norm": float(np.max(embedding_norms))
    }
    np.save(os.path.join(OUTPUT_DIR, "embedding_norms.npy"), embedding_norms)

    # ✅ INSERT HERE — 4. Save embedding clone (first 32100 rows)
    embedding_core = embedding_weight[:32100].detach().cpu()
    torch.save(embedding_core, os.path.join(OUTPUT_DIR, "embedding_core_32100.pt"))
    np.save(os.path.join(OUTPUT_DIR, "embedding_core_32100.npy"), embedding_core.numpy())

    # ✅ INSERT HERE — 5. Save <extra_id_*> token mappings explicitly
    extra_ids = [f"<extra_id_{i}>" for i in range(100)]
    extra_token_info = {
        tok: tokenizer.convert_tokens_to_ids(tok)
        for tok in extra_ids if tok in tokenizer.get_vocab()
    }
    analysis["vocabulary"]["extra_token_ids"] = extra_token_info

    # Save raw weights
    save_tensor(embedding_weight, "embedding_matrix")
    save_tensor(lm_head_weight, "lm_head_matrix")

    # Special tokens to analyze
    special_tokens = ["<pad>", "<unk>", "</s>"]
    for token in special_tokens:
        analysis["special_token_embeddings"][token] = inspect_token(tokenizer, embedding_weight, token)

    # Custom hierarchical tokens
    future_tokens = ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]
    for token in future_tokens:
        tokenized = tokenizer.tokenize(token)
        token_ids = tokenizer.convert_tokens_to_ids(tokenized)
        analysis["custom_token_check"][token] = {
            "tokenized": tokenized,
            "token_ids": token_ids,
            "split_behavior": len(tokenized) > 1
        }

    save_json(analysis, "baseline_metrics.json")
    print("✅ Pre-tokenization analysis complete.")

if __name__ == "__main__":
    run_analysis()
