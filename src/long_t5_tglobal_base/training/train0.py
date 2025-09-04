# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# import json
# import logging
# import torch
# from tqdm import tqdm
# from transformers import LongT5ForConditionalGeneration, get_linear_schedule_with_warmup
# from torch.nn import CrossEntropyLoss
# from peft import LoraConfig, get_peft_model, PeftModel
# #from src.long_t5_tglobal_base.preprocessing.data_loading import LongT5DataLoader
# from long_t5_tglobal_base.preprocessing.data_loading import LongT5DataLoader
# from datetime import datetime

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
# logger = logging.getLogger("LongT5_Training")

# def setup_experiment_dir(config):
#     """Create timestamped run directory based on config and current time"""
#     timestamp = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
#     run_name = f"{timestamp}_lr{config['training']['learning_rate']}_bs{config['training']['batch_size']}_epochs{config['training']['num_epochs']}_run001"
#     save_dir = os.path.join(config['training']['save_dir'], run_name)
#     os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
#     os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
#     return save_dir

# def set_seed(seed: int = 42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def save_checkpoint(model, optimizer, scheduler, epoch, save_dir):
#     ckpt_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pth")
#     torch.save({
#         "epoch": epoch + 1,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "scheduler_state_dict": scheduler.state_dict()
#     }, ckpt_path)
#     logger.info(f"Checkpoint saved to: {ckpt_path}")

# def train():
#     set_seed()

#     logger.info("✅ Initializing DataLoader and Config")
#     data_loader = LongT5DataLoader()
#     config = data_loader.config

#     train_loader = data_loader.get_dataloader("train")
#     val_loader = data_loader.get_dataloader("val")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LongT5ForConditionalGeneration.from_pretrained(config["model"]["path"]).to(device)

#     if not config.get("debug_mode", True):
#         lora_cfg = LoraConfig(
#             r=config['training']['lora']['r'],
#             lora_alpha=config['training']['lora']['alpha'],
#             target_modules=config['training']['lora']['target_modules'],
#             lora_dropout=config['training']['lora']['dropout'],
#             bias="none",
#             task_type="SEQ_2_SEQ_LM"
#         )
#         model = get_peft_model(model, lora_cfg)
#         logger.info("🔗 LoRA adapters added.")

#     tokenizer = data_loader.tokenizer

#     optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
#     num_training_steps = len(train_loader) * config['training']['num_epochs']
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=config['training']['scheduler'].get("warmup_steps", 0),
#         num_training_steps=num_training_steps
#     )

#     save_dir = setup_experiment_dir(config)

#     for epoch in range(config['training']['num_epochs']):
#         model.train()
#         total_loss = 0
#         progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

#         for batch in progress:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             labels[labels == tokenizer.pad_token_id] = -100  # Ignore pad tokens

#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )

#             loss = outputs.loss
#             loss.backward()

#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()

#             total_loss += loss.item()
#             progress.set_postfix({"loss": loss.item()})

#         logger.info(f"Epoch {epoch+1}: Training Loss = {total_loss / len(train_loader):.4f}")
#         save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(save_dir, "checkpoints"))

#         # model.eval() for val phase can be added here
#         # but for now, basic training is the focus

#     if isinstance(model, PeftModel):
#         logger.info("💾 Saving LoRA model...")
#         adapter_dir = os.path.join(save_dir, "checkpoints")
#         model.save_pretrained(adapter_dir)
#         tokenizer.save_pretrained(adapter_dir)
#         logger.info(f"LoRA adapter and tokenizer saved to: {adapter_dir}")

# if __name__ == "__main__":
#     train()

# # ===============================
# # STRUCTURAL TOKEN LOSS BLOCK (commented for now)
# # 
# # special_tokens = ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']
# # special_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in special_tokens]
# # weight = torch.ones(model.config.vocab_size).to(device)
# # for tok_id in special_token_ids:
# #     weight[tok_id] = 3.0
# # loss_fct = CrossEntropyLoss(weight=weight, ignore_index=-100)
# # logits = outputs.logits.view(-1, model.config.vocab_size)
# # target = labels.view(-1)
# # loss = loss_fct(logits, target)
# # ===============================











# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# import json
# import logging
# import torch
# from tqdm import tqdm
# from transformers import LongT5ForConditionalGeneration, get_linear_schedule_with_warmup
# from torch.nn import CrossEntropyLoss
# from peft import LoraConfig, get_peft_model, PeftModel
# from long_t5_tglobal_base.preprocessing.data_loading import LongT5DataLoader
# from datetime import datetime

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
# logger = logging.getLogger("LongT5_Training")

# def setup_experiment_dir(config):
#     timestamp = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
#     run_name = f"{timestamp}_lr{config['training']['learning_rate']}_bs{config['training']['batch_size']}_epochs{config['training']['num_epochs']}_run001"
#     save_dir = os.path.join(config['training']['save_dir'], run_name)
#     os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
#     os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
#     return save_dir

# def set_seed(seed: int = 42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def save_checkpoint(model, optimizer, scheduler, epoch, save_dir, is_best=False):
#     ckpt_name = "best_model.pth" if is_best else f"epoch_{epoch + 1}.pth"
#     ckpt_path = os.path.join(save_dir, "checkpoints", ckpt_name)
#     torch.save({
#         "epoch": epoch + 1,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "scheduler_state_dict": scheduler.state_dict()
#     }, ckpt_path)
#     logger.info(f"Checkpoint saved to: {ckpt_path}")

# def train():
#     set_seed()

#     logger.info("\u2705 Initializing DataLoader and Config")
#     data_loader = LongT5DataLoader()
#     config = data_loader.config

#     train_loader = data_loader.get_dataloader("train")
#     val_loader = data_loader.get_dataloader("val")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LongT5ForConditionalGeneration.from_pretrained(config["model"]["path"]).to(device)

#     if not config.get("debug_mode", True):
#         lora_cfg = LoraConfig(
#             r=config['training']['lora']['r'],
#             lora_alpha=config['training']['lora']['alpha'],
#             target_modules=config['training']['lora']['target_modules'],
#             lora_dropout=config['training']['lora']['dropout'],
#             bias="none",
#             task_type="SEQ_2_SEQ_LM"
#         )
#         model = get_peft_model(model, lora_cfg)
#         logger.info("\ud83d\udd17 LoRA adapters added.")

#     tokenizer = data_loader.tokenizer

#     optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
#     num_training_steps = len(train_loader) * config['training']['num_epochs']
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=config['training']['scheduler'].get("warmup_steps", 0),
#         num_training_steps=num_training_steps
#     )

#     save_dir = setup_experiment_dir(config)
#     best_val_loss = float("inf")

#     for epoch in range(config['training']['num_epochs']):
#         model.train()
#         total_loss = 0
#         progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

#         for step, batch in enumerate(progress):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             labels[labels == tokenizer.pad_token_id] = -100  # Ignore pad tokens

#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )

#             # ================== Structural Token Loss ==================
#             logits = outputs.logits.view(-1, model.config.vocab_size)
#             target = labels.view(-1)

#             special_tokens = ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']
#             special_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in special_tokens]

#             weight = torch.ones(model.config.vocab_size).to(device)
#             for tok_id in special_token_ids:
#                 weight[tok_id] = 3.0

#             loss_fct = CrossEntropyLoss(weight=weight, ignore_index=-100)
#             loss = loss_fct(logits, target)
#             # ===========================================================

#             # === Normal loss version ===
#             # loss = outputs.loss

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()

#             total_loss += loss.item()
#             progress.set_postfix({"loss": loss.item()})

#             if epoch == 0 and step == 0:
#                 logger.info(f"First batch input shape: {input_ids.shape}, labels shape: {labels.shape}")
#                 logger.info(f"Decoded labels[0]: {tokenizer.decode(labels[0], skip_special_tokens=False)}")

#         avg_train_loss = total_loss / len(train_loader)
#         logger.info(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.4f}")

#         save_checkpoint(model, optimizer, scheduler, epoch, save_dir)

#         # Add val loop later here

#         # Save best model (for now we use train loss as proxy)
#         if avg_train_loss < best_val_loss:
#             best_val_loss = avg_train_loss
#             save_checkpoint(model, optimizer, scheduler, epoch, save_dir, is_best=True)

#     if isinstance(model, PeftModel):
#         logger.info("\ud83d\udcc2 Saving LoRA model and tokenizer...")
#         adapter_dir = os.path.join(save_dir, "checkpoints")
#         model.save_pretrained(adapter_dir)
#         tokenizer.save_pretrained(adapter_dir)
#         logger.info(f"LoRA adapter and tokenizer saved to: {adapter_dir}")

# if __name__ == "__main__":
#     train()

import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import json
import logging
import torch
from tqdm import tqdm
from datetime import datetime
from transformers import LongT5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from peft import LoraConfig, get_peft_model, PeftModel
from long_t5_tglobal_base.preprocessing.data_loading import LongT5DataLoader
from evaluate import load as load_metric
import matplotlib.pyplot as plt
import bert_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("LongT5_Training")

def setup_experiment_dir(config):
    timestamp = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    run_name = f"{timestamp}_lr{config['training']['learning_rate']}_bs{config['training']['batch_size']}_epochs{config['training']['num_epochs']}_run001"
    save_dir = os.path.join(config['training']['save_dir'], run_name)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
    return save_dir

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, scheduler, epoch, save_dir, config, is_best=False):
    ckpt = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }
    # torch.save(ckpt, os.path.join(save_dir, 'checkpoints', f"epoch_{epoch+1}.pth"))
    # if is_best:
    #     torch.save(ckpt, os.path.join(save_dir, 'checkpoints', 'best_model.pth'))
    #     logger.info("💎 Best model checkpoint saved")
    
    # if is_best:
    #     torch.save(ckpt, os.path.join(save_dir, 'checkpoints', 'best_model.pth'))
    #     logger.info("💎 Best model checkpoint saved")

    # # Save periodic checkpoints only on intervals (not best)
    # if not is_best and (epoch + 1) % config['training'].get('save_interval', 1) == 0:
    #     torch.save(ckpt, os.path.join(save_dir, 'checkpoints', f"epoch_{epoch+1}.pth"))
    #     logger.info(f"📝 Checkpoint saved at epoch {epoch+1}")

    if is_best:
        path = os.path.join(save_dir, 'checkpoints', 'best_model.pth')
        torch.save(ckpt, path)
        logger.info("💎 Best model checkpoint saved")
        logger.info(f"Saved checkpoint to {path}")

    if not is_best and (epoch + 1) % config['training'].get('save_interval', 1) == 0:
        path = os.path.join(save_dir, 'checkpoints', f"epoch_{epoch+1}.pth")
        torch.save(ckpt, path)
        logger.info(f"📝 Checkpoint saved at epoch {epoch+1}")
        logger.info(f"Saved checkpoint to {path}")
    

def plot_metrics(train_losses, val_losses, perplexities, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss per Epoch")
    plt.savefig(os.path.join(save_path, "loss_plot.png"))
    plt.close()

    plt.figure()
    plt.plot(perplexities, label="Perplexity")
    plt.title("Perplexity per Epoch")
    plt.savefig(os.path.join(save_path, "perplexity_plot.png"))
    plt.close()

def train():
    set_seed()

    logger.info("✅ Initializing DataLoader and Config")
    #logger.info("🧪 Running in ablation mode: structural tokens will be stripped")
    data_loader = LongT5DataLoader(remove_structural_tokens=True)
    if data_loader.remove_structural_tokens:
        logger.info("🧪 Running in ablation mode: structural tokens will be stripped")
    else:
        logger.info("✅ Structural tokens included in training")

    config = data_loader.config

    train_loader = data_loader.get_dataloader("train")
    val_loader = data_loader.get_dataloader("val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LongT5ForConditionalGeneration.from_pretrained(config["model"]["path"]).to(device)

    # 🔍 Log and optionally override dropout
    logger.info(f"Model dropout: {model.config.dropout_rate}")
    model.config.dropout_rate = 0.0  # set to 0.0 to disable dropout during training (ablation)
    logger.info("🔧 Overriding dropout rate to 0.0 for ablation")

    if not config.get("debug_mode", True):
        lora_cfg = LoraConfig(
            r=config['training']['lora']['r'],
            lora_alpha=config['training']['lora']['alpha'],
            target_modules=config['training']['lora']['target_modules'],
            lora_dropout=config['training']['lora']['dropout'],
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_cfg)
        logger.info("🔗 LoRA adapters added.")
        logger.info(f"LoRA config: r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, dropout={lora_cfg.lora_dropout}")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable:,}/{total:,} ({100 * trainable/total:.2f}%)")

    tokenizer = data_loader.tokenizer
    logger.info(f"Special tokens: {tokenizer.special_tokens_map}")
    logger.info(f"Additional: {tokenizer.additional_special_tokens}")
    logger.info(f"IDs: {tokenizer.convert_tokens_to_ids(['[SEC]', '[SUBSEC]', '[SUBSUBSEC]'])}")
    #sys.exit()
    # ✅ NEW: Confirm special tokens are not in tokenizer if ablation mode
    logger.info(f"Tokenizer size: {len(tokenizer)}")
    logger.info(f"Special tokens in tokenizer vocab: {tokenizer.additional_special_tokens}")
    #assert all(tok not in tokenizer.get_vocab() for tok in ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]), \
    #    "⚠️ Structural tokens still present in tokenizer!"
    assert all(tok not in tokenizer.get_vocab() for tok in ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]), "⚠️ Structural tokens still present in tokenizer!"


    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['scheduler'].get("warmup_steps", 0),
        num_training_steps=num_training_steps
    )

    save_dir = setup_experiment_dir(config)
    # Save config used for this run
    with open(os.path.join(save_dir, "config_used.json"), "w") as f:
        json.dump(config, f, indent=2)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses, perplexities = [], [], []
    rougeL_scores, bleu_scores = [], []
    bert_precisions, bert_recalls, bert_f1s = [], [], []
    rouge_metric = load_metric("rouge")
    bleu_metric = load_metric("bleu")
    
    for epoch in range(config['training']['num_epochs']):
        start_time = time.time()
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for step, batch in enumerate(progress):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)


            # === CLEAN LABELS ===
            labels = labels.clone()

            # Replace padding tokens with -100 (ignore index)
            labels = torch.where(labels == tokenizer.pad_token_id, labels.new_full(labels.shape, -100), labels)

            # Replace any invalid values < -1
            invalid_mask = labels < -1
            if invalid_mask.any():
                logger.warning("⚠️ Found label tokens less than -1 (invalid). Fixing them.")
                labels = torch.where(invalid_mask, labels.new_full(labels.shape, -100), labels)

            # Replace tokens >= vocab size
            if (labels >= model.config.vocab_size).any():
                logger.warning("⚠️ Found label tokens outside vocab size. Fixing them.")
                labels = torch.where(labels >= model.config.vocab_size, labels.new_full(labels.shape, -100), labels)

            invalid_tokens = (labels < -1).sum().item()
            if invalid_tokens > 0:
                logger.warning(f"🧨 Number of label tokens < -1: {invalid_tokens}")

            # ✅ Final catch: Only -100 should remain below 0
            if (labels < -1).any():
                offending = labels[labels < -1]
                if (offending != -100).any():
                    logger.warning(f"⚠️ Found truly invalid tokens < -1 (not -100): {offending}")
                else:
                    logger.info("ℹ️ Labels contain only expected -100 ignore tokens.")

            # (Optional) log sample bad values
            if invalid_mask.any() and epoch == 0 and step == 0:
                bad_vals = labels[invalid_mask]
                logger.warning(f"🔍 Invalid label values: {bad_vals[:10]}")

            # ✅ DEBUG STRUCTURAL TOKENS IN LABELS
            if epoch == 0 and step == 0:
                special_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]]
                logger.info(f"Special Token IDs: {special_token_ids}")
                logger.info(f"Label sample (raw): {labels[0][:50]}")
                for tok_id in special_token_ids:
                    if tok_id in labels[0]:
                        logger.info(f"✅ Found special token ID {tok_id} in first label batch.")
                    else:
                        logger.warning(f"⚠️ Special token ID {tok_id} NOT found in first label batch.")

                # ✅ Decode label to check structural tokens in text
                safe_label_ids = labels[0][labels[0] != -100].tolist()
                decoded = tokenizer.decode(safe_label_ids, skip_special_tokens=True)
                logger.info(f"Decoded label: {decoded[:300]}")




            #labels = torch.where(labels == tokenizer.pad_token_id, torch.tensor(-100).to(labels), labels)
            #labels = torch.where(labels == tokenizer.pad_token_id, labels.new_tensor(-100), labels)

            # Remove any invalid tokens above vocab_size
            #labels = torch.where(labels >= tokenizer.vocab_size, torch.tensor(-100).to(labels), labels)
            #labels = torch.where(labels >= tokenizer.vocab_size, labels.new_tensor(-100), labels)
            
            # Optional: Remove anything less than -1
            #labels = torch.where(labels < -1, torch.tensor(-100).to(labels), labels)
            #labels = torch.where(labels < -1, labels.new_tensor(-100), labels) 
            
            
            # ✅ ADD THIS SAFETY CHECK HERE
            #assert labels.shape[1] == model.config.max_length, f"Label length mismatch: got {labels.shape[1]} vs expected {model.config.max_length}"
            #assert labels.shape[1] <= model.config.max_length, f"Label too long: {labels.shape[1]} > {model.config.max_length}"

            expected_label_len = data_loader.max_output_length
            assert labels.shape[1] == expected_label_len, f"Label length mismatch: got {labels.shape[1]} vs expected {expected_label_len}"

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            logits = outputs.logits.view(-1, model.config.vocab_size)
            target = labels.view(-1)
            # weight = torch.ones(model.config.vocab_size).to(device)
            # for token in ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]:
            #     tok_id = tokenizer.convert_tokens_to_ids(token)
            #     weight[tok_id] = 1.5
            # loss_fct = CrossEntropyLoss(weight=weight, ignore_index=-100)

            # Conditional weighting only if structural tokens are used
            if not data_loader.remove_structural_tokens:
                weight = torch.ones(model.config.vocab_size).to(device)
                for token in ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]:
                    tok_id = tokenizer.convert_tokens_to_ids(token)
                    weight[tok_id] = 1.5
                loss_fct = CrossEntropyLoss(weight=weight, ignore_index=-100)
                logger.info("🎯 Using structural token-weighted loss.")
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                logger.info("🔬 Using unweighted CrossEntropyLoss (ablation mode).")

            loss = loss_fct(logits, target)

            if epoch == 0 and step == 0:
                unweighted_loss = outputs.loss.item()
                logger.info(f"Unweighted loss: {unweighted_loss:.4f}, Structural loss: {loss.item():.4f}")

            if torch.isnan(loss):
                logger.warning("⚠️ Loss is NaN. Skipping step.")
                continue

            # # --- Normal loss version ---
            # loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

            # if epoch == 0 and step == 0:
            #     logger.info(f"🔍 First batch decoded:")
            #     label_ids = labels[0]
            #     label_ids = label_ids[label_ids != -100]  # filter out ignored label positions
            #     #logger.info(tokenizer.decode(label_ids, skip_special_tokens=False))

            #     # === B: Print shapes and label sample ===
            #     logger.info(f"input_ids shape: {input_ids.shape}")
            #     logger.info(f"labels shape: {labels.shape}")
            #     logger.info(f"Labels sample: {labels[0][:50]}")  # First 50 tokens
            #     logger.info(f"Pad token ID: {tokenizer.pad_token_id}")
    
            #     # === C: Check for invalid token IDs ===
            #     if (labels > tokenizer.vocab_size).any():
            #         logger.warning("⚠️ Found label tokens outside vocab size")
            #     if (labels < -1).any():
            #         logger.warning("⚠️ Found label tokens less than -1 (invalid)")

            #     # === Optional: Decode safely to avoid overflow ===
            #     try:
            #         decoded = tokenizer.decode(labels[0].tolist(), skip_special_tokens=False)
            #         logger.info(decoded)
            #     except OverflowError as e:
            #         logger.warning(f"Decode failed due to overflow: {e}")

            if epoch == 0 and step == 0:
                logger.info(f"🔍 First batch decoded:")

                # === B: Print shapes and label sample ===
                logger.info(f"input_ids shape: {input_ids.shape}")
                logger.info(f"labels shape: {labels.shape}")
                logger.info(f"Labels sample: {labels[0][:50]}")  # First 50 tokens
                logger.info(f"Pad token ID: {tokenizer.pad_token_id}")
    
                # === C: Check for invalid token IDs ===
                if (labels > tokenizer.vocab_size).any():
                    logger.warning("⚠️ Found label tokens outside vocab size")
                if (labels < -1).any():
                    logger.warning("⚠️ Found label tokens less than -1 (invalid)")
                    logger.warning(f"⚠️ Offending label values: {labels[labels < -1][:10].tolist()}")

                # === Optional: Decode safely to avoid overflow ===
                try:
                    #label_ids = labels[0]
                    #label_ids = label_ids[label_ids != -100]  # filter out ignored label positions
                    #label_ids = label_ids[label_ids != -100]  # filter out -100s
                    label_ids = labels[0][labels[0] != -100]  # filter out ignore index
                    decoded = tokenizer.decode(label_ids.tolist(), skip_special_tokens=True)
                    #logger.info(decoded)
                    logger.info(f"Decoded label: {decoded[:300]}")
                except OverflowError as e:
                    #logger.warning(f"Decode failed due to overflow: {e}")
                    logger.warning(f"⚠️ Safe decode failed: {e}")

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.4f}")

        logger.info(f"🕒 Epoch {epoch+1} time: {(time.time() - start_time) / 60:.2f} min")

        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        predictions, references = [], []
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                labels = val_batch['labels'].to(device)
                labels[labels == tokenizer.pad_token_id] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                logits = outputs.logits.view(-1, model.config.vocab_size)
                target = labels.view(-1)
                loss = loss_fct(logits, target)
                val_loss += loss.item()

                generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=labels.shape[1])
                decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                #decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode([label[label != -100] for label in labels], skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend([[ref] for ref in decoded_labels])

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logger.info(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

        # === EARLY STOPPING CHECK WITH MIN_DELTA ===
        if config["training"]["early_stopping"]["enabled"]:
            improvement = best_val_loss - avg_val_loss
            if improvement > config["training"]["early_stopping"]["min_delta"]:
                is_best = True
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                is_best = False
                epochs_no_improve += 1
                if epochs_no_improve >= config["training"]["early_stopping"]["patience"]:
                    logger.info(f"⏹️ Early stopping triggered. No significant improvement in {epochs_no_improve} epochs.")
                    break
        else:
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss



        #save_checkpoint(model, optimizer, scheduler, epoch, save_dir, is_best=is_best)
        if (epoch + 1) % config['training'].get('save_interval', 1) == 0 or is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, save_dir, config, is_best=is_best)



        safe_loss = min(avg_val_loss, 20)
        perplexity = torch.exp(torch.tensor(safe_loss)).item()
        perplexities.append(perplexity)
        logger.info(f"Perplexity = {perplexity:.4f}")

        #rouge_score = rouge_metric.compute(predictions=predictions, references=[ref[0] for ref in references])
        # rouge_score = rouge_score['rougeL'] if isinstance(rouge_score['rougeL'], float) else rouge_score['rougeL'].mid.fmeasure
        # bleu_score = bleu_metric.compute(predictions=predictions, references=references)
        # logger.info(f"ROUGE-L: {rouge_score['rougeL'].mid.fmeasure:.4f}, BLEU: {bleu_score['bleu']:.4f}")

        # metrics = {
        #     "train_losses": train_losses,
        #     "val_losses": val_losses,
        #     "perplexities": perplexities,
        #     "rougeL": rouge_score['rougeL'].mid.fmeasure,
        #     "bleu": bleu_score['bleu'],
        #     "current_epoch": epoch
        # }

        # rougeL = rouge_score['rougeL'] if isinstance(rouge_score['rougeL'], float) else rouge_score['rougeL'].mid.fmeasure
        # bleu_score = bleu_metric.compute(predictions=predictions, references=references)
        # logger.info(f"ROUGE-L: {rougeL:.4f}, BLEU: {bleu_score['bleu']:.4f}")

        rouge_result = rouge_metric.compute(predictions=predictions, references=[ref[0] for ref in references])
        bleu_result = bleu_metric.compute(predictions=predictions, references=references)

        rougeL = rouge_result['rougeL'].mid.fmeasure if hasattr(rouge_result['rougeL'], 'mid') else rouge_result['rougeL']
        bleu = bleu_result['bleu']


        # ✅ Add BERTScore computation here
        P, R, F1 = bert_score.score(predictions, [ref[0] for ref in references], lang="en", verbose=False)
        bert_f1 = F1.mean().item()
        bert_precision = P.mean().item()
        bert_recall = R.mean().item()

        logger.info(f"BERTScore - Precision: {bert_precision:.4f}, Recall: {bert_recall:.4f}, F1: {bert_f1:.4f}")

        rougeL_scores.append(rougeL)
        bleu_scores.append(bleu)
        bert_precisions.append(bert_precision)
        bert_recalls.append(bert_recall)
        bert_f1s.append(bert_f1)

        # metrics = {
        #     "train_losses": train_losses,
        #     "val_losses": val_losses,
        #     "perplexities": perplexities,
        #     "rougeL": rougeL,
        #     "bleu": bleu,
        #     "bert_precision": bert_precision,
        #     "bert_recall": bert_recall,
        #     "bert_f1": bert_f1,
        #     "current_epoch": epoch
        # }

        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "perplexities": perplexities,
            "rougeL_scores": rougeL_scores,
            "bleu_scores": bleu_scores,
            "bert_precisions": bert_precisions,
            "bert_recalls": bert_recalls,
            "bert_f1s": bert_f1s,
            "current_epoch": epoch
        }


        json.dump(metrics, open(os.path.join(save_dir, "results", "metrics.json"), "w"), indent=2)
        torch.save(metrics, os.path.join(save_dir, "results", "metrics.pt"))
        plot_metrics(train_losses, val_losses, perplexities, os.path.join(save_dir, "results"))

    if isinstance(model, PeftModel):
        logger.info("💾 Saving LoRA model...")
        adapter_dir = os.path.join(save_dir, "checkpoints")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        logger.info(f"LoRA adapter and tokenizer saved to: {adapter_dir}")

if __name__ == "__main__":
    train()