# import time
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# import json
# import logging
# import torch
# from tqdm import tqdm
# from datetime import datetime
# from transformers import LongT5ForConditionalGeneration, get_linear_schedule_with_warmup
# from torch.nn import CrossEntropyLoss
# from peft import LoraConfig, get_peft_model, PeftModel
# from long_t5_tglobal_base.preprocessing.data_loading import LongT5DataLoader
# from evaluate import load as load_metric
# import matplotlib.pyplot as plt

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
#     ckpt = {
#         "epoch": epoch + 1,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "scheduler_state_dict": scheduler.state_dict()
#     }
#     torch.save(ckpt, os.path.join(save_dir, 'checkpoints', f"epoch_{epoch+1}.pth"))
#     if is_best:
#         torch.save(ckpt, os.path.join(save_dir, 'checkpoints', 'best_model.pth'))
#         logger.info("💎 Best model checkpoint saved")

# def plot_metrics(train_losses, val_losses, perplexities, save_path):
#     plt.figure()
#     plt.plot(train_losses, label="Train Loss")
#     plt.plot(val_losses, label="Val Loss")
#     plt.legend()
#     plt.title("Loss per Epoch")
#     plt.savefig(os.path.join(save_path, "loss_plot.png"))
#     plt.close()

#     plt.figure()
#     plt.plot(perplexities, label="Perplexity")
#     plt.title("Perplexity per Epoch")
#     plt.savefig(os.path.join(save_path, "perplexity_plot.png"))
#     plt.close()

# def train():
#     set_seed()

#     logger.info("✅ Initializing DataLoader and Config")
#     #logger.info("🧪 Running in ablation mode: structural tokens will be stripped")
#     data_loader = LongT5DataLoader(remove_structural_tokens=False)
#     if data_loader.remove_structural_tokens:
#         logger.info("🧪 Running in ablation mode: structural tokens will be stripped")
#     else:
#         logger.info("✅ Structural tokens included in training")

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
#     best_val_loss = float("inf")
#     train_losses, val_losses, perplexities = [], [], []
#     rouge_metric = load_metric("rouge")
#     bleu_metric = load_metric("bleu")

    
#     for epoch in range(config['training']['num_epochs']):
#         start_time = time.time()
#         model.train()
#         total_loss = 0
#         progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

#         for step, batch in enumerate(progress):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             # ✅ DEBUG STRUCTURAL TOKENS IN LABELS
#             if epoch == 0 and step == 0:
#                 special_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]]
#                 logger.info(f"Special Token IDs: {special_token_ids}")
#                 logger.info(f"Label sample (raw): {labels[0][:50]}")
#                 for tok_id in special_token_ids:
#                     if tok_id in labels[0]:
#                         logger.info(f"✅ Found special token ID {tok_id} in first label batch.")
#                     else:
#                         logger.warning(f"⚠️ Special token ID {tok_id} NOT found in first label batch.")

#                 # ✅ Decode label to check structural tokens in text
#                 decoded = tokenizer.decode(labels[0].tolist(), skip_special_tokens=False)
#                 logger.info(f"Decoded label: {decoded[:300]}")




#             #labels = torch.where(labels == tokenizer.pad_token_id, torch.tensor(-100).to(labels), labels)
#             labels = torch.where(labels == tokenizer.pad_token_id, labels.new_tensor(-100), labels)

#             # Remove any invalid tokens above vocab_size
#             #labels = torch.where(labels >= tokenizer.vocab_size, torch.tensor(-100).to(labels), labels)
#             #labels = torch.where(labels >= tokenizer.vocab_size, labels.new_tensor(-100), labels)
            
#             # Optional: Remove anything less than -1
#             #labels = torch.where(labels < -1, torch.tensor(-100).to(labels), labels)
#             #labels = torch.where(labels < -1, labels.new_tensor(-100), labels) 


#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )

#             logits = outputs.logits.view(-1, model.config.vocab_size)
#             target = labels.view(-1)
#             weight = torch.ones(model.config.vocab_size).to(device)
#             for token in ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]:
#                 tok_id = tokenizer.convert_tokens_to_ids(token)
#                 weight[tok_id] = 1.5
#             loss_fct = CrossEntropyLoss(weight=weight, ignore_index=-100)
#             loss = loss_fct(logits, target)

#             if epoch == 0 and step == 0:
#                 unweighted_loss = outputs.loss.item()
#                 logger.info(f"Unweighted loss: {unweighted_loss:.4f}, Structural loss: {loss.item():.4f}")

#             if torch.isnan(loss):
#                 logger.warning("⚠️ Loss is NaN. Skipping step.")
#                 continue

#             # # --- Normal loss version ---
#             # loss = outputs.loss

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()

#             total_loss += loss.item()
#             progress.set_postfix({"loss": loss.item()})

#             # if epoch == 0 and step == 0:
#             #     logger.info(f"🔍 First batch decoded:")
#             #     label_ids = labels[0]
#             #     label_ids = label_ids[label_ids != -100]  # filter out ignored label positions
#             #     #logger.info(tokenizer.decode(label_ids, skip_special_tokens=False))

#             #     # === B: Print shapes and label sample ===
#             #     logger.info(f"input_ids shape: {input_ids.shape}")
#             #     logger.info(f"labels shape: {labels.shape}")
#             #     logger.info(f"Labels sample: {labels[0][:50]}")  # First 50 tokens
#             #     logger.info(f"Pad token ID: {tokenizer.pad_token_id}")
    
#             #     # === C: Check for invalid token IDs ===
#             #     if (labels > tokenizer.vocab_size).any():
#             #         logger.warning("⚠️ Found label tokens outside vocab size")
#             #     if (labels < -1).any():
#             #         logger.warning("⚠️ Found label tokens less than -1 (invalid)")

#             #     # === Optional: Decode safely to avoid overflow ===
#             #     try:
#             #         decoded = tokenizer.decode(labels[0].tolist(), skip_special_tokens=False)
#             #         logger.info(decoded)
#             #     except OverflowError as e:
#             #         logger.warning(f"Decode failed due to overflow: {e}")

#             if epoch == 0 and step == 0:
#                 logger.info(f"🔍 First batch decoded:")

#                 # === B: Print shapes and label sample ===
#                 logger.info(f"input_ids shape: {input_ids.shape}")
#                 logger.info(f"labels shape: {labels.shape}")
#                 logger.info(f"Labels sample: {labels[0][:50]}")  # First 50 tokens
#                 logger.info(f"Pad token ID: {tokenizer.pad_token_id}")
    
#                 # === C: Check for invalid token IDs ===
#                 if (labels > tokenizer.vocab_size).any():
#                     logger.warning("⚠️ Found label tokens outside vocab size")
#                 if (labels < -1).any():
#                     logger.warning("⚠️ Found label tokens less than -1 (invalid)")

#                 # === Optional: Decode safely to avoid overflow ===
#                 try:
#                     label_ids = labels[0]
#                     label_ids = label_ids[label_ids != -100]  # filter out ignored label positions
#                     decoded = tokenizer.decode(label_ids.tolist(), skip_special_tokens=False)
#                     logger.info(decoded)
#                 except OverflowError as e:
#                     logger.warning(f"Decode failed due to overflow: {e}")

#         avg_train_loss = total_loss / len(train_loader)
#         train_losses.append(avg_train_loss)
#         logger.info(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.4f}")

#         logger.info(f"🕒 Epoch {epoch+1} time: {(time.time() - start_time) / 60:.2f} min")

#         # --- Validation Loop ---
#         model.eval()
#         val_loss = 0
#         predictions, references = [], []
#         with torch.no_grad():
#             for val_batch in tqdm(val_loader, desc="Validation"):
#                 input_ids = val_batch['input_ids'].to(device)
#                 attention_mask = val_batch['attention_mask'].to(device)
#                 labels = val_batch['labels'].to(device)
#                 labels[labels == tokenizer.pad_token_id] = -100

#                 outputs = model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     labels=labels
#                 )

#                 logits = outputs.logits.view(-1, model.config.vocab_size)
#                 target = labels.view(-1)
#                 loss = loss_fct(logits, target)
#                 val_loss += loss.item()

#                 generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=labels.shape[1])
#                 decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
#                 #decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#                 decoded_labels = tokenizer.batch_decode([label[label != -100] for label in labels], skip_special_tokens=False)
#                 predictions.extend(decoded_preds)
#                 references.extend([[ref] for ref in decoded_labels])

#         avg_val_loss = val_loss / len(val_loader)
#         val_losses.append(avg_val_loss)
#         logger.info(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

#         is_best = avg_val_loss < best_val_loss
#         if is_best:
#             best_val_loss = avg_val_loss

#         save_checkpoint(model, optimizer, scheduler, epoch, save_dir, is_best=is_best)

#         safe_loss = min(avg_val_loss, 20)
#         perplexity = torch.exp(torch.tensor(safe_loss)).item()
#         perplexities.append(perplexity)
#         logger.info(f"Perplexity = {perplexity:.4f}")

#         #rouge_score = rouge_metric.compute(predictions=predictions, references=[ref[0] for ref in references])
#         # rouge_score = rouge_score['rougeL'] if isinstance(rouge_score['rougeL'], float) else rouge_score['rougeL'].mid.fmeasure
#         # bleu_score = bleu_metric.compute(predictions=predictions, references=references)
#         # logger.info(f"ROUGE-L: {rouge_score['rougeL'].mid.fmeasure:.4f}, BLEU: {bleu_score['bleu']:.4f}")

#         # metrics = {
#         #     "train_losses": train_losses,
#         #     "val_losses": val_losses,
#         #     "perplexities": perplexities,
#         #     "rougeL": rouge_score['rougeL'].mid.fmeasure,
#         #     "bleu": bleu_score['bleu'],
#         #     "current_epoch": epoch
#         # }

#         # rougeL = rouge_score['rougeL'] if isinstance(rouge_score['rougeL'], float) else rouge_score['rougeL'].mid.fmeasure
#         # bleu_score = bleu_metric.compute(predictions=predictions, references=references)
#         # logger.info(f"ROUGE-L: {rougeL:.4f}, BLEU: {bleu_score['bleu']:.4f}")

#         rouge_result = rouge_metric.compute(predictions=predictions, references=[ref[0] for ref in references])
#         bleu_result = bleu_metric.compute(predictions=predictions, references=references)

#         rougeL = rouge_result['rougeL'].mid.fmeasure if hasattr(rouge_result['rougeL'], 'mid') else rouge_result['rougeL']
#         bleu = bleu_result['bleu']

#         logger.info(f"ROUGE-L: {rougeL:.4f}, BLEU: {bleu:.4f}")

#         metrics = {
#             "train_losses": train_losses,
#             "val_losses": val_losses,
#             "perplexities": perplexities,
#             "rougeL": rougeL,
#             "bleu": bleu,
#             "current_epoch": epoch
#         }

#         json.dump(metrics, open(os.path.join(save_dir, "results", "metrics.json"), "w"), indent=2)
#         torch.save(metrics, os.path.join(save_dir, "results", "metrics.pt"))
#         plot_metrics(train_losses, val_losses, perplexities, os.path.join(save_dir, "results"))

#     if isinstance(model, PeftModel):
#         logger.info("💾 Saving LoRA model...")
#         adapter_dir = os.path.join(save_dir, "checkpoints")
#         model.save_pretrained(adapter_dir)
#         tokenizer.save_pretrained(adapter_dir)
#         logger.info(f"LoRA adapter and tokenizer saved to: {adapter_dir}")

# if __name__ == "__main__":
#     train()

















