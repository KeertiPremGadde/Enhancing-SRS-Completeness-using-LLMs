import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import re
import json
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple
import sys
sys.path.append('/home/gadde/Thesis/src/led-base-16384')
from transformers import LEDForConditionalGeneration, get_linear_schedule_with_warmup
from training.seeding import set_deterministic_seed, seed_worker, get_torch_generator
# Call this FIRST
set_deterministic_seed(42)
from peft import LoraConfig, get_peft_model
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
import bert_score
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import random
import time
from torch.utils.data import DataLoader
from preprocessing.data_loading_no_special_tokens import LEDDataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger("LED_Training")

def log_tensor_info(tensor_dict, name=""):
    """Log detailed tensor information"""
    logger.info(f"\n=== {name} Tensor Information ===")
    for key, tensor in tensor_dict.items():
        logger.info(f"{key}:")
        logger.info(f"  Shape: {tensor.shape}")
        logger.info(f"  Device: {tensor.device}")
        logger.info(f"  Dtype: {tensor.dtype}")
        logger.info(f"  Min/Max: {tensor.min():.2f}/{tensor.max():.2f}")

def setup_experiment_dir(config):
    """Create experiment directory with timestamp"""
    timestamp = time.strftime("%Y_%m_%d-%H:%M:%S")
    save_dir = os.path.join(
        config['training']['save_dir'],
        f"{timestamp}_lr{config['training']['learning_rate']}_bs{config['training']['batch_size']}_epochs{config['training']['num_epochs']}_run001"
    )
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
    return save_dir

#def save_metrics(train_losses, val_losses, perplexities, save_dir, epoch,avg_rouge=None, bleu_score=None, bert_f1=None, config=None):
def save_metrics(train_losses, val_losses, perplexities, rouge_scores, bleu_scores, bert_f1_scores, save_dir, epoch, config=None):
    """Save training metrics and plots"""
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'perplexities': perplexities,
        'rougeL': rouge_scores,
        'bleu4': bleu_scores,
        'bert_f1': bert_f1_scores,
        'current_epoch': epoch
    }
    metrics_path = os.path.join(save_dir, 'results', 'metrics.pt')
    torch.save(metrics, metrics_path)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    #plt.plot(train_losses, label='Training Loss')
    #plt.plot(val_losses, label='Validation Loss')
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'results', 'loss_curves.png'))
    plt.close()
    
    # Plot perplexity
    plt.figure(figsize=(10, 6))
    #plt.plot(perplexities, label='Validation Perplexity')
    plt.plot(range(1, len(perplexities)+1), perplexities, label='Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'results', 'perplexity.png'))
    plt.close()

    # Save JSON version for readability/plotting
    json_metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'perplexities': perplexities,
        'rougeL': rouge_scores,
        'bleu4': bleu_scores,
        'bert_f1': bert_f1_scores,
        'current_epoch': epoch
    }

    #Why this code is commented
    # if avg_rouge is not None and bleu_score is not None and bert_f1 is not None:
    #     json_metrics.update({
    #         'rougeL': avg_rouge,
    #         'bleu4': bleu_score,
    #         'bert_f1': bert_f1
    #     })

    #Add config
    if config is not None:
        json_metrics['config'] = config

    # Save updated metrics
    json_path = os.path.join(save_dir, 'results', 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)



def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    return torch.exp(torch.tensor(loss))

def log_gpu_memory():
    """Log GPU memory usage"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            cached = torch.cuda.memory_reserved() / 1024**2
            logger.info(f"GPU Memory: {allocated:.2f}MB allocated, {cached:.2f}MB cached")
    except Exception as e:
        logger.warning(f"Failed to log GPU memory: {e}")

# # At top of your script, is this required now?
# def set_seed(seed: int = 42):

#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if multi-GPU

#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def inspect_lora_dropout(model):
    logger.info("🔍 Inspecting LoRA modules for dropout...")
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            logger.info(f"  {name} → LoRA dropout: {module.lora_dropout}")





def train(model, train_loader, val_loader, optimizer, scheduler, config, device, tokenizer):
    """Complete training loop"""
    logger.info("Starting training...")
    #set_seed(config.get("seed", 42))  # 👈 seed call early
    if torch.cuda.is_available():
        log_gpu_memory()
        torch.cuda.empty_cache()

    # Get max lengths from model config at the start
    max_decoder_length = model.config.max_decoder_position_embeddings  # 1024
    logger.info(f"\n=== Model Configuration ===")
    logger.info(f"Max decoder length: {max_decoder_length}")

    save_dir = setup_experiment_dir(config)
    num_epochs = config['training']['num_epochs']
    save_interval = config['training'].get('save_interval', 100)
    best_loss = float('inf')
    
    train_losses = []
    val_losses = []
    perplexities = []
    rougeL_scores = []
    bleu4_scores = []
    bert_f1_scores = []




    #early Stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stopping_cfg = config['training'].get('early_stopping', {})

    
    # Initial dimension check with first batch
    logger.info("\n=== Checking Initial Data Dimensions ===")
    for batch in train_loader:
        # Truncate tensors for initial check
        input_ids = batch['input_ids'][:, :max_decoder_length]
        attention_mask = batch['attention_mask'][:, :max_decoder_length]
        global_attention_mask = batch['global_attention_mask'][:, :max_decoder_length]
        labels = batch['labels'][:, :max_decoder_length]
        
        logger.info(f"Initial batch dimensions after truncation:")
        logger.info(f"Input IDs: {input_ids.shape}")
        logger.info(f"Attention Mask: {attention_mask.shape}")
        logger.info(f"Global Attention: {global_attention_mask.shape}")
        logger.info(f"Labels: {labels.shape}")

        assert input_ids.size(1) <= model.config.max_encoder_position_embeddings, \
            f"Input sequence length ({input_ids.size(1)}) exceeds model limit ({model.config.max_encoder_position_embeddings})"

        assert labels.size(1) <= model.config.max_decoder_position_embeddings, \
            f"Label sequence length ({labels.size(1)}) exceeds model limit ({model.config.max_decoder_position_embeddings})"

        batch_size = input_ids.size(0)
        if batch_size != config['training']['batch_size']:
            logger.warning(f"Actual batch size ({batch_size}) differs from configured batch size ({config['training']['batch_size']})")
        
        # Verify shapes
        assert len(input_ids.shape) == 2, f"Unexpected input_ids shape: {input_ids.shape}"
        assert len(attention_mask.shape) == 2, f"Unexpected attention_mask shape: {attention_mask.shape}"
        assert len(global_attention_mask.shape) == 2, f"Unexpected global_attention_mask shape: {global_attention_mask.shape}"
        assert len(labels.shape) == 2, f"Unexpected labels shape: {labels.shape}"
        break
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
         # Confirm LoRA modules are active
        if isinstance(model, torch.nn.Module) and hasattr(model, "peft_config"):
            logger.info("✅ Training with LoRA-enabled model.")
            inspect_lora_dropout(model)
        else:
            logger.info("⚠️ LoRA not active — training full model.")
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True, dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Truncate all tensors consistently and move to device
                input_ids = batch['input_ids'][:, :max_decoder_length].to(device)
                attention_mask = batch['attention_mask'][:, :max_decoder_length].to(device)
                global_attention_mask = batch['global_attention_mask'][:, :max_decoder_length].to(device)
                labels = batch['labels'][:, :max_decoder_length].to(device)
                print("🔍 Decoded labels[0]:\n", tokenizer.decode(labels[0], skip_special_tokens=False))
                #exit()


                if epoch == 0 and batch_idx % 200 == 0:
                    label_ids = labels[0]
                    logger.info(f"🧠 Label Tokens: {tokenizer.convert_ids_to_tokens(label_ids.tolist())}")
                    logger.info(f"🧠 Label Text: {tokenizer.decode(label_ids, skip_special_tokens=False)}")

                # Debug first batch
                if epoch == 0 and batch_idx == 0:
                    logger.info("\n=== First Training Batch Analysis ===")
                    logger.info(f"Input IDs shape: {input_ids.shape}")
                    logger.info(f"Attention mask shape: {attention_mask.shape}")
                    logger.info(f"Global attention shape: {global_attention_mask.shape}")
                    logger.info(f"Labels shape: {labels.shape}")
                    
                    # Verify dimensions match
                    # ✅ Encoder inputs must align
                    assert input_ids.size(1) == attention_mask.size(1) == global_attention_mask.size(1), \
                        "Encoder input dimensions must match"

                    # ✅ Decoder labels should match model's max decoder length
                    assert labels.size(1) == model.config.max_decoder_position_embeddings, \
                        f"Labels length must match decoder max length: expected {model.config.max_decoder_position_embeddings}, got {labels.size(1)}"
                    
                    logger.debug(f"✓ Input shape: {input_ids.shape}, Labels shape: {labels.shape}")

                #step1:
                optimizer.zero_grad()
                
                # outputs = model(
                #     input_ids=input_ids,
                #     attention_mask=attention_mask,
                #     global_attention_mask=global_attention_mask,
                #     labels=labels
                # )

                #step2:
                labels = labels.clone()  # avoid modifying original labels
                labels[labels == tokenizer.pad_token_id] = -100                
                #outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                #step3:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    labels=labels # Will not be used for loss now, just output.logits
                )
                #loss = outputs.loss
                
                # Check output shapes on first batch
                #step4: Debugging shapes (can be inside or outside this block, doesn't affect logic)
                # Check output shapes on first batch
                if epoch == 0 and batch_idx == 0:
                    logger.info("\n=== Model Output Shapes ===")
                    if hasattr(outputs, 'logits'):
                        logger.info(f"Logits shape: {outputs.logits.shape}")
                    logger.info(f"Loss shape: {outputs.loss.shape}")
                    assert outputs.loss.dim() == 0, f"Loss has extra dimensions: {outputs.loss.shape}"
                
                #Step5: Loss calculation or weighted loss
                loss = outputs.loss
                # Weighted loss for structural markers
                #logits = outputs.logits  # [B, T, V]
                #vocab_size = model.config.vocab_size
                #logits = logits.view(-1, vocab_size)       # [B*T, V]
                #target = labels.view(-1)                   # [B*T]

                #special_tokens = ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']
                #special_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in special_tokens]

                #weight = torch.ones(vocab_size).to(device)
                #for tok_id in special_token_ids:
                #    weight[tok_id] = 3.0  # 3x weight for structure tokens

                #loss_fct = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=tokenizer.pad_token_id)
                #loss = loss_fct(logits, target)
                
                #Step : Backprop
                epoch_loss += loss.item()
                
                loss.backward()
                grad_norm = log_gradient_norms(model)
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"Gradient norm after first backward pass: {grad_norm:.4f}")
                
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['optimizer']['gradient_clip']
                )
                optimizer.step()
                scheduler.step()
                
                if batch_idx % 100 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    #logger.info(f"Current learning rate: {current_lr:.2e}")
                    tqdm.write(f"Current learning rate: {current_lr:.2e}")
                # Update progress bar
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({'loss': loss.item()})
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GPU OOM in batch {batch_idx}. Trying to recover...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    logger.error(f"Input shapes - ids: {input_ids.shape}, "
                                f"attention: {attention_mask.shape}, "
                                f"global: {global_attention_mask.shape}")
                    raise e
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_progress = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}', leave=False, dynamic_ncols=True)
        # Track structure token metrics across epoch
        structure_tokens = ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]
        token_match_stats = {tok: {"pred": 0, "label": 0, "match": 0} for tok in structure_tokens}
        predictions = []
        references = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_progress):
                # Truncate validation tensors consistently
                input_ids = batch['input_ids'][:, :max_decoder_length].to(device)
                attention_mask = batch['attention_mask'][:, :max_decoder_length].to(device)
                global_attention_mask = batch['global_attention_mask'][:, :max_decoder_length].to(device)
                labels = batch['labels'][:, :max_decoder_length].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()
                
                # Decode predictions and references for metric eval
                preds = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    max_length=max_decoder_length,
                    num_beams=4,
                    early_stopping=True
                )

                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

                if batch_idx % 200 == 0:
                    logger.info(f"🧾 Raw token IDs: {preds[0]}")
                    logger.info(f"🧾 Decoded (no skip): {tokenizer.decode(preds[0], skip_special_tokens=False)}")

                    #Comment out for non structural tokens run
                    structure_tokens = ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]
                    for tok in structure_tokens:
                        pred_count = decoded_preds[0].count(tok)
                        label_count = decoded_labels[0].count(tok)
                        match_count = min(pred_count, label_count)
                        logger.info(f"📐 {tok} — Predicted: {pred_count}, Target: {label_count}, Matched approx: {match_count}")

                
                
                # Accumulate structure token stats
                for tok in structure_tokens:
                    pred_count = decoded_preds[0].count(tok)
                    label_count = decoded_labels[0].count(tok)
                    match_count = min(pred_count, label_count)

                    token_match_stats[tok]["pred"] += pred_count
                    token_match_stats[tok]["label"] += label_count
                    token_match_stats[tok]["match"] += match_count        

                predictions.extend(decoded_preds)
                references.extend(decoded_labels)

                # 👇 This will print every 200 batches during the first epoch only.
                if epoch == 0 and batch_idx % 200 == 0:
                    logger.info(f"🔁 [Predicted]:\n{decoded_preds[0][:300]}")
                    logger.info(f"✅ [Reference]:\n{decoded_labels[0][:300]}")

                if batch_idx % 100 == 0:
                    val_progress.set_postfix({'val_loss': outputs.loss.item()})
            
            # After the validation loop — log final stats
            logger.info("📊 Structure Token Match Summary (Validation):")
            for tok in structure_tokens:
                pred = token_match_stats[tok]["pred"]
                label = token_match_stats[tok]["label"]
                match = token_match_stats[tok]["match"]
                match_rate = match / label if label > 0 else 0.0
                
                logger.info(f"  {tok} → Predicted: {pred}, Target: {label}, Matched: {match}, Match Rate: {match_rate:.2%}")


        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        perplexity = calculate_perplexity(avg_val_loss)
        perplexities.append(perplexity.item())
        
        # Logging
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Training Loss: {avg_epoch_loss:.4f}')
        logger.info(f'Validation Loss: {avg_val_loss:.4f}')
        logger.info(f'Perplexity: {perplexity:.4f}')

        # === Early stopping old logic ===
        # if config['training'].get('early_stopping', {}).get('enabled', False):
        #     if avg_val_loss < best_val_loss - config['training']['early_stopping']['min_delta']:
        #         best_val_loss = avg_val_loss
        #         early_stop_counter = 0
        #     else:
        #         early_stop_counter += 1
        #         logger.info(f"⚠️ Early stopping patience: {early_stop_counter} / {config['training']['early_stopping']['patience']}")
        #         if early_stop_counter >= config['training']['early_stopping']['patience']:
        #             logger.info("🛑 Early stopping triggered.")
        #             break

        # === Early stopping logic ===
        if early_stopping_cfg.get('enabled', False):
            min_delta = early_stopping_cfg.get('min_delta', 0.0005)
            patience = early_stopping_cfg.get('patience', 3)

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            logger.info(f"✅ Validation loss improved to {avg_val_loss:.6f}")
            # Optional: Save best model checkpoint here
            torch.save(model.state_dict(), os.path.join(config['training']['save_dir'], "best_model.pth"))
        else:
            early_stop_counter += 1
            logger.info(f"⚠️ Early stopping patience: {early_stop_counter} / {patience}")
            if early_stop_counter >= patience:
                logger.info("🛑 Early stopping triggered. Stopping training.")
                break




        # ROUGE-L, BLEU-4, BERTScore evaluation
        if len(predictions) > 0:
            rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_scores = [rouge.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)]
            avg_rouge = sum(rouge_scores) / len(rouge_scores)

            bleu = corpus_bleu(predictions, [references])  # references must be list of lists
            P, R, F1 = bert_score.score(predictions, references, lang="en", verbose=False)

            logger.info(f"[Eval Metrics] ROUGE-L: {avg_rouge:.4f}")
            logger.info(f"[Eval Metrics] BLEU-4: {bleu.score:.2f}")
            logger.info(f"[Eval Metrics] BERTScore-F1: {F1.mean().item():.4f}")

            # save_metrics(
            #     train_losses, val_losses, perplexities, save_dir, epoch,
            #     avg_rouge=avg_rouge,
            #     bleu_score=bleu.score,
            #     bert_f1=F1.mean().item(),
            #     config=config
            # )

            rougeL_scores.append(avg_rouge)
            bleu4_scores.append(bleu.score)
            bert_f1_scores.append(F1.mean().item())

            save_metrics(
                train_losses, val_losses, perplexities,
                rougeL_scores, bleu4_scores, bert_f1_scores,
                save_dir, epoch, config=config
            )

            logger.info(f"📊 Metrics (ROUGE, BLEU, BERTScore) saved to: {os.path.join(save_dir, 'results', 'metrics.json')}")

        
        # Save checkpoints
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_epoch_loss,
                'val_loss': avg_val_loss,
                'perplexity': perplexity.item(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'perplexities': perplexities,
                'rougeL': avg_rouge,
                'bleu4': bleu.score,
                'bert_f1': F1.mean().item()
            }
            torch.save(
                checkpoint,
                os.path.join(save_dir, 'checkpoints', f'epoch_{epoch+1}.pth')
            )


        # ✅ Save LoRA adapter (if LoRA is used)
        if isinstance(model, PeftModel):
            logger.info("💾 Saving LoRA adapter model...")
            model.save_pretrained(os.path.join(save_dir, 'checkpoints'))

            logger.info("💾 Saving tokenizer files...")
            tokenizer.save_pretrained(os.path.join(save_dir, 'checkpoints'))
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_epoch_loss,
                'val_loss': avg_val_loss,
                'perplexity': perplexity.item(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'perplexities': perplexities,
                'rougeL': avg_rouge,
                'bleu4': bleu.score,
                'bert_f1': F1.mean().item()
            }
            torch.save(
                checkpoint,
                os.path.join(save_dir, 'checkpoints', 'best_model.pth')
            )
            logger.info('Saved best model checkpoint')
        
        # Save metrics and plots
        #save_metrics(train_losses, val_losses, perplexities, save_dir, epoch)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ✅ Restore best model if early stopping was used
    if config['training'].get('early_stopping', {}).get('enabled', False):
        best_model_path = os.path.join(config['training']['save_dir'], "best_model.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            logger.info("✅ Restored best model after early stopping.")
    
    return train_losses, val_losses, perplexities, rougeL_scores, bleu4_scores, bert_f1_scores, save_dir

def print_trainable_parameters(model, config):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(f"Model name: {config['model']['name']}")
    logger.info(f"Model path: {config['model']['path']}")
    logger.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )

def log_gradient_norms(model):
    """Log the norm of gradients"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def validate_config(config):
    """Validate essential config parameters"""
    required_fields = [
        'model.path',
        'training.num_epochs',
        'training.batch_size',
        'training.learning_rate',
        'training.lora.r',
        'training.lora.alpha'
    ]
    
    for field in required_fields:
        keys = field.split('.')
        current = config
        for key in keys:
            if key not in current:
                raise ValueError(f"Missing required config field: {field}")
            current = current[key]

# 1. Direct from model config
def check_model_lengths(model):
    logger.info("\n=== Model Position Embedding Limits ===")
    logger.info(f"Max encoder position embeddings: {model.config.max_encoder_position_embeddings}")
    logger.info(f"Max decoder position embeddings: {model.config.max_decoder_position_embeddings}")
    
    # Check actual embedding sizes
    encoder_pos_embed = model.led.encoder.embed_positions
    decoder_pos_embed = model.led.decoder.embed_positions
    
    logger.info("\n=== Actual Embedding Sizes ===")
    logger.info(f"Encoder position embedding size: {encoder_pos_embed.weight.shape}")
    logger.info(f"Decoder position embedding size: {decoder_pos_embed.weight.shape}")

# 2. Check model configuration details
def print_model_config(model):
    logger.info("\n=== Full Model Configuration ===")
    for key, value in model.config.__dict__.items():
        if 'position' in key.lower() or 'length' in key.lower():
            logger.info(f"{key}: {value}")

def main():
    try:
        #set_seed(42)  # 👈 Right here, only once for the whole script
        # Initialize data loader and get config
        data_loader = LEDDataLoader()
        config = data_loader.config
        validate_config(config)

        # ✅ Debug: Inspect a sample flattened label before training
        train_map = json.load(open(config['data']['train_mapping']))
        sample_path = train_map[0]['output']
        sample_label_flat = LEDDataLoader.flatten_srs_to_text(json.load(open(sample_path)))
        logger.info("🧪 Sample training label (flattened):\n" + sample_label_flat[:1500])
               
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model and move to device
        model = LEDForConditionalGeneration.from_pretrained(config['model']['path'])
        model.to(device)

        # Disable dropout for overfit-debug mode
        # if config.get("debug_mode", False):
        #     logger.info("⏩ Disabling dropout layers for overfit-debug mode...")
        #     def disable_dropout(module):
        #         if isinstance(module, torch.nn.Dropout):
        #             module.p = 0.0
        #     model.apply(disable_dropout)

        tokenizer = data_loader.tokenizer

        # Add the new checks here
        check_model_lengths(model)
        print_model_config(model)

        # Print detailed model configuration
        logger.info("\n=== Model Configuration ===")
        logger.info(f"Number of attention heads: {model.config.num_attention_heads}")
        logger.info(f"Hidden size: {model.config.hidden_size}")
        logger.info(f"Vocab size: {model.config.vocab_size}")
        # Fix for LED-specific attributes
        logger.info(f"Max encoder position embeddings: {model.config.max_encoder_position_embeddings}")
        logger.info(f"Max decoder position embeddings: {model.config.max_decoder_position_embeddings}")
        logger.info(f"Encoder layers: {model.config.encoder_layers}")
        logger.info(f"Decoder layers: {model.config.decoder_layers}")
        logger.info(f"Attention window size: {model.config.attention_window}")
        # Optional: Add more LED-specific config logging if needed
        if hasattr(model.config, 'gradient_checkpointing'):
            logger.info(f"Gradient checkpointing: {model.config.gradient_checkpointing}")
        if hasattr(model.config, 'attention_dropout'):
            logger.info(f"Attention dropout: {model.config.attention_dropout}")
        if hasattr(model.config, 'dropout'):
            logger.info(f"Hidden dropout: {model.config.dropout}")
        # Print initial model parameters
        logger.info("=== Model Parameters (Before LoRA) ===")
        print_trainable_parameters(model, config)
        
        if not config.get("debug_mode", False):
        # Setup LoRA
            lora_config = LoraConfig(
                r=config['training']['lora']['r'],
                lora_alpha=config['training']['lora']['alpha'],
                target_modules=config['training']['lora']['target_modules'],
                lora_dropout=config['training']['lora']['dropout'],
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            model = get_peft_model(model, lora_config)
            logger.info("✓ LoRA adapters applied.")
        else:
            logger.info("⏩ Skipping LoRA adapters (debug mode active)")
        
        # Print parameters after LoRA
        logger.info("=== Model Parameters (After LoRA) ===")
        print_trainable_parameters(model, config)
        
        # Get dataloaders
        logger.info("Creating DataLoaders...")
        train_loader = data_loader.get_dataloader("train")
        val_loader = data_loader.get_dataloader("val")
        
        # Calculate training steps
        num_training_steps = len(train_loader) * config['training']['num_epochs']
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['training']['learning_rate'], 
            weight_decay=config['training']['optimizer']['weight_decay']
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['training']['scheduler']['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        # Train model
        train_losses, val_losses, perplexities, rougeL_scores, bleu4_scores, bert_f1_scores, save_dir  = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            tokenizer=tokenizer
        )
        
        # Save only if LoRA is used
        if isinstance(model, PeftModel):
            model.save_pretrained(os.path.join(save_dir, "checkpoints"))
            tokenizer.save_pretrained(os.path.join(save_dir, "checkpoints"))  # optional but helpful
            logger.info(f"✓ LoRA adapter and tokenizer saved at: {save_dir}/checkpoints")

        logger.info(f"📉 Final Training Losses: {train_losses}")
        logger.info(f"📊 Final Validation Losses: {val_losses}")
        logger.info(f"📈 Final Perplexities: {perplexities}")
        logger.info(f"📉 Final ROUGE Scores: {rougeL_scores}")
        logger.info(f"📊 Final BLEU Scores: {bleu4_scores}")
        logger.info(f"📈 Final BERT F1 Scores: {bert_f1_scores}")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Program failed: {e}")
        raise

if __name__ == "__main__":
    main()