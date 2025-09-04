from transformers import LEDForConditionalGeneration, LEDTokenizer
import torch
import json
from pathlib import Path
from datetime import datetime 
import os
import torch.nn as nn

class ModelAnalyzer:
    def __init__(self, original_path: str, target_path: str):
        """
        original_path: Path to untouched model
        target_path: Path to model that will be modified
        """
        self.orig_model = LEDForConditionalGeneration.from_pretrained(original_path)
        self.orig_tokenizer = LEDTokenizer.from_pretrained(original_path)
        
        self.target_model = LEDForConditionalGeneration.from_pretrained(target_path)
        self.target_tokenizer = LEDTokenizer.from_pretrained(target_path)
        
    def analyze_weights(self):
        """Compare embeddings between original and target models"""
        print("\n=== Embedding Analysis ===")
        
        # Original model stats
        orig_embeds = self.orig_model.get_input_embeddings().weight
        orig_lm_head = self.orig_model.get_output_embeddings().weight
        
        print(f"Original Model:")
        print(f"Input Embedding Shape: {orig_embeds.shape}")
        print(f"LM Head Shape: {orig_lm_head.shape}")
        
        # Target model stats
        target_embeds = self.target_model.get_input_embeddings().weight
        target_lm_head = self.target_model.get_output_embeddings().weight
        
        print(f"\nTarget Model (Pre-modification):")
        print(f"Input Embedding Shape: {target_embeds.shape}")
        print(f"LM Head Shape: {target_lm_head.shape}")
        
        # Verify weights are identical (they should be before modification)
        weights_identical = torch.equal(orig_embeds, target_embeds)
        print(f"\nWeights Identical: {weights_identical}")
        
        self.analyze_special_tokens()
    
    def analyze_special_tokens(self):
        """Analyze special token embeddings"""
        print("\n=== Special Token Analysis ===")
        special_tokens = {
            'pad': 1,
            'eos': 2,
            'unk': 3,
            'mask': 50264
        }
        
        for name, token_id in special_tokens.items():
            orig_embed = self.orig_model.get_input_embeddings().weight[token_id]
            target_embed = self.target_model.get_input_embeddings().weight[token_id]
            
            print(f"\n{name.upper()} Token (id={token_id}):")
            print(f"Mean: {orig_embed.mean().item():.4f}")
            print(f"Std: {orig_embed.std().item():.4f}")
            print(f"First 5 values: {orig_embed[:5].tolist()}")
            print(f"Matches target: {torch.equal(orig_embed, target_embed)}")
    
    def check_architecture(self):
        """Compare model architectures"""
        print("\n=== Architecture Verification ===")
        
        checks = {
            'vocab_size': self.orig_model.config.vocab_size,
            'hidden_size': self.orig_model.config.d_model,
            'encoder_layers': self.orig_model.config.encoder_layers,
            'decoder_layers': self.orig_model.config.decoder_layers,
            'encoder_attention_heads': self.orig_model.config.encoder_attention_heads,
            'decoder_attention_heads': self.orig_model.config.decoder_attention_heads,
            'encoder_ffn_dim': self.orig_model.config.encoder_ffn_dim,
            'decoder_ffn_dim': self.orig_model.config.decoder_ffn_dim,
            'max_encoder_position_embeddings': self.orig_model.config.max_encoder_position_embeddings,
            'max_decoder_position_embeddings': self.orig_model.config.max_decoder_position_embeddings
        }
        
        for key, value in checks.items():
            target_value = getattr(self.target_model.config, key)
            print(f"{key}:")
            print(f"  Original: {value}")
            print(f"  Target: {target_value}")
            print(f"  Match: {value == target_value}")
    
    def verify_tokenizer_state(self):
        """Compare tokenizer states"""
        print("\n=== Tokenizer Verification ===")
        
        # Compare vocabulary sizes
        print(f"Vocabulary size:")
        print(f"  Original: {len(self.orig_tokenizer)}")
        print(f"  Target: {len(self.target_tokenizer)}")
        
        # Compare special tokens
        print("\nSpecial tokens mapping:")
        for key, value in self.orig_tokenizer.special_tokens_map.items():
            target_value = self.target_tokenizer.special_tokens_map.get(key)
            print(f"\n{key}:")
            print(f"  Original: {value}")
            print(f"  Target: {target_value}")
            print(f"  Match: {value == target_value}")
        
        # Test tokenization of future special tokens
        test_tokens = ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]
        print("\nTest tokenization of new special tokens:")
        for token in test_tokens:
            orig_tokens = self.orig_tokenizer.tokenize(token)
            target_tokens = self.target_tokenizer.tokenize(token)
            print(f"\n{token}:")
            print(f"  Original splits: {orig_tokens}")
            print(f"  Target splits: {target_tokens}")
    
    def check_buffer_tokens(self):
        """Analyze buffer token space and usage"""
        print("\n=== Buffer Token Analysis ===")
        
        # Check embedding matrix size vs vocab size
        embed_size = self.orig_model.get_input_embeddings().weight.shape[0]
        vocab_size = len(self.orig_tokenizer)
        buffer_size = embed_size - vocab_size
        
        print(f"Embedding matrix size: {embed_size}")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Buffer size: {buffer_size}")
        
        # Check last few token IDs
        print("\nLast few token mappings:")
        for i in range(vocab_size-5, vocab_size):
            token = self.orig_tokenizer.convert_ids_to_tokens(i)
            print(f"ID {i}: {token}")
    
    def verify_weight_tying(self):
        """Check embedding weight sharing status"""
        print("\n=== Weight Tying Analysis ===")
        
        # Check if input and output embeddings share memory
        input_embed = self.orig_model.get_input_embeddings()
        output_embed = self.orig_model.get_output_embeddings()
        
        # Check memory location
        same_memory = (input_embed.weight.data_ptr() == output_embed.weight.data_ptr())
        print(f"Input/Output embeddings share memory: {same_memory}")
        
        # Check if weights are equal
        weights_equal = torch.equal(input_embed.weight, output_embed.weight)
        print(f"Input/Output weights are equal: {weights_equal}")
        
        # Check if they're the same object
        same_object = (input_embed is output_embed)
        print(f"Input/Output embeddings are same object: {same_object}")
    
    def analyze_memory_usage(self):
        """Analyze model memory allocation"""
        print("\n=== Memory Usage Analysis ===")
        
        # Calculate parameter counts
        total_params = sum(p.numel() for p in self.orig_model.parameters())
        trainable_params = sum(p.numel() for p in self.orig_model.parameters() if p.requires_grad)
        
        # Calculate embedding memory
        embed_params = self.orig_model.get_input_embeddings().weight.numel()
        embed_memory = embed_params * 4  # 4 bytes per float32
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Embedding parameters: {embed_params:,}")
        print(f"Embedding memory: {embed_memory/1024/1024:.2f} MB")
        
        # Memory per new token
        memory_per_token = self.orig_model.config.d_model * 4  # bytes
        print(f"\nMemory per new token: {memory_per_token/1024:.2f} KB")
    
    def check_attention_config(self):
        """Verify attention settings"""
        print("\n=== Attention Configuration ===")
        
        config = self.orig_model.config
        print("Global Attention Settings:")
        print(f"Attention window sizes: {config.attention_window}")
        print(f"Attention dropout: {config.attention_dropout}")
        
        # LED specific settings
        print("\nLED Specific Settings:")
        print(f"Max encoder length: {config.max_encoder_position_embeddings}")
        print(f"Max decoder length: {config.max_decoder_position_embeddings}")
    
    def verify_generation_config(self):
        """Check generation settings"""
        print("\n=== Generation Configuration ===")
        
        gen_config_path = os.path.join(os.path.dirname(self.orig_model.config._name_or_path), 
                                     "generation_config.json")
        
        if os.path.exists(gen_config_path):
            with open(gen_config_path, 'r') as f:
                gen_config = json.load(f)
                print("Generation Config:")
                for key, value in gen_config.items():
                    print(f"  {key}: {value}")
        else:
            print("No separate generation config found")
            print("\nDefault Generation Settings:")
            print(f"BOS token id: {self.orig_model.config.bos_token_id}")
            print(f"EOS token id: {self.orig_model.config.eos_token_id}")
            print(f"PAD token id: {self.orig_model.config.pad_token_id}")

def main():
    try:
        print("Starting Model Analysis...")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        original_path = "/home/gadde/Thesis/models/pretrained/led-base-16384"
        target_path = "/home/gadde/Thesis/models/pretrained/led-base-16384-updated"
        
        # Verify paths exist
        if not Path(original_path).exists():
            raise FileNotFoundError(f"Original model path not found: {original_path}")
        if not Path(target_path).exists():
            raise FileNotFoundError(f"Target model path not found: {target_path}")
            
        print(f"\nAnalyzing models:")
        print(f"Original: {original_path}")
        print(f"Target: {target_path}")
        
        analyzer = ModelAnalyzer(original_path, target_path)
        
        # Run all analyses including new ones
        print("\nRunning analysis steps:")
        print("1. Analyzing weights and embeddings...")
        analyzer.analyze_weights()
        
        print("\n2. Checking model architecture...")
        analyzer.check_architecture()
        
        print("\n3. Verifying tokenizer state...")
        analyzer.verify_tokenizer_state()
        
        print("\n4. Checking buffer tokens...")
        analyzer.check_buffer_tokens()
        
        print("\n5. Verifying weight tying...")
        analyzer.verify_weight_tying()
        
        print("\n6. Analyzing memory usage...")
        analyzer.analyze_memory_usage()
        
        print("\n7. Checking attention configuration...")
        analyzer.check_attention_config()
        
        print("\n8. Verifying generation configuration...")
        analyzer.verify_generation_config()
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()