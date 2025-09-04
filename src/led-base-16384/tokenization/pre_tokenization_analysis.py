import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration
import os
import json
from datetime import datetime
import torch.nn.functional as F
import time
import psutil
from sklearn.cluster import KMeans
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # For CPU, Disable GPU

class PreTokenAnalyzer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = LEDForConditionalGeneration.from_pretrained(
            model_path, 
            low_cpu_mem_usage=True,
            device_map="cpu"  # For CPU
        )
        self.model.to("cpu")  # For CPU
        self.tokenizer = LEDTokenizer.from_pretrained(model_path)
        
        # Create output directory in src/led-base-16384/tokenization
        self.output_dir = "/home/gadde/Thesis/src/led-base-16384/tokenization/pre_analysis"
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_vocabulary_state(self):
        """Analyze current vocabulary state"""
        vocab_info = {
            'vocab_size': len(self.tokenizer),
            'special_tokens': self.tokenizer.special_tokens_map,
            'last_token_id': max(self.tokenizer.get_vocab().values()),
            'buffer_tokens': self.check_buffer_tokens()
        }
        return vocab_info

    def check_buffer_tokens(self):
        """Check for buffer tokens in vocabulary"""
        vocab = self.tokenizer.get_vocab()
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        return embedding_size - len(vocab)

    def analyze_embeddings(self):
        """Analyze embedding configurations"""
        input_embeds = self.model.get_input_embeddings()
        output_embeds = self.model.get_output_embeddings()
        
        return {
            'input_shape': list(input_embeds.weight.shape),  # Convert to list for JSON serialization
            'output_shape': list(output_embeds.weight.shape),
            'weight_stats': self.get_embedding_stats(input_embeds.weight),
            'special_token_embeddings': self.get_special_token_embeddings()
        }

    def get_embedding_stats(self, weight_matrix):
        """Calculate embedding statistics"""
        return {
            'mean': float(weight_matrix.mean().item()),
            'std': float(weight_matrix.std().item()),
            'min': float(weight_matrix.min().item()),
            'max': float(weight_matrix.max().item())
        }

    def get_special_token_embeddings(self):
        """Get embeddings for special tokens"""
        special_tokens = {
            'pad': self.tokenizer.pad_token_id,
            'eos': self.tokenizer.eos_token_id,
            'unk': self.tokenizer.unk_token_id,
            'mask': self.tokenizer.mask_token_id
        }
        
        embeddings = self.model.get_input_embeddings().weight
        stats = {}
        
        for name, token_id in special_tokens.items():
            if token_id is not None:
                token_embedding = embeddings[token_id]
                stats[name] = {
                    'id': token_id,
                    'mean': float(token_embedding.mean().item()),
                    'std': float(token_embedding.std().item()),
                    'first_5_values': token_embedding[:5].tolist()
                }
        return stats

    def analyze_architecture(self):
        """Get model architecture details"""
        return {
            'hidden_size': self.model.config.hidden_size,
            'num_layers': self.model.config.encoder_layers,
            'attention_heads': self.model.config.encoder_attention_heads,
            'max_position_embeddings': self.model.config.max_encoder_position_embeddings
        }

    def check_weight_tying(self):
        """Verify weight tying status"""
        input_embeds = self.model.get_input_embeddings()
        output_embeds = self.model.get_output_embeddings()
        
        return {
            'share_memory': input_embeds.weight.data_ptr() == output_embeds.weight.data_ptr(),
            'weights_equal': torch.equal(input_embeds.weight, output_embeds.weight),
            'same_object': input_embeds is output_embeds
        }

    def analyze_memory(self):
        """Analyze memory usage"""
        param_size = sum(p.numel() for p in self.model.parameters())
        embedding_size = self.model.get_input_embeddings().weight.numel()
        
        return {
            'total_parameters': param_size,
            'embedding_parameters': embedding_size,
            'embedding_memory_mb': embedding_size * 4 / (1024 * 1024)  # 4 bytes per float
        }

    def test_future_tokens(self):
        """Test how future special tokens will be handled"""
        test_tokens = ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]
        results = {}
        for token in test_tokens:
            results[token] = {
                'current_tokenization': self.tokenizer.tokenize(token),
                'current_ids': self.tokenizer.encode(token, add_special_tokens=False),
                'split_behavior': len(self.tokenizer.tokenize(token)) > 1  # Check if token gets split
            }
        return results

    def analyze_madeupword_tokens(self):
        """Analyze madeupword token positions"""
        madeupword_ids = {
            'madeupword0000': 50261,
            'madeupword0001': 50262,
            'madeupword0002': 50263
        }
        results = {}
        for token, id in madeupword_ids.items():
            results[token] = {
                'id': id,
                'current_token': self.tokenizer.convert_ids_to_tokens(id),
                'embedding_stats': self.get_embedding_stats(
                    self.model.get_input_embeddings().weight[id]
                )
            }
        return results

    def check_model_config(self):
        """Verify LED-specific configurations"""
        return {
            'attention_window': self.model.config.attention_window,
            'max_encoder_length': self.model.config.max_encoder_position_embeddings,
            'generation_config': self.model.generation_config is not None,
            'attention_dropout': self.model.config.attention_dropout,
            #'hidden_dropout': self.model.config.hidden_dropout
            # Fix: Use correct dropout parameter names for LED
            'dropout': self.model.config.dropout,  # Instead of hidden_dropout
            'activation_dropout': self.model.config.activation_dropout
        }
    
    def analyze_special_token_similarities(self):
        """Analyze similarities between special tokens"""
        special_tokens = {
            'bos': self.tokenizer.bos_token_id,
            'eos': self.tokenizer.eos_token_id,
            'pad': self.tokenizer.pad_token_id
        }
    
        embeddings = self.model.get_input_embeddings().weight
        similarities = {}
    
        for name1, id1 in special_tokens.items():
            for name2, id2 in special_tokens.items():
                if id1 is not None and id2 is not None and name1 < name2:
                    sim = torch.nn.functional.cosine_similarity(
                        embeddings[id1].unsqueeze(0),
                        embeddings[id2].unsqueeze(0)
                    )
                    similarities[f"{name1}_{name2}"] = float(sim.item())
        
        return similarities
    
    def analyze_embedding_space(self):
        """Analyze embedding space characteristics"""
        embeddings = self.model.get_input_embeddings().weight
        
        return {
            'embedding_norm': float(torch.norm(embeddings, dim=1).mean().item()),
            'embedding_variance': float(torch.var(embeddings, dim=1).mean().item()),
            'embedding_sparsity': float((embeddings == 0).float().mean().item())
        }
    
    def verify_tokenizer_model_alignment(self):
        """Verify tokenizer and model are properly aligned"""
        return {
            'vocab_size_match': len(self.tokenizer) == self.model.config.vocab_size,
            'embedding_size_match': self.model.get_input_embeddings().weight.shape[0] == len(self.tokenizer),
            'special_tokens_in_config': all(
            token in self.tokenizer.get_vocab() 
                for token in self.tokenizer.special_tokens_map.values() 
                if isinstance(token, str)
            )
        }
    
    def measure_performance(self):
        """Measure baseline performance metrics"""  
        results = {}
        
        # Tokenization speed
        test_text = "This is a test document." * 1000
        start_time = time.time()
        self.tokenizer.encode(test_text)
        results['tokenization_speed_ms'] = (time.time() - start_time) * 1000
    
        # Memory usage
        results['memory_usage_mb'] = psutil.Process().memory_info().rss / 1024 / 1024
    
        # Inference time
        input_ids = self.tokenizer.encode("Test", return_tensors="pt")
        start_time = time.time()
        with torch.no_grad():
            self.model(input_ids)
        results['inference_time_ms'] = (time.time() - start_time) * 1000
    
        return results
    
    def analyze_token_relationships(self):
        """Analyze relationships between existing tokens"""
        embeddings = self.model.get_input_embeddings().weight
        special_tokens = {
            'bos': self.tokenizer.bos_token_id,
            'eos': self.tokenizer.eos_token_id,
            'pad': self.tokenizer.pad_token_id,
            'unk': self.tokenizer.unk_token_id
        }
    
        relationships = {
            'similarities': {},
            'distances': {},
            'clustering': self.analyze_token_clusters()
        }
    
        # Calculate similarities between special tokens
        for name1, id1 in special_tokens.items():
            for name2, id2 in special_tokens.items():
                if id1 is not None and id2 is not None and name1 < name2:
                    sim = torch.nn.functional.cosine_similarity(
                        embeddings[id1].unsqueeze(0),
                        embeddings[id2].unsqueeze(0)
                    )
                    relationships['similarities'][f"{name1}_{name2}"] = float(sim.item())
    
        return relationships
    
    def analyze_token_clusters(self):
        """Analyze token clustering in embedding space"""
        from sklearn.cluster import KMeans
        
        embeddings = self.model.get_input_embeddings().weight.detach().numpy()
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Get cluster assignments for special tokens
        special_tokens = {
            'bos': self.tokenizer.bos_token_id,
            'eos': self.tokenizer.eos_token_id,
            'pad': self.tokenizer.pad_token_id,
            'unk': self.tokenizer.unk_token_id
        }
    
        cluster_info = {
            'n_clusters': n_clusters,
            'special_token_clusters': {
                name: int(cluster_labels[token_id])
            for name, token_id in special_tokens.items()
            if token_id is not None
        }
    }
    
        return cluster_info

    def save_baseline_metrics(self):
        """Save all analysis results"""
        metrics = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_path': self.model_path,
            'vocabulary': self.analyze_vocabulary_state(),
            'embeddings': self.analyze_embeddings(),
            'embedding_space': self.analyze_embedding_space(),
            'special_token_similarities': self.analyze_special_token_similarities(),
            'architecture': self.analyze_architecture(),
            'weight_tying': self.check_weight_tying(),
            'memory': self.analyze_memory(),
            'future_tokens': self.test_future_tokens(),
            'madeupword_analysis': self.analyze_madeupword_tokens(),
            'model_config': self.check_model_config(),
            'model_weights_and_tying': self.check_model_weights_and_tying(),  # Add this
            'performance': self.measure_performance(),
            'token_relationships': self.analyze_token_relationships(),
            'model_state': {
                'training_mode': self.model.training,
                'requires_grad': {
                    'embeddings': self.model.get_input_embeddings().weight.requires_grad,
                    'lm_head': self.model.get_output_embeddings().weight.requires_grad
                }
            }
        }
        
        # Save metrics
        metrics_path = os.path.join(self.output_dir, 'baseline_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Backup embeddings
        self.save_embeddings_backup()
        
        return metrics

    def save_embeddings_backup(self):
        """Save original embeddings"""
        embeddings = self.model.get_input_embeddings().weight.detach().clone()
        torch.save(embeddings, os.path.join(self.output_dir, 'original_embeddings.pt'))

    def check_model_weights_and_tying(self):
        """Comprehensive check of model weights and tying status"""
        checks = {}
        
        # 1. Basic weight tying checks
        checks['weight_tying'] = {
            'embeddings_lm_head': torch.equal(
                self.model.get_input_embeddings().weight,
                self.model.get_output_embeddings().weight
            ),
           'encoder_decoder_shared': torch.equal(
            self.model.led.encoder.embed_tokens.weight,
                self.model.led.decoder.embed_tokens.weight
            )
        }
    
        # 2. Gradient and training state checks
        checks['training_state'] = {
            'embeddings_grad': self.model.get_input_embeddings().weight.requires_grad,
            'lm_head_grad': self.model.get_output_embeddings().weight.requires_grad,
            'training_mode': self.model.training
        }

        # 3. Shape consistency checks
        vocab_size = self.model.config.vocab_size
        d_model = self.model.config.d_model
        checks['shape_consistency'] = {
            'embeddings_shape': list(self.model.get_input_embeddings().weight.shape),
            'lm_head_shape': list(self.model.get_output_embeddings().weight.shape),
            'encoder_embed_shape': list(self.model.led.encoder.embed_tokens.weight.shape),
            'decoder_embed_shape': list(self.model.led.decoder.embed_tokens.weight.shape)
        }

        # 4. Parameter initialization checks
        checks['initialization'] = {
            'embeddings_std': float(self.model.get_input_embeddings().weight.std().item()),
            'lm_head_std': float(self.model.get_output_embeddings().weight.std().item())
        }

        # 5. Device consistency
        device = self.model.device
        checks['device_consistency'] = {
            'embeddings_device': str(self.model.get_input_embeddings().weight.device),
            'lm_head_device': str(self.model.get_output_embeddings().weight.device),
            'encoder_device': str(self.model.led.encoder.embed_tokens.weight.device),
            'decoder_device': str(self.model.led.decoder.embed_tokens.weight.device)
        }

        return checks

    def __del__(self):
        """Cleanup method"""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    

def main():
    # Model path - where we load from
    model_path = "/home/gadde/Thesis/models/pretrained/led-base-16384-updated"
    
    print("Starting Pre-Token Analysis...")
    print(f"Model path: {model_path}")
    print(f"Output directory: /home/gadde/Thesis/src/led-base-16384/tokenization/pre_analysis")
    
    analyzer = PreTokenAnalyzer(model_path)
    
    try:
        print("\nRunning analysis steps:")
        print("1. Analyzing vocabulary and embeddings...")
        print("2. Checking model architecture...")
        print("3. Testing future token handling...")
        print("4. Analyzing madeupword tokens...")
        print("5. Verifying model configuration...")
        print("6. Saving model state...")
        print("7. Backing up embeddings...")
        
        metrics = analyzer.save_baseline_metrics()
        
        print("\nAnalysis complete! Results saved in:", analyzer.output_dir)
        print("\nSaved files:")
        print("- baseline_metrics.json")
        print("- original_embeddings.pt")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()