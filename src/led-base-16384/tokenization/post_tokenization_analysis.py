import torch
import torch.nn.functional as F
from transformers import LEDTokenizer, LEDForConditionalGeneration, AutoConfig
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
import time
from pathlib import Path
import pandas as pd
from scipy import stats
import psutil
import os
from tqdm import tqdm
import gc

class TokenizationAnalyzer:
    def __init__(self):
        # Define paths
        self.model_path = "/home/gadde/Thesis/models/pretrained/led-base-16384-updated"
        self.original_embeddings_path = "/home/gadde/Thesis/src/led-base-16384/tokenization/pre_analysis/original_embeddings.pt"
        self.output_dir = Path("/home/gadde/Thesis/src/led-base-16384/tokenization/post_analysis")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        print("Loading model and tokenizer...")
        self.tokenizer = LEDTokenizer.from_pretrained(self.model_path)
        self.model = LEDForConditionalGeneration.from_pretrained(
            self.model_path,
            ignore_mismatched_sizes=True
        )
        
        print("Loading original embeddings...")
        self.original_embeddings = torch.load(self.original_embeddings_path)
        
        # Expected ratios for hierarchical relationships
        self.expected_ratios = {
            'SEC_BOS': 0.95,  # Expected similarity between SEC and BOS
            'SUBSEC_SEC': 0.80,  # 80% similarity with SEC
            'SUBSEC_BOUNDARY': 0.20,  # 20% boundary influence
            'SUBSUBSEC_SUBSEC': 0.80,  # 80% similarity with SUBSEC
            'SUBSUBSEC_BOUNDARY': 0.20  # 20% boundary component
        }
        
        print("Initialization complete.")

    def embedding_preservation_check(self):
        """Compare original vs current embeddings for base vocabulary."""
        print("\nPerforming embedding preservation check...")
        current_embeddings = self.model.get_input_embeddings().weight
        
        # Compare first 50265 embeddings (original vocab size)
        original = self.original_embeddings[:50265]
        current = current_embeddings[:50265]
        
        # Calculate differences
        diff = torch.norm(original - current, dim=1)
        #max_diff = torch.max(diff)
        #avg_diff = torch.mean(diff)
        max_diff = float(torch.max(diff).item())  # Convert to float
        avg_diff = float(torch.mean(diff).item()) # Convert to float
        
        # Statistical tests
        ks_statistic, p_value = stats.ks_2samp(
            original.detach().cpu().flatten().numpy(),  # Add cpu()
            current.detach().cpu().flatten().numpy()    # Add cpu()
        )
        
        # Calculate percentage change
        if torch.all(original == current):
            percent_change = 0.0
        else:
            percent_change = float(torch.mean(torch.abs(original - current) / torch.abs(original)).item()) * 100
        
        results = {
            "max_difference": max_diff,  # Already float, don't call .item()
            "average_difference": avg_diff,  # Already float, don't call .item()
            "percentage_change": percent_change,  # Already float, don't call .item()
            "ks_test_statistic": float(ks_statistic),
            "ks_test_p_value": float(p_value)
        }
        
        return results

    def analyze_padding_impact(self):
        """Analyze impact of padding on embeddings and performance."""
        print("\nAnalyzing padding impact...")
        
        # Get current embeddings
        current_embeddings = self.model.get_input_embeddings().weight
        
        # Calculate padded size
        vocab_size = len(self.tokenizer)
        padded_size = ((vocab_size + 8 - 1) // 8) * 8
        
        results = {
            "current_vocab_size": vocab_size,
            "padded_size": padded_size,
            "padding_difference": padded_size - vocab_size,
        }
        
        # Performance implications
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        results["memory_usage_mb"] = memory_usage
        
        return results

    def analyze_token_usage(self, sample_texts=None):
        """Analyze token usage patterns with sample texts."""
        print("\nAnalyzing token usage patterns...")
        
        if sample_texts is None:
            sample_texts = [
                "[SEC] Introduction\nThis is a test. [SUBSEC] Background\nMore details here. [SUBSUBSEC] Details\nEven more specific.",
                "[SEC] Methods\n[SUBSEC] Data Collection\n[SUBSUBSEC] Survey Design\nThe survey was designed...",
                "[SEC] Results\nMain findings. [SUBSEC] Analysis\nDetailed analysis. [SUBSUBSEC] Statistical Tests\nP-values..."
            ]
        
        usage_stats = {
            "average_sequence_length": 0,
            "special_token_frequency": {
                "SEC": 0,
                "SUBSEC": 0,
                "SUBSUBSEC": 0
            },
            "context_window_usage": []
        }
        
        for text in sample_texts:
            tokens = self.tokenizer.encode(text)
            usage_stats["average_sequence_length"] += len(tokens)
            
            # Count special tokens
            usage_stats["special_token_frequency"]["SEC"] += text.count("[SEC]")
            usage_stats["special_token_frequency"]["SUBSEC"] += text.count("[SUBSEC]")
            usage_stats["special_token_frequency"]["SUBSUBSEC"] += text.count("[SUBSUBSEC]")
            
            # Calculate context window usage
            usage_stats["context_window_usage"].append(len(tokens) / 16384)  # 16384 is max length
        
        usage_stats["average_sequence_length"] /= len(sample_texts)
        
        return usage_stats
    
    def analyze_embedding_space(self):
        """Analyze embedding space characteristics."""
        print("\nAnalyzing embedding space characteristics...")
        embeddings = self.model.get_input_embeddings().weight
        
        # Get special token embeddings
        special_tokens = ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']
        special_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tqdm(special_tokens, desc="Processing special tokens")]
        special_embeddings = embeddings[special_ids]
        
        # Find nearest neighbors
        similarities = F.cosine_similarity(special_embeddings.unsqueeze(1), 
                                         embeddings.unsqueeze(0), dim=2)
        
        # Get top 5 nearest neighbors for each special token
        k = 5
        nearest_neighbors = {}
        for i, token in enumerate(special_tokens):
            _, indices = similarities[i].topk(k + 1)  # +1 to exclude self
            neighbors = [self.tokenizer.decode([idx.item()]) for idx in indices[1:]]  # exclude self
            nearest_neighbors[token] = neighbors
        
        # Perform clustering analysis
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings.detach().numpy())
        
        # Get cluster assignments for special tokens
        special_clusters = {token: cluster_labels[id].item() 
                          for token, id in zip(special_tokens, special_ids)}
        
        return {
            "nearest_neighbors": nearest_neighbors,
            "cluster_assignments": special_clusters
        }

    def verify_model_behavior(self):
        """Verify model behavior with special tokens."""
        print("\nVerifying model behavior...")
        
        test_inputs = [
            "[SEC] Test section\n",
            "[SUBSEC] Test subsection\n",
            "[SUBSUBSEC] Test subsubsection\n"
        ]
    
        results = {}
        for input_text in test_inputs:
           inputs = self.tokenizer(input_text, return_tensors="pt")
        
           # Get model outputs
           with torch.no_grad():
              outputs = self.model(**inputs)
            
           # Instead of attention patterns, let's analyze logits
           logits = outputs.logits
        
           # Get the prediction scores for special tokens
           special_token_ids = [
               self.tokenizer.convert_tokens_to_ids(token)
               for token in ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']
           ]
        
           special_token_scores = logits[0, 0, special_token_ids].detach()
        
           results[input_text] = {
               "special_token_scores": {
                   token: score.item()
                   for token, score in zip(['SEC', 'SUBSEC', 'SUBSUBSEC'], special_token_scores)
               }
           }
        
        return results

    def measure_performance(self):
        """Measure performance metrics."""
        print("\nMeasuring performance metrics...")
        
        results = {}
        
        # Tokenization speed
        start_time = time.time()
        test_text = "[SEC] Test\n" * 1000
        self.tokenizer.encode(test_text)
        tokenization_time = time.time() - start_time
        
        # Memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Inference time
        input_ids = self.tokenizer.encode("Test", return_tensors="pt")
        start_time = time.time()
        with torch.no_grad():
            self.model(input_ids)
        inference_time = time.time() - start_time
        
        results = {
            "tokenization_speed_ms": tokenization_time * 1000,
            "memory_usage_mb": memory_usage,
            "inference_time_ms": inference_time * 1000
        }
        
        return results

    def check_file_consistency(self):
        """Check consistency of model files."""
        print("\nChecking file consistency...")
        
        files_to_check = [
            "config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer_config.json"
        ]
        
        results = {}
        for file in files_to_check:
            file_path = Path(self.model_path) / file
            if file_path.exists():
                if file.endswith('.json'):
                    with open(file_path) as f:
                        content = json.load(f)
                        results[file] = {
                            "exists": True,
                            "size": file_path.stat().st_size,
                            "content_summary": {
                                k: type(v).__name__ for k, v in content.items()
                            }
                        }
                else:
                    results[file] = {
                        "exists": True,
                        "size": file_path.stat().st_size
                    }
            else:
                results[file] = {"exists": False}
        
        return results
    
    def convert_tensors_to_python(self, obj):
        """Convert tensor values to Python native types for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().item() if obj.numel() == 1 else obj.cpu().detach().tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_tensors_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_tensors_to_python(item) for item in obj]
        return obj

    def run_all_analyses(self):
        """Run all analyses and save results."""
        try:
            print("Starting complete analysis...")
            
            # 1. First check model weights and configuration
            results = {
                'model_weights_check': self.check_model_weights(),
                'hierarchical_relationships': self.verify_hierarchical_relationships(),
                'expected_values': self.verify_expected_values(),
                # Add new analyses here
                'end_token_analysis': self.analyze_end_tokens(),
                'random_init_check': self.check_random_initializations(),
                'lm_head_analysis': self.analyze_lm_head()
            }
        
            # 2. Check embedding preservation and structure
            results.update({
                'embedding_preservation': self.embedding_preservation_check(),
                'padding_impact': self.analyze_padding_impact(),
            })
        
            # 3. Analyze usage and behavior
            results.update({
                'token_usage': self.analyze_token_usage(),
                'embedding_space': self.analyze_embedding_space(),
                'model_behavior': self.verify_model_behavior(),
            })
        
            # 4. Performance and consistency checks
            results.update({
                'performance_metrics': self.measure_performance(),
                'file_consistency': self.check_file_consistency()
            })

            # Convert all tensor values to Python native types
            results = self.convert_tensors_to_python(results)
        
            # 5. Save results
            with open(self.output_dir / 'analysis_results.json', 'w') as f:
                json.dump(results, f, indent=4)
        
            # 6. Create visualizations
            self.visualize_embeddings()
            
            return results
        
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None    

    def visualize_embeddings(self):
        """Visualize embedding relationships using dimensionality reduction."""
        try:
            print("\nCreating embedding visualizations...")
            embeddings = self.model.get_input_embeddings().weight
        
            # Get special token indices
            special_indices = [
                self.tokenizer.bos_token_id,
                self.tokenizer.convert_tokens_to_ids('[SEC]'),
                self.tokenizer.convert_tokens_to_ids('[SUBSEC]'),
                self.tokenizer.convert_tokens_to_ids('[SUBSUBSEC]'),
                # Add end tokens
                self.tokenizer.convert_tokens_to_ids('[/SEC]'),
                self.tokenizer.convert_tokens_to_ids('[/SUBSEC]'),
                self.tokenizer.convert_tokens_to_ids('[/SUBSUBSEC]')
            ]
    
            # Get embeddings for visualization
            special_embeddings = embeddings[special_indices].detach().numpy()
    
            # Perform dimensionality reduction
            # Set perplexity to 2 (must be less than n_samples)
            tsne = TSNE(n_components=2, random_state=42, perplexity=2)
            pca = PCA(n_components=2)
        
            # Create visualizations
            tsne_results = tsne.fit_transform(special_embeddings)
            pca_results = pca.fit_transform(special_embeddings)
        
            # Plot results
            self.plot_embedding_projections(tsne_results, "t-SNE")
            self.plot_embedding_projections(pca_results, "PCA")
        
            # Create similarity heatmap
            similarity_matrix = torch.nn.functional.cosine_similarity(
                torch.tensor(special_embeddings).unsqueeze(1),
                torch.tensor(special_embeddings).unsqueeze(0),
                dim=2
            )
    
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                similarity_matrix,
                annot=True,
                fmt='.3f',
                xticklabels=['BOS', 'SEC', 'SUBSEC', 'SUBSUBSEC'],
                yticklabels=['BOS', 'SEC', 'SUBSEC', 'SUBSUBSEC']
            )
            plt.title('Cosine Similarity Between Special Tokens')
            plt.savefig(self.output_dir / 'similarity_heatmap.png')
            plt.close()
        
        except Exception as e:
            print(f"Error during visualization: {str(e)}")

    def plot_embedding_projections(self, data, method):
        """Helper function to plot embedding projections."""
        plt.figure(figsize=(10, 8))
        
        # Plot points
        plt.scatter(data[:, 0], data[:, 1])
        
        # Add labels
        labels = ['BOS', 'SEC', 'SUBSEC', 'SUBSUBSEC']
        for i, label in enumerate(labels):
            plt.annotate(
                label,
                (data[i, 0], data[i, 1]),
                xytext=(5, 5),
                textcoords='offset points'
            )
    
        plt.title(f'{method} Projection of Special Token Embeddings')
        plt.xlabel(f'{method} Component 1')
        plt.ylabel(f'{method} Component 2')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plt.savefig(self.output_dir / f'{method.lower()}_projection.png')
        plt.close()


    def check_model_weights(self):
        """Check weight tying and other model configurations"""
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
            'embeddings_shape': self.model.get_input_embeddings().weight.shape == (vocab_size, d_model),
            'lm_head_shape': self.model.get_output_embeddings().weight.shape == (vocab_size, d_model),
            'encoder_embed_shape': self.model.led.encoder.embed_tokens.weight.shape == (vocab_size, d_model),
            'decoder_embed_shape': self.model.led.decoder.embed_tokens.weight.shape == (vocab_size, d_model)
        }
    
        # 4. Parameter initialization checks
        checks['initialization'] = {
            'embeddings_std': self.model.get_input_embeddings().weight.std().item(),
            'lm_head_std': self.model.get_output_embeddings().weight.std().item()
        }
    
        # 5. Attention mask and position embedding checks
        checks['attention_components'] = {
            'attention_mask_fn': hasattr(self.model, 'get_extended_attention_mask'),
            'pos_embeddings_max_len': self.model.config.max_encoder_position_embeddings
        }
    
        # 6. Device consistency
        device = self.model.device
        checks['device_consistency'] = {
            'embeddings_device': self.model.get_input_embeddings().weight.device == device,
            'lm_head_device': self.model.get_output_embeddings().weight.device == device,
            'encoder_device': self.model.led.encoder.embed_tokens.weight.device == device,
            'decoder_device': self.model.led.decoder.embed_tokens.weight.device == device
        }
    
        return checks
    
    def verify_hierarchical_relationships(self):
        """Verify if embeddings maintain intended hierarchical relationships"""
        embeddings = self.model.get_input_embeddings().weight
        
        # Get embeddings for special tokens
        sec_id = self.tokenizer.convert_tokens_to_ids('[SEC]')
        subsec_id = self.tokenizer.convert_tokens_to_ids('[SUBSEC]')
        subsubsec_id = self.tokenizer.convert_tokens_to_ids('[SUBSUBSEC]')
        
        sec_embed = embeddings[sec_id]
        subsec_embed = embeddings[subsec_id]
        subsubsec_embed = embeddings[subsubsec_id]
        
        # Calculate cosine similarities
        # similarities = {
        #     'sec_subsec': F.cosine_similarity(sec_embed.unsqueeze(0), 
        #                                     subsec_embed.unsqueeze(0)),
        #     'subsec_subsubsec': F.cosine_similarity(subsec_embed.unsqueeze(0), 
        #                                       subsubsec_embed.unsqueeze(0)),
        #     'sec_subsubsec': F.cosine_similarity(sec_embed.unsqueeze(0), 
        #                                    subsubsec_embed.unsqueeze(0))
        # }

        # Calculate cosine similarities and convert to Python floats
        similarities = {
            'sec_subsec': float(F.cosine_similarity(sec_embed.unsqueeze(0), 
                                        subsec_embed.unsqueeze(0)).item()),
            'subsec_subsubsec': float(F.cosine_similarity(subsec_embed.unsqueeze(0), 
                                          subsubsec_embed.unsqueeze(0)).item()),
            'sec_subsubsec': float(F.cosine_similarity(sec_embed.unsqueeze(0), 
                                       subsubsec_embed.unsqueeze(0)).item())
        }
        
        return similarities
    
    def __del__(self):
        """Cleanup method"""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    def verify_expected_values(self):
        """Verify if embeddings match expected initialization values"""
        try:
            current_embeddings = self.model.get_input_embeddings().weight
        
            # Compare with expected ratios from initialization
            actual_ratios = {}
            for key in self.expected_ratios.keys():
                try:
                    actual_ratios[key] = self.calculate_similarity_ratio(current_embeddings, key)
                except Exception as e:
                    print(f"Error calculating ratio for {key}: {str(e)}")
                    actual_ratios[key] = 0.0
        
            return {
                'expected': self.expected_ratios,
                'actual': actual_ratios,
                'differences': {
                    k: abs(self.expected_ratios[k] - actual_ratios[k])
                    for k in self.expected_ratios.keys()
                }
            }
        except Exception as e:
            print(f"Error in verify_expected_values: {str(e)}")
            return {
                'expected': self.expected_ratios,
                'actual': {},
                'differences': {}
            }

    def calculate_similarity_ratio(self, embeddings, relationship_key):
        """Calculate similarity ratio between token pairs based on relationship key."""
        token_pairs = {
            'SEC_BOS': ('[SEC]', self.tokenizer.bos_token),
            'SUBSEC_SEC': ('[SUBSEC]', '[SEC]'),
            'SUBSEC_BOUNDARY': ('[SUBSEC]', self.tokenizer.eos_token),  # Using EOS as boundary
            'SUBSUBSEC_SUBSEC': ('[SUBSUBSEC]', '[SUBSEC]'),
            'SUBSUBSEC_SEC': ('[SUBSUBSEC]', '[SEC]'),
            'SUBSUBSEC_BOUNDARY': ('[SUBSUBSEC]', self.tokenizer.eos_token)
        }
        
        try:
            token1, token2 = token_pairs[relationship_key]
            id1 = self.tokenizer.convert_tokens_to_ids(token1)
            id2 = self.tokenizer.convert_tokens_to_ids(token2)
            
            embed1 = embeddings[id1]
            embed2 = embeddings[id2]
            
            similarity = F.cosine_similarity(
                embed1.unsqueeze(0),
                embed2.unsqueeze(0)
            )
    
            return float(similarity.item())
        except Exception as e:
            print(f"Error calculating similarity for {relationship_key}: {str(e)}")
            return 0.0

    def analyze_end_tokens(self):
        """Analyze end token embeddings and their relationships"""
        end_tokens = ['[/SEC]', '[/SUBSEC]', '[/SUBSUBSEC]']
        start_tokens = ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']
        
        embeddings = self.model.get_input_embeddings().weight
        
        # Get token pairs similarities
        similarities = {}
        for start, end in zip(start_tokens, end_tokens):
            start_id = self.tokenizer.convert_tokens_to_ids(start)
            end_id = self.tokenizer.convert_tokens_to_ids(end)
            sim = F.cosine_similarity(
                embeddings[start_id].unsqueeze(0),
                embeddings[end_id].unsqueeze(0)
            )
            similarities[f"{start}-{end}"] = sim.item()
    
        return similarities
    
    def check_random_initializations(self):
        """Check randomly initialized tokens"""
        embeddings = self.model.get_input_embeddings().weight
        
        # Get end tokens and unused token
        random_tokens = ['[/SEC]', '[/SUBSEC]', '[/SUBSUBSEC]', '[UNUSED1]']
        random_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in random_tokens]
        
        # Analyze their properties
        stats = {}
        for token, idx in zip(random_tokens, random_ids):
            embedding = embeddings[idx]
            stats[token] = {
                'norm': torch.norm(embedding).item(),
                'mean': embedding.mean().item(),
                'std': embedding.std().item()
            }
    
        return stats
    
    def analyze_lm_head(self):
        """Analyze LM head weights for new tokens"""
        input_embeddings = self.model.get_input_embeddings().weight
        output_embeddings = self.model.get_output_embeddings().weight
    
        # Compare input vs output embeddings for new tokens
        new_tokens = ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]', 
                    '[/SEC]', '[/SUBSEC]', '[/SUBSUBSEC]', '[UNUSED1]']
    
        comparison = {}
        for token in new_tokens:
            idx = self.tokenizer.convert_tokens_to_ids(token)
            is_equal = torch.equal(input_embeddings[idx], output_embeddings[idx])
            comparison[token] = {
                'tied': is_equal,
                'input_norm': torch.norm(input_embeddings[idx]).item(),
                'output_norm': torch.norm(output_embeddings[idx]).item()
            }
    
        return comparison

def main():
    analyzer = TokenizationAnalyzer()
    results = analyzer.run_all_analyses()
    
    print("\nAnalysis complete. Results saved to:", analyzer.output_dir)
    print("\nKey findings:")
    print("-" * 50)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()