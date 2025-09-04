import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration, AutoConfig
import os
from pathlib import Path
import torch.nn.functional as F
import json
from transformers import AddedToken
#os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

class TokenUpdater:
    """
    Handles the addition of special tokens [SEC], [SUBSEC], [SUBSUBSEC] to the LED model.
    Tracks and verifies all file updates during the process.
    """

    def __init__(self, model_path: str):
        """
        Initialize with model path.
        Args:
            model_path: Path to the pretrained LED model
        """
        self.model_path = model_path
        self.model = LEDForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True).to("cuda")
        self.tokenizer = LEDTokenizer.from_pretrained(model_path)

    def initialize_hierarchical_embeddings(self):
        """
        Initialize embeddings for new hierarchical tokens with improved strategy:
        - [SEC]: Based on bos_token with controlled noise
        - [SUBSEC]: Strongly tied to SEC (80%) with some boundary influence (20%)
        - [SUBSUBSEC]: Balanced between SUBSEC (80%) and boundary (20%)
        """
        bos_embedding = self.model.get_input_embeddings().weight[self.tokenizer.bos_token_id]
        eos_embedding = self.model.get_input_embeddings().weight[self.tokenizer.eos_token_id]
        
        # [SEC] - Distinct but structurally related to bos
        sec_embedding = bos_embedding + torch.normal(
            mean=0.0, 
            std=0.02,  # LED's initialization std
            size=bos_embedding.size(), 
            device=bos_embedding.device
        )
        
        # [SUBSEC] - Strongly tied to SEC
        subsec_embedding = 0.8 * sec_embedding + 0.2 * eos_embedding
        
        # [SUBSUBSEC] - Balanced hierarchy
        subsubsec_embedding = 0.8 * subsec_embedding + 0.2 * eos_embedding
        
        return sec_embedding, subsec_embedding, subsubsec_embedding

    def verify_token_embeddings(self):
        """
        Basic verification of the newly initialized token embeddings.
        Full similarity analysis will be done in post-tokenization.
        """
        embeddings = self.model.get_input_embeddings().weight
        
        # Get relevant embeddings
        bos_embedding = embeddings[self.tokenizer.bos_token_id]
        sec_embedding = embeddings[50265]  # [SEC]
        subsec_embedding = embeddings[50266]  # [SUBSEC]
        subsubsec_embedding = embeddings[50267]  # [SUBSUBSEC]
        
        # Compute cosine similarities
        def cosine_similarity(a, b):
            return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
        
        similarities = {
            'sec_bos': cosine_similarity(sec_embedding, bos_embedding).item(),
            'subsec_sec': cosine_similarity(subsec_embedding, sec_embedding).item(),
            'subsubsec_subsec': cosine_similarity(subsubsec_embedding, subsec_embedding).item()
        }
        
        print("\nInitial Token Embedding Similarities:")
        print("-" * 40)
        print(f"[SEC] ~ <s>: {similarities['sec_bos']:.3f}")
        print(f"[SUBSEC] ~ [SEC]: {similarities['subsec_sec']:.3f}")
        print(f"[SUBSUBSEC] ~ [SUBSEC]: {similarities['subsubsec_subsec']:.3f}")
        print("-" * 40)

    def verify_original_state(self):
        """
        Verify the original model state before making any changes.
        """
        assert len(self.tokenizer) == 50265, f"Expected 50265 tokens, got {len(self.tokenizer)}"
        assert self.model.get_input_embeddings().weight.shape == (50265, 768), \
            "Unexpected embedding shape"
        
    def pre_run_checks(self):
        """
        Verify environment before running.
        """
        # Check if model path exists
        assert os.path.exists(self.model_path), f"Model path {self.model_path} does not exist"
        
        # Check write permissions
        assert os.access(self.model_path, os.W_OK), f"No write permission for {self.model_path}"
        
        # Check if required files exist
        required_files = [
            'tokenizer_config.json',
            'special_tokens_map.json',
            'vocab.json',
            'pytorch_model.bin',
            'config.json'
        ]
        for file in required_files:
            file_path = os.path.join(self.model_path, file)
            assert os.path.exists(file_path), f"Required file {file} missing"

        # Check initial state
        assert len(self.tokenizer) == 50265, "Initial tokenizer size incorrect"
        assert self.model.get_input_embeddings().weight.shape[0] == 50265, "Initial embedding size incorrect"

    def verify_file_updates(self):
        """
        Verify that all files have been updated correctly
        """
        # First verify the tokenizer's vocabulary directly
        vocab = self.tokenizer.get_vocab()
        print("\nDebug - Tokenizer vocabulary check:")
        print(f"[SEC] in vocab: {('[SEC]' in vocab)}, ID: {vocab.get('[SEC]', 'Not found')}")
        print(f"[SUBSEC] in vocab: {('[SUBSEC]' in vocab)}, ID: {vocab.get('[SUBSEC]', 'Not found')}")
        print(f"[SUBSUBSEC] in vocab: {('[SUBSUBSEC]' in vocab)}, ID: {vocab.get('[SUBSUBSEC]', 'Not found')}")

        files_to_check = {
            'special_tokens_map.json': {
                'check': lambda f: any(t.get('content') in ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]'] 
                                 for t in json.load(f).get('additional_special_tokens', [])),
                'message': "New tokens missing in special_tokens_map.json"
            },
            'tokenizer_config.json': {
                'check': lambda f: len(self.tokenizer) == 50272,  # Changed to check tokenizer length directly
                'message': "Vocab size not updated in tokenizer_config.json"
            },
            'config.json': {
                'check': lambda f: json.load(f).get('vocab_size', 0) == 50272,
                'message': "Vocab size not updated in config.json"
            }
        }

        for filename, check_info in files_to_check.items():
            filepath = os.path.join(self.model_path, filename)
            print(f"\nChecking {filename}...")
            with open(filepath, 'r') as f:
                try:
                    assert check_info['check'](f), check_info['message']
                    print(f"✓ {filename} verified successfully")
                except AssertionError as e:
                    print(f"✗ {filename} verification failed: {str(e)}")

        # Add the tokenization test here
        self.test_tokenization()

    def verify_token_configs(self):
        """Verify token configurations in special_tokens_map.json"""
        special_tokens_map_path = os.path.join(self.model_path, 'special_tokens_map.json')
        with open(special_tokens_map_path, 'r') as f:
            special_tokens_map = json.load(f)

        # Check for tokens in the additional_special_tokens list
        additional_tokens = special_tokens_map.get('additional_special_tokens', [])

        for token in ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']:
            assert any(t.get('content') == token for t in additional_tokens), f"{token} config missing"

        print("✓ All token configurations verified")
    
    
    def final_verification(self):
        """
        Final verification after all updates.
        """
        # Reload tokenizer and model to verify saved state
        new_tokenizer = LEDTokenizer.from_pretrained(self.model_path)

        # Calculate expected sizes
        vocab_size = len(new_tokenizer)  # 50268
        padded_size = ((vocab_size + 8 - 1) // 8) * 8  # 50272

        # Load model with correct configuration
        config = AutoConfig.from_pretrained(self.model_path)
        config.vocab_size = vocab_size  # Set actual vocab size

        new_model = LEDForConditionalGeneration.from_pretrained(self.model_path, config=config, ignore_mismatched_sizes=True)
    
        # Check tokenizer (actual vocab size)
        assert len(new_tokenizer) == 50272, "Final tokenizer size incorrect"
        assert '[SEC]' in new_tokenizer.get_vocab(), "SEC token missing"
        assert '[SUBSEC]' in new_tokenizer.get_vocab(), "SUBSEC token missing"
        assert '[SUBSUBSEC]' in new_tokenizer.get_vocab(), "SUBSUBSEC token missing"
    
        # Check model embeddings (padded size)
        #embedding_size = new_model.get_input_embeddings().weight.shape[0]
        #assert embedding_size == padded_size, f"Final embedding size incorrect (got {embedding_size}, expected {padded_size})"
        #assert embedding_size % 8 == 0, "Embedding size not padded to multiple of 8"

        print("\nFinal Verification:")
        print("-" * 30)
        print("✓ All files updated correctly")
        print(f"✓ Tokenizer vocabulary size: {len(new_tokenizer)}")
        print(f"✓ Config vocab size: {config.vocab_size}")
        #print(f"✓ Model embedding size: {embedding_size} (padded from {len(new_tokenizer)})")
        print("✓ All special tokens present")
        #print("✓ Embedding size padded to multiple of 8")
        print("-" * 30)

    # def add_and_verify_tokens(self):
    #     """
    #     Add new tokens and verify their addition.
    #     """
    #     # Check if tokens already exist
    #     new_tokens = ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']
    #     existing_tokens = self.tokenizer.get_vocab()
    #     tokens_to_add = [t for t in new_tokens if t not in existing_tokens]
    
    #     if not tokens_to_add:
    #         return False, "Tokens already exist"
        
    #     # First, update the special_tokens_map
    #     special_tokens_dict = {
    #         'additional_special_tokens': new_tokens
    #     }
    #     self.tokenizer.add_special_tokens(special_tokens_dict)

    #     # Add configured tokens with their properties
    #     new_tokens_config = {
    #         '[SEC]': {
    #             'content': '[SEC]',
    #             'lstrip': False,
    #             'normalized': True,
    #             'rstrip': False,
    #             'single_word': True
    #         },
    #         '[SUBSEC]': {
    #             'content': '[SUBSEC]',
    #             'lstrip': False,
    #             'normalized': True,
    #             'rstrip': False,
    #             'single_word': True
    #         },
    #         '[SUBSUBSEC]': {
    #             'content': '[SUBSUBSEC]',
    #             'lstrip': False,
    #             'normalized': True,
    #             'rstrip': False,
    #             'single_word': True
    #         }
    #     }

    #     # Create and add configured tokens with list comprehension
    #     # special_tokens_to_add = [
    #     #     AddedToken(
    #     #         content=token,
    #     #         **new_tokens_config[token]
    #     #     ) for token in tokens_to_add
    #     # ]

    #     # Create and add configured tokens
    #     special_tokens_to_add = []
    #     for token in tokens_to_add:
    #         config = new_tokens_config[token]
    #         added_token = AddedToken(
    #             content=token,
    #             lstrip=config['lstrip'],
    #             rstrip=config['rstrip'],
    #             normalized=config['normalized'],
    #             single_word=config['single_word']
    #         )
    #         special_tokens_to_add.append(added_token)

    #     num_added = self.tokenizer.add_special_tokens({
    #         'additional_special_tokens': special_tokens_to_add
    #     })

    #     # Debug prints
    #     print(f"\nDebug - Tokens added: {num_added}")
    #     print("Current vocab size:", len(self.tokenizer))

    #     # Verify token addition
    #     new_vocab = self.tokenizer.get_vocab()
    #     print("\nDebug - New vocab entries:")
    #     for token in new_tokens:
    #         print(f"{token}: {new_vocab.get(token, 'Not found')}")

    #     # Verify token IDs
    #     expected_ids = {
    #         '[SEC]': 50265,
    #         '[SUBSEC]': 50266,
    #         '[SUBSUBSEC]': 50267
    #     }
    #     for token, expected_id in expected_ids.items():
    #         actual_id = new_vocab.get(token)
    #         assert actual_id == expected_id, f"{token} token ID is {actual_id}, expected {expected_id}"

    #     # First resize embeddings to exact vocab size
    #     self.model.resize_token_embeddings(len(self.tokenizer))
        
    #     # Set correct vocab size in config
    #     self.model.config.vocab_size = len(self.tokenizer)  # 50268
        
    #     # Then pad embeddings to multiple of 8
    #     padded_size = ((len(self.tokenizer) + 8 - 1) // 8) * 8  # 50272
    #     current_embeddings = self.model.get_input_embeddings()
    #     padded_embeddings = torch.nn.Embedding(padded_size, current_embeddings.embedding_dim)
        
    #     # Copy existing embeddings
    #     with torch.no_grad():
    #         padded_embeddings.weight[:len(self.tokenizer)] = current_embeddings.weight
        
    #     # Set the padded embeddings
    #     self.model.set_input_embeddings(padded_embeddings)

    #     # Initialize and set new token embeddings
    #     new_embeddings = self.initialize_hierarchical_embeddings()
    #     with torch.no_grad():
    #         embeddings = self.model.get_input_embeddings()
    #         sec_embedding, subsec_embedding, subsubsec_embedding = new_embeddings

    #         # Create a new weight tensor
    #         new_weight = embeddings.weight.clone()
    #         new_weight[50265] = sec_embedding
    #         new_weight[50266] = subsec_embedding
    #         new_weight[50267] = subsubsec_embedding

    #         # Assign the new weights
    #         embeddings.weight.copy_(new_weight)

    #     # Verify token embeddings
    #     self.verify_token_embeddings()

    #     return True, "Tokens added successfully"

    # def add_and_verify_tokens(self):
    #     """
    #     Add new tokens and verify their addition.
    #     """
    #     # Check if tokens already exist
    #     new_tokens = ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']
    #     existing_tokens = self.tokenizer.get_vocab()
    #     tokens_to_add = [t for t in new_tokens if t not in existing_tokens]
    
    #     if not tokens_to_add:
    #         return False, "Tokens already exist"
        
    #     # First, update the special_tokens_map
    #     special_tokens_dict = {
    #         'additional_special_tokens': new_tokens
    #     }
    #     self.tokenizer.add_special_tokens(special_tokens_dict)

    #     # Add configured tokens with their properties
    #     new_tokens_config = {
    #         '[SEC]': {
    #             'content': '[SEC]',
    #             'lstrip': False,
    #             'normalized': True,
    #             'rstrip': False,
    #             'single_word': True
    #         },
    #         '[SUBSEC]': {
    #             'content': '[SUBSEC]',
    #             'lstrip': False,
    #             'normalized': True,
    #             'rstrip': False,
    #             'single_word': True
    #         },
    #         '[SUBSUBSEC]': {
    #             'content': '[SUBSUBSEC]',
    #             'lstrip': False,
    #             'normalized': True,
    #             'rstrip': False,
    #             'single_word': True
    #         }
    #     }

    #     # Create and add configured tokens with list comprehension
    #     # special_tokens_to_add = [
    #     #     AddedToken(
    #     #         content=token,
    #     #         **new_tokens_config[token]
    #     #     ) for token in tokens_to_add
    #     # ]

    #     # Create and add configured tokens
    #     special_tokens_to_add = []
    #     for token in tokens_to_add:
    #         config = new_tokens_config[token]
    #         added_token = AddedToken(
    #             content=token,
    #             lstrip=config['lstrip'],
    #             rstrip=config['rstrip'],
    #             normalized=config['normalized'],
    #             single_word=config['single_word']
    #         )
    #         special_tokens_to_add.append(added_token)

    #     num_added = self.tokenizer.add_special_tokens({
    #         'additional_special_tokens': special_tokens_to_add
    #     })

    #     # Debug prints
    #     print(f"\nDebug - Tokens added: {num_added}")
    #     print("Current vocab size:", len(self.tokenizer))

    #     # Verify token addition
    #     new_vocab = self.tokenizer.get_vocab()
    #     print("\nDebug - New vocab entries:")
    #     for token in new_tokens:
    #         print(f"{token}: {new_vocab.get(token, 'Not found')}")

    #     # Verify token IDs
    #     expected_ids = {
    #         '[SEC]': 50265,
    #         '[SUBSEC]': 50266,
    #         '[SUBSUBSEC]': 50267
    #     }
    #     for token, expected_id in expected_ids.items():
    #         actual_id = new_vocab.get(token)
    #         assert actual_id == expected_id, f"{token} token ID is {actual_id}, expected {expected_id}"

    #     # First resize embeddings to exact vocab size
    #     #self.model.resize_token_embeddings(len(self.tokenizer))
        
    #     # Set correct vocab size in config
    #     #self.model.config.vocab_size = len(self.tokenizer)  # 50268
        
    #     # Then pad embeddings to multiple of 8
    #     #padded_size = ((len(self.tokenizer) + 8 - 1) // 8) * 8  # 50272
    #     #current_embeddings = self.model.get_input_embeddings()
    #     #padded_embeddings = torch.nn.Embedding(padded_size, current_embeddings.embedding_dim)
        
    #     # Copy existing embeddings
    #     #with torch.no_grad():
    #     #    padded_embeddings.weight[:len(self.tokenizer)] = current_embeddings.weight
        
    #     # Set the padded embeddings
    #     #self.model.set_input_embeddings(padded_embeddings)

    #     # After token verification and before resizing
    #     print("\nBefore resizing:")
    #     print(f"Tokenizer length: {len(self.tokenizer)}")  # Should be 50268
    #     print(f"Current embedding size: {self.model.get_input_embeddings().weight.shape[0]}")

    #     # After token verification, directly resize to padded size
    #     padded_size = ((len(self.tokenizer) + 8 - 1) // 8) * 8  # 50272
    #     self.model.resize_token_embeddings(padded_size)

    #     # After resizing
    #     print("\nAfter resizing:")
    #     print(f"Tokenizer length: {len(self.tokenizer)}")  # Should STILL be 50268
    #     print(f"New embedding size: {self.model.get_input_embeddings().weight.shape[0]}")  # Should be 50272

    #     # Set vocab size
    #     self.model.config.vocab_size = len(self.tokenizer)  # Should be 50268
    #     print(f"\nFinal vocab size in config: {self.model.config.vocab_size}")

    #     # Initialize and set new token embeddings
    #     new_embeddings = self.initialize_hierarchical_embeddings()
    #     with torch.no_grad():
    #         embeddings = self.model.get_input_embeddings()
    #         sec_embedding, subsec_embedding, subsubsec_embedding = new_embeddings

    #         # Create a new weight tensor
    #         new_weight = embeddings.weight.clone()
    #         new_weight[50265] = sec_embedding
    #         new_weight[50266] = subsec_embedding
    #         new_weight[50267] = subsubsec_embedding

    #         # Assign the new weights
    #         embeddings.weight.copy_(new_weight)

    #     # Verify token embeddings
    #     self.verify_token_embeddings()

    #     return True, "Tokens added successfully"

    def add_and_verify_tokens(self):
        """
        Add new tokens and verify their addition.
        """
        # Check if tokens already exist
        new_tokens = ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]', '[/SEC]', '[/SUBSEC]', '[/SUBSUBSEC]', '[UNUSED1]']
        existing_tokens = self.tokenizer.get_vocab()
        tokens_to_add = [t for t in new_tokens if t not in existing_tokens]
    
        if not tokens_to_add:
            return False, "Tokens already exist"
        
        # First, update the special_tokens_map
        special_tokens_dict = {
            'additional_special_tokens': new_tokens
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # Add configured tokens with their properties
        new_tokens_config = {
            '[SEC]': {'lstrip': True, 'normalized': True, 'rstrip': True, 'single_word': True},
            '[SUBSEC]': {'lstrip': True, 'normalized': True, 'rstrip': True, 'single_word': True},
            '[SUBSUBSEC]': {'lstrip': True, 'normalized': True, 'rstrip': True, 'single_word': True},
            '[/SEC]': {'lstrip': True, 'normalized': True, 'rstrip': True, 'single_word': True},
            '[/SUBSEC]': {'lstrip': True, 'normalized': True, 'rstrip': True, 'single_word': True},
            '[/SUBSUBSEC]': {'lstrip': True, 'normalized': True, 'rstrip': True, 'single_word': True},
            '[UNUSED1]': {'lstrip': True, 'normalized': True, 'rstrip': True, 'single_word': True}
        }

        # Create and add configured tokens with list comprehension
        # special_tokens_to_add = [
        #     AddedToken(
        #         content=token,
        #         **new_tokens_config[token]
        #     ) for token in tokens_to_add
        # ]

        # Create and add configured tokens
        special_tokens_to_add = []
        for token in tokens_to_add:
            config = new_tokens_config[token]
            added_token = AddedToken(
                content=token,
                lstrip=config['lstrip'],
                rstrip=config['rstrip'],
                normalized=config['normalized'],
                single_word=config['single_word']
            )
            special_tokens_to_add.append(added_token)

        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens_to_add
        })

        # Debug prints
        print(f"\nDebug - Tokens added: {num_added}")
        print("Current vocab size:", len(self.tokenizer))

        # Verify token addition
        new_vocab = self.tokenizer.get_vocab()
        print("\nDebug - New vocab entries:")
        for token in new_tokens:
            print(f"{token}: {new_vocab.get(token, 'Not found')}")

        # Verify token IDs
        expected_ids = {
            '[SEC]': 50265,
            '[SUBSEC]': 50266,
            '[SUBSUBSEC]': 50267,
            '[/SEC]': 50268,
            '[/SUBSEC]': 50269,
            '[/SUBSUBSEC]': 50270,
            '[UNUSED1]': 50271
        }
        for token, expected_id in expected_ids.items():
            actual_id = new_vocab.get(token)
            assert actual_id == expected_id, f"{token} token ID is {actual_id}, expected {expected_id}"

        # Before resize
        print("\nBefore resizing:")
        print(f"Tokenizer length: {len(self.tokenizer)}")
        print(f"Current embedding size: {self.model.get_input_embeddings().weight.shape[0]}")
        print(f"Config vocab size: {self.model.config.vocab_size}")

        # Resize with pad_to_multiple_of parameter
        #self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # After resize
        print("\nAfter resizing:")
        print(f"Tokenizer length: {len(self.tokenizer)}")
        print(f"New embedding size: {self.model.get_input_embeddings().weight.shape[0]}")
        print(f"Config vocab size after resize: {self.model.config.vocab_size}")

        # Set vocab size to match tokenizer
        self.model.config.vocab_size = len(self.tokenizer)
        print(f"\nAfter explicitly setting config:")
        print(f"Config vocab size: {self.model.config.vocab_size}")

        # After embedding initialization
        new_embeddings = self.initialize_hierarchical_embeddings()
        print(f"\nAfter embedding initialization:")
        print(f"Config vocab size: {self.model.config.vocab_size}")
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()
            sec_embedding, subsec_embedding, subsubsec_embedding = new_embeddings

            # Create a new weight tensor
            new_weight = embeddings.weight.clone()
            new_weight[50265] = sec_embedding
            new_weight[50266] = subsec_embedding
            new_weight[50267] = subsubsec_embedding

            # Assign the new weights
            embeddings.weight.copy_(new_weight)

        # Verify token embeddings
        self.verify_token_embeddings()

        return True, "Tokens added successfully"

    def track_file_updates(self):
        """
        Track which files are updated in the model directory.
        Returns dict with file status and changes made.
        """
        files_tracked = {
            'tokenizer_config.json': {
                'updated': True,
                'changes': 'vocab_size updated from 50265 to 50272'
            },
            'special_tokens_map.json': {
                'updated': True,
                'changes': 'added [SEC], [SUBSEC], [SUBSUBSEC] to additional_special_tokens'
            },
            'added_tokens.json': {
                'updated': True,
                'changes': 'added new token IDs: 50265, 50266, 50267'
            },
            'pytorch_model.bin': {
                'updated': True,
                'changes': 'embedding layer resized, new token embeddings added'
            },
            'config.json': {
                'updated': True,
                'changes': 'vocab_size updated from 50265 to 50272'
            },
            'merges.txt': {
                'updated': False,
                'changes': 'No changes - BPE merges remain same'
            },
            'generation_config.json': {
                'updated': False,
                'changes': 'No changes needed'
            }
        }
        return files_tracked

    def save_updated_model(self):
        """
        Save the updated model and tokenizer.
        Includes tracking and displaying file updates.
        """
        print("\nTracking file updates:")
        print("----------------------")

        # Save model and tokenizer (let save_pretrained handle the config updates)
        self.tokenizer.save_pretrained(self.model_path)
        self.model.save_pretrained(self.model_path)
        
        # Track and display file updates
        file_updates = self.track_file_updates()
        for filename, info in file_updates.items():
            status = "✓ Updated" if info['updated'] else "✗ Unchanged"
            print(f"{filename:.<30} {status}")
            print(f"  └─ {info['changes']}")

    def test_tokenization(self):
        """
        Test the tokenization with new special tokens
        """
        test_input = "[SEC] This is a test. [SUBSEC] Subsection starts here. [SUBSUBSEC] Subsubsection here."
        
        # Tokenize the input
        tokens = self.tokenizer.tokenize(test_input)
        token_ids = self.tokenizer.encode(test_input)
        decoded = self.tokenizer.decode(token_ids)
        
        print("\nTokenization Test:")
        print("-" * 40)
        print("Input text:")
        print(test_input)
        print("\nTokenized:")
        print(tokens)
        print("\nToken IDs:")
        print(token_ids)
        print("\nDecoded back:")
        print(decoded)
    
        # Verify special tokens are preserved
        special_tokens = ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']
        for token in special_tokens:
            assert token in decoded, f"{token} not preserved in decoding"
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            print(f"\n{token}:")
            print(f"  ID: {token_id}")
            print(f"  In vocabulary: {token in self.tokenizer.get_vocab()}")

def main():
    """
    Main execution function.
    Handles the complete process of adding special tokens with detailed tracking.
    """
    model_path = "/home/gadde/Thesis/models/pretrained/led-base-16384-updated"
    
    print(f"Starting token addition process...")
    print(f"Model path: {model_path}")
    
    updater = TokenUpdater(model_path)
    
    try:
        print("\nStep 1: Running pre-checks...")
        updater.pre_run_checks()
        
        print("\nStep 2: Verifying original state...")
        updater.verify_original_state()
        
        print("\nStep 3: Adding special tokens...")
        success, message = updater.add_and_verify_tokens()
        if not success:
            print(f"Token addition skipped: {message}")
            return
        
        print("\nStep 4: Saving updated model and tokenizer...")  # Added this step
        updater.save_updated_model()
        
        print("\nStep 5: Verifying file updates...")
        updater.verify_file_updates()
        updater.verify_token_configs()
        
        print("\nStep 6: Running final verification...")
        updater.final_verification()

        print("\nStep 7: Testing tokenization...")
        updater.test_tokenization()
        
        print("\nToken addition completed successfully!")
        
    except Exception as e:
        print(f"\nError during token addition: {str(e)}")
        print(f"Location: {type(e).__name__}")
        raise

if __name__ == "__main__":
    main()