import json
from pathlib import Path

def check_vocab_tokens():
    # Path to vocab.json
    vocab_path = Path("/home/gadde/Thesis/models/pretrained/led-base-16384-updated/vocab.json")
    
    # Read vocab.json
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Method 1: Get tokens by their IDs
    token_ids = {50265, 50266, 50267}
    tokens_by_id = {id: token for token, id in vocab.items() if id in token_ids}
    
    # Method 2: Get specific tokens
    specific_tokens = {'[SEC]', '[SUBSEC]', '[SUBSUBSEC]'}
    tokens_by_name = {token: vocab[token] for token in specific_tokens if token in vocab}
    
    print("\nTokens found by ID:")
    for id, token in tokens_by_id.items():
        print(f"ID {id}: {token}")
        
    print("\nTokens found by name:")
    for token, id in tokens_by_name.items():
        print(f"Token {token}: {id}")

if __name__ == "__main__":
    check_vocab_tokens()