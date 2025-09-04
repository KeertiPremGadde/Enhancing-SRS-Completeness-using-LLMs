import json
import re
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import LEDTokenizer
from typing import Dict, List, Tuple
from tqdm import tqdm
from training.seeding import seed_worker, get_torch_generator

logging.basicConfig(
    #level=logging.INFO,
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger("LEDDataLoader")

# def flatten_srs_to_text(_, srs_dict) -> str:
#         flat_text = ""

#         def clean_key(key):
#             return re.sub(r"\[(SEC|SUBSEC|SUBSUBSEC)\]\s*", "", key).strip()

#         def recurse(d):
#             nonlocal flat_text
#             if isinstance(d, dict):
#                 for k, v in d.items():
#                     logger.debug(f"Original key: '{k}', cleaned: '{clean_key(k)}'")
#                     flat_text += f"{clean_key(k)}\n"
#                     recurse(v)
#             elif isinstance(d, list):
#                 flat_text += "\n".join(str(item) for item in d) + "\n"
#             elif isinstance(d, str):
#                 flat_text += d + "\n"

#         recurse(srs_dict)
#         return flat_text



class LEDDataLoader:

    #@staticmethod
    # def seed_worker(worker_id):
    #     worker_seed = torch.initial_seed() % 2**32
    #     np.random.seed(worker_seed)
    #     random.seed(worker_seed)
    #     logger.debug(f"Seeded worker {worker_id} with seed {worker_seed}")

    def __init__(self, config_path: str = "/home/gadde/Thesis/configs/config_led.json"):
        logger.info(f"Initializing LED DataLoader with config path: {config_path}")
        self.config = self._load_config(config_path)
        self.max_input_length = self.config['model'].get('max_input_tokens', 16384)
        self.max_output_length = self.config['model'].get('max_output_tokens', 1024)

        logger.info(f"✓ Expected input_ids: [batch_size, {self.max_input_length}]")
        logger.info(f"✓ Expected labels:    [batch_size, {self.max_output_length}]")

        assert self.max_input_length <= 16384
        assert self.max_output_length <= 1024

        self.tokenizer = None
        self._validate_paths()
        self.load_tokenizer()

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return json.load(f)

    def _validate_paths(self) -> None:
        paths = {
            "Model path": self.config["model"]["path"],
            "Tokenizer path": self.config["model"]["tokenizer_path"]
        }
        for name, path in paths.items():
            if not os.path.exists(path):
                logger.error(f"❌ {name} not found: {path}")
            else:
                logger.info(f"✓ {name} exists: {path}")

    def load_tokenizer(self) -> None:
        self.tokenizer = LEDTokenizer.from_pretrained(self.config["model"]["tokenizer_path"])
        #self.tokenizer.add_tokens(['[SEC]', '[SUBSEC]', '[SUBSUBSEC]'])

        # ✅ Verification logs
        for tok in ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']:
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            logger.info(f"Token: {tok} → ID: {tok_id}")
            if tok_id == self.tokenizer.unk_token_id:
                logger.warning(f"❌ Token {tok} not found in tokenizer vocabulary!")

        # ✅ Optional: strict safety check (will raise error if any special token is missing)
        missing = [tok for tok in ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']
               if self.tokenizer.convert_tokens_to_ids(tok) == self.tokenizer.unk_token_id]
        if missing:
            raise ValueError(f"The tokenizer is missing these special tokens: {missing}")


#Flattening for non hierarchical tokens
    # def flatten_srs_to_text(self, srs_dict) -> str:
    #     flat_text = ""
    #     def recurse(d):
    #         nonlocal flat_text
    #         if isinstance(d, dict):
    #             for k, v in d.items():
    #                 flat_text += f"{k}\n"
    #                 recurse(v)
    #         elif isinstance(d, list):
    #             flat_text += "\n".join(str(item) for item in d) + "\n"
    #         elif isinstance(d, str):
    #             flat_text += d + "\n"
    #     recurse(srs_dict)
    #     return flat_text

    


    

#Flattening for hierarchical tokens
    # def flatten_srs_to_text(self, srs_dict) -> str:
    #     def flatten(node):
    #         text = ''
    #         if isinstance(node, dict):
    #             for key, value in node.items():
    #                 if key.startswith("1."):
    #                     text += f"[SEC] {key}\n"
    #                 elif re.match(r"\d\.\d+", key):
    #                     text += f"[SUBSEC] {key}\n"
    #                 elif re.match(r"\d\.\d+\.\d+", key):
    #                     text += f"[SUBSUBSEC] {key}\n"
    #                 else:
    #                     text += f"{key}\n"
    #                     text += flatten(value)
    #         elif isinstance(node, list):
    #             text += '\n'.join(str(item) for item in node) + '\n'
    #         else:
    #             text += str(node) + '\n'
    #         return text
    #     return flatten(srs_dict)

#missing Req
    # @staticmethod
    # def flatten_srs_to_text(srs_dict) -> str:
    #     def flatten(node):
    #         text = ''
    #         if isinstance(node, dict):
    #             for key, value in node.items():
    #                 if key.startswith("[SEC]"):
    #                     text += f"{key}\n"
    #                 elif key.startswith("[SUBSEC]"):
    #                     text += f"{key}\n"
    #                 elif key.startswith("[SUBSUBSEC]"):
    #                     text += f"{key}\n"
    #                 text += flatten(value)
    #         elif isinstance(node, list):
    #             text += '\n'.join(str(item) for item in node) + '\n'
    #         else:
    #             text += str(node) + '\n'
    #         return text

    #     return flatten(srs_dict)

    @staticmethod
    def flatten_srs_to_text(srs_dict) -> str:
        def flatten(node):
            text = ''
            if isinstance(node, dict):
                for key, value in node.items():
                    if key.startswith("[SEC]") or key.startswith("[SUBSEC]") or key.startswith("[SUBSUBSEC]"):
                        text += f"{key}\n"
                        text += flatten(value)
                    elif re.match(r"REQ-[A-Z]+-\d+", key):  # <-- this catches REQ IDs
                        text += f"{key}\n{value}\n"
                    else:
                        text += f"{key}\n"
                        text += flatten(value)
            elif isinstance(node, list):
                text += '\n'.join(str(item) for item in node) + '\n'
            else:
                text += str(node) + '\n'
            return text

        return flatten(srs_dict)

    def process_file(self, file_info: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            input_path = os.path.join("/home/gadde/Thesis", file_info['input'])
            output_path = os.path.join("/home/gadde/Thesis", file_info['output'])

            with open(input_path, 'r') as f:
                #input_text = self.flatten_srs_to_text(json.load(f))
                input_text = LEDDataLoader.flatten_srs_to_text(json.load(f))

            with open(output_path, 'r') as f:
                #output_text = self.flatten_srs_to_text(json.load(f))
                output_text = LEDDataLoader.flatten_srs_to_text(json.load(f))


            input_encoding = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            output_encoding = self.tokenizer(
                output_text,
                max_length=self.max_output_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            global_attention_mask = torch.zeros_like(input_encoding['input_ids'])
            
            #Comment out for non hierarchical tokens
            for i, token_id in enumerate(input_encoding['input_ids'][0]):
                token = self.tokenizer.decode([token_id])
                if any(marker in token for marker in ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']):
                    global_attention_mask[0][i] = 1

            # Enable one global token (usually the first) for LED to function properly
            #global_attention_mask[0][0] = 1
            # Always ensure at least one token has global attention
            #if global_attention_mask.sum() == 0:
            #    global_attention_mask[0][0] = 1

            
            #loss for special tokens
            # labels = output_encoding["input_ids"].squeeze(0)
            # labels[labels == self.tokenizer.pad_token_id] = -100

            
            return (
                input_encoding["input_ids"],
                input_encoding["attention_mask"],
                global_attention_mask,
                output_encoding["input_ids"]
                #labels.unsqueeze(0)  # Maintain batch dimension, special token loss
            )

        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            return None

    def __getitem__(self, idx):
        if not hasattr(self, 'processed_items'):
            self.processed_items = self.load_dataset()
        item = self.processed_items[idx]
        return {
            'input_ids': item[0][:, :self.max_input_length],
            'attention_mask': item[1][:, :self.max_input_length],
            'global_attention_mask': item[2][:, :self.max_input_length],
            'labels': item[3][:, :self.max_output_length]
        }

    def load_dataset(self, split: str = "train") -> List[Tuple]:
        if not self.tokenizer:
            self.load_tokenizer()
        mapping_path = self.config["data"][f"{split}_mapping"]
        with open(mapping_path, 'r') as f:
            file_mapping = json.load(f)

        processed_items = []
        skipped = 0
        for file_info in tqdm(file_mapping, desc="Loading dataset"):
            result = self.process_file(file_info)
            if result:
                processed_items.append(result)
            else:
                skipped += 1
        return processed_items

    @staticmethod
    def collate_fn(batch):
        input_ids, attention_masks, global_attention_masks, labels = zip(*batch)
        return {
            'input_ids': torch.cat(input_ids, dim=0),
            'attention_mask': torch.cat(attention_masks, dim=0),
            'global_attention_mask': torch.cat(global_attention_masks, dim=0),
            'labels': torch.cat(labels, dim=0)
        }

    # def get_dataloader(self, split: str = "train") -> DataLoader:
    #     dataset = self.load_dataset(split)
    #     return DataLoader(
    #         dataset,
    #         batch_size=self.config["training"]["batch_size"],
    #         shuffle=(split == "train"),
    #         collate_fn=self.collate_fn
    #     )

    def get_dataloader(self, split: str = "train") -> DataLoader:
        dataset = self.load_dataset(split)
    
        if split == "train":
            seed = self.config.get("seed", 42)
            generator = torch.Generator()
            generator.manual_seed(seed)
            return DataLoader(
                dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=True,
                collate_fn=self.collate_fn,
                #worker_init_fn=self.seed_worker,
                #generator=generator,
                worker_init_fn=seed_worker,                    # 🔄 From training/seeding.py
                generator=get_torch_generator(seed)            # 🔄 From training/seeding.py
            )
        else:
        # No randomness in val/test
            return DataLoader(
                dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                collate_fn=self.collate_fn
            )

__all__ = ['LEDDataLoader']
if __name__ == "__main__":
    try:
        logger.info("✅ Debug: Initializing and testing LEDDataLoader")
        loader = LEDDataLoader()
        train_loader = loader.get_dataloader("train")
        for batch in train_loader:
            logger.info(f"✓ Test batch loaded: input shape {batch['input_ids'].shape}, labels shape {batch['labels'].shape}")
            
            # ✅ Decode the input and output (first item in batch)
            input_text = loader.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
            label_text = loader.tokenizer.decode(batch["labels"][0], skip_special_tokens=False)

            print("\n\n✅ DECODED INPUT TEXT SAMPLE:\n")
            print(input_text)

            print("\n\n✅ DECODED LABEL TEXT SAMPLE:\n")
            print(label_text)

            # Optional: look for REQ IDs or section headers
            assert "REQ-" in input_text or "[SEC]" in input_text, "❌ No REQ ID or section token found in input!"
            assert "REQ-" in label_text or "[SEC]" in label_text, "❌ No REQ ID or section token found in label!"
            
            
            
            break
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        raise