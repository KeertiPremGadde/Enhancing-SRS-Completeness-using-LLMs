# import os
# import json
# import logging
# import random
# import re
# from typing import Dict, List, Tuple

# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
# from tqdm import tqdm

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
# )
# logger = logging.getLogger("LongT5DataLoader")

# class LongT5DataLoader:

#     @staticmethod
#     def seed_worker(worker_id):
#         worker_seed = torch.initial_seed() % 2**32
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)

#     def __init__(self, config_path: str = "/home/gadde/Thesis/configs/config_long_t5.json"):
#         logger.info(f"Initializing LongT5 DataLoader with config: {config_path}")
#         self.config = self._load_config(config_path)
#         self.max_input_length = self.config['model']['max_input_tokens']
#         self.max_output_length = self.config['model']['max_output_tokens']

#         self.tokenizer = None
#         self._validate_paths()
#         self.load_tokenizer()

#     def _load_config(self, config_path: str) -> Dict:
#         with open(config_path, 'r') as f:
#             return json.load(f)

#     def _validate_paths(self) -> None:
#         paths = {
#             "Model path": self.config["model"]["path"],
#             "Tokenizer path": self.config["model"]["tokenizer_path"]
#         }
#         for name, path in paths.items():
#             if not os.path.exists(path):
#                 logger.error(f"❌ {name} not found: {path}")
#             else:
#                 logger.info(f"✓ {name} exists: {path}")

#     def load_tokenizer(self) -> None:
#         self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["tokenizer_path"])
#         self.tokenizer.add_tokens(['[SEC]', '[SUBSEC]', '[SUBSUBSEC]'])

#     # === With structural tokens, no punctuation in keys ===
#     def flatten_srs_to_text(self, srs_dict) -> str:
#         def flatten(node):
#             text = ''
#             if isinstance(node, dict):
#                 for key, value in node.items():
#                     if key.startswith("1."):
#                         text += f"[SEC] {key}\n"
#                     elif re.match(r"\d+\.\d+$", key):
#                         text += f"[SUBSEC] {key}\n"
#                     elif re.match(r"\d+\.\d+\.\d+$", key):
#                         text += f"[SUBSUBSEC] {key}\n"
#                     else:
#                         text += f"{key}\n"
#                     text += flatten(value)
#             elif isinstance(node, list):
#                 text += '\n'.join(str(item) for item in node) + '\n'
#             else:
#                 text += str(node) + '\n'
#             return text
#         return flatten(srs_dict)

#     # === Alternative version with punctuation (commented out) ===
#     # def flatten_srs_to_text(self, srs_dict) -> str:
#     #     flat_text = ""
#     #     def recurse(d):
#     #         nonlocal flat_text
#     #         if isinstance(d, dict):
#     #             for k, v in d.items():
#     #                 flat_text += f"{k}\n"
#     #                 recurse(v)
#     #         elif isinstance(d, list):
#     #             flat_text += "\n".join(str(item) for item in d) + "\n"
#     #         elif isinstance(d, str):
#     #             flat_text += d + "\n"
#     #     recurse(srs_dict)
#     #     return flat_text

#     def process_file(self, file_info: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         input_path = os.path.join("/home/gadde/Thesis", file_info['input'])
#         output_path = os.path.join("/home/gadde/Thesis", file_info['output'])

#         with open(input_path, 'r') as f:
#             input_text = self.flatten_srs_to_text(json.load(f))
#         with open(output_path, 'r') as f:
#             output_text = self.flatten_srs_to_text(json.load(f))

#         input_encoding = self.tokenizer(
#             input_text,
#             max_length=self.max_input_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
#         output_encoding = self.tokenizer(
#             output_text,
#             max_length=self.max_output_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )

#         return (
#             input_encoding["input_ids"],
#             input_encoding["attention_mask"],
#             output_encoding["input_ids"]
#         )

#     def __getitem__(self, idx):
#         if not hasattr(self, 'processed_items'):
#             self.processed_items = self.load_dataset()
#         item = self.processed_items[idx]
#         return {
#             'input_ids': item[0][:, :self.max_input_length],
#             'attention_mask': item[1][:, :self.max_input_length],
#             'labels': item[2][:, :self.max_output_length]
#         }

#     def load_dataset(self, split: str = "train") -> List[Tuple]:
#         if not self.tokenizer:
#             self.load_tokenizer()
#         mapping_path = self.config["data"][f"{split}_mapping"]
#         with open(mapping_path, 'r') as f:
#             file_mapping = json.load(f)

#         processed_items = []
#         skipped = 0
#         for file_info in tqdm(file_mapping, desc=f"Loading {split} dataset"):
#             result = self.process_file(file_info)
#             if result:
#                 processed_items.append(result)
#             else:
#                 skipped += 1
#         return processed_items

#     @staticmethod
#     def collate_fn(batch):
#         input_ids, attention_masks, labels = zip(*batch)
#         return {
#             'input_ids': torch.cat(input_ids, dim=0),
#             'attention_mask': torch.cat(attention_masks, dim=0),
#             'labels': torch.cat(labels, dim=0)
#         }

#     def get_dataloader(self, split: str = "train") -> DataLoader:
#         dataset = self.load_dataset(split)

#         if split == "train":
#             seed = self.config.get("seed", 42)
#             generator = torch.Generator()
#             generator.manual_seed(seed)
#             return DataLoader(
#                 dataset,
#                 batch_size=self.config["training"]["batch_size"],
#                 shuffle=True,
#                 collate_fn=self.collate_fn,
#                 worker_init_fn=self.seed_worker,
#                 generator=generator
#             )
#         else:
#             return DataLoader(
#                 dataset,
#                 batch_size=self.config["training"]["batch_size"],
#                 shuffle=False,
#                 collate_fn=self.collate_fn
#             )

# if __name__ == "__main__":
#     try:
#         logger.info("✅ Debug: Initializing and testing LongT5DataLoader")
#         loader = LongT5DataLoader()
#         train_loader = loader.get_dataloader("train")
#         for batch in train_loader:
#             logger.info(f"✓ Test batch loaded: input shape {batch['input_ids'].shape}, labels shape {batch['labels'].shape}")
#             break
#     except Exception as e:
#         logger.error(f"Error during testing: {e}", exc_info=True)
#         raise





# import os
# import json
# import logging
# import random
# import re
# from typing import Dict, List, Tuple
# from transformers import T5TokenizerFast
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
# from tqdm import tqdm

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
# )
# logger = logging.getLogger("LongT5DataLoader")

# class LongT5DataLoader:

#     @staticmethod
#     def seed_worker(worker_id):
#         worker_seed = torch.initial_seed() % 2**32
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)

#     def __init__(self, config_path: str = "/home/gadde/Thesis/configs/config_long_t5.json", remove_structural_tokens: bool = True):
#         logger.info(f"Initializing LongT5 DataLoader with config: {config_path}")
#         self.config = self._load_config(config_path)
#         self.max_input_length = self.config['model']['max_input_tokens']
#         self.max_output_length = self.config['model']['max_output_tokens']
#         self.remove_structural_tokens = remove_structural_tokens

#         self.tokenizer = None
#         self._validate_paths()
#         self.load_tokenizer()

#         if self.remove_structural_tokens:
#             logger.warning("⚠️ Structural tokens will be removed from inputs and outputs (ablation test enabled).")
#         else:
#             logger.info("✅ Structural tokens will be retained.")

#     def _load_config(self, config_path: str) -> Dict:
#         with open(config_path, 'r') as f:
#             return json.load(f)

#     def _validate_paths(self) -> None:
#         paths = {
#             "Model path": self.config["model"]["path"],
#             "Tokenizer path": self.config["model"]["tokenizer_path"]
#         }
#         for name, path in paths.items():
#             if not os.path.exists(path):
#                 logger.error(f"❌ {name} not found: {path}")
#             else:
#                 logger.info(f"✓ {name} exists: {path}")

#     def load_tokenizer(self) -> None:
#         #self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["tokenizer_path"])
#         self.tokenizer = T5TokenizerFast.from_pretrained(self.config["model"]["path"])
#         #self.tokenizer.add_tokens(['[SEC]', '[SUBSEC]', '[SUBSUBSEC]', '[/SEC]', '[/SUBSEC]', '[/SUBSUBSEC]'])
#         # ✅ Add logs here for debugging tokenizer
#         logger.info(f"Tokenizer loaded from: {self.config['model']['path']}")
#         logger.info(f"Special tokens in tokenizer vocab: {self.tokenizer.additional_special_tokens}")
#         logger.info(f"Tokenizer size: {len(self.tokenizer)}")
#         if self.remove_structural_tokens:
#             assert all(tok not in self.tokenizer.get_vocab() for tok in ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]), \
#                 "⚠️ Structural tokens still present in tokenizer during ablation!"

#     def flatten_srs_to_text(self, srs_dict) -> str:
#         def flatten(node):
#             text = ''
#             if isinstance(node, dict):
#                 for key, value in node.items():
#                     if key.startswith("1."):
#                         text += f"[SEC] {key}\n"
#                     elif re.match(r"\d+\.\d+$", key):
#                         text += f"[SUBSEC] {key}\n"
#                     elif re.match(r"\d+\.\d+\.\d+$", key):
#                         text += f"[SUBSUBSEC] {key}\n"
#                     else:
#                         text += f"{key}\n"
#                     text += flatten(value)
#             elif isinstance(node, list):
#                 text += '\n'.join(str(item) for item in node) + '\n'
#             else:
#                 text += str(node) + '\n'
#             return text
#         return flatten(srs_dict)

#     def strip_structural_tokens(self, text: str) -> str:
#         return re.sub(r"\[(\/?SUBSUBSEC|\/?SUBSEC|\/?SEC)\]", "", text)

#     def process_file(self, file_info: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         input_path = os.path.join("/home/gadde/Thesis", file_info['input'])
#         output_path = os.path.join("/home/gadde/Thesis", file_info['output'])

#         with open(input_path, 'r') as f:
#             input_text = self.flatten_srs_to_text(json.load(f))
#         with open(output_path, 'r') as f:
#             output_text = self.flatten_srs_to_text(json.load(f))

#         if self.remove_structural_tokens:
#             input_text = self.strip_structural_tokens(input_text)
#             output_text = self.strip_structural_tokens(output_text)

#         input_encoding = self.tokenizer(
#             input_text,
#             max_length=self.max_input_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
#         output_encoding = self.tokenizer(
#             output_text,
#             max_length=self.max_output_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )

#         return (
#             input_encoding["input_ids"],
#             input_encoding["attention_mask"],
#             output_encoding["input_ids"]
#         )

#     def __getitem__(self, idx):
#         if not hasattr(self, 'processed_items'):
#             self.processed_items = self.load_dataset()
#         item = self.processed_items[idx]
#         return {
#             'input_ids': item[0][:, :self.max_input_length],
#             'attention_mask': item[1][:, :self.max_input_length],
#             'labels': item[2][:, :self.max_output_length]
#         }

#     def load_dataset(self, split: str = "train") -> List[Tuple]:
#         if not self.tokenizer:
#             self.load_tokenizer()
#         mapping_path = self.config["data"][f"{split}_mapping"]
#         with open(mapping_path, 'r') as f:
#             file_mapping = json.load(f)

#         processed_items = []
#         skipped = 0
#         for file_info in tqdm(file_mapping, desc=f"Loading {split} dataset"):
#             result = self.process_file(file_info)
#             if result:
#                 processed_items.append(result)
#             else:
#                 skipped += 1
#         return processed_items

#     @staticmethod
#     def collate_fn(batch):
#         input_ids, attention_masks, labels = zip(*batch)
#         return {
#             'input_ids': torch.cat(input_ids, dim=0),
#             'attention_mask': torch.cat(attention_masks, dim=0),
#             'labels': torch.cat(labels, dim=0)
#         }

#     def get_dataloader(self, split: str = "train") -> DataLoader:
#         dataset = self.load_dataset(split)

#         if split == "train":
#             seed = self.config.get("seed", 42)
#             generator = torch.Generator()
#             generator.manual_seed(seed)
#             return DataLoader(
#                 dataset,
#                 batch_size=self.config["training"]["batch_size"],
#                 shuffle=True,
#                 collate_fn=self.collate_fn,
#                 worker_init_fn=self.seed_worker,
#                 generator=generator
#             )
#         else:
#             return DataLoader(
#                 dataset,
#                 batch_size=self.config["training"]["batch_size"],
#                 shuffle=False,
#                 collate_fn=self.collate_fn
#             )

# if __name__ == "__main__":
#     try:
#         logger.info("✅ Debug: Initializing and testing LongT5DataLoader")
#         loader = LongT5DataLoader(remove_structural_tokens=True)  # 👈 Set this to False to disable ablation
#         train_loader = loader.get_dataloader("train")
#         for batch in train_loader:
#             logger.info(f"✓ Test batch loaded: input shape {batch['input_ids'].shape}, labels shape {batch['labels'].shape}")
#             break
#     except Exception as e:
#         logger.error(f"Error during testing: {e}", exc_info=True)
#         raise


import os
import json
import logging
import random
import re
from typing import Dict, List, Tuple
from transformers import T5TokenizerFast
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger("LongT5DataLoader")

class LongT5DataLoader:

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def __init__(self, config_path: str = "/home/gadde/Thesis/configs/config_long_t5.json", remove_structural_tokens: bool = True):
        logger.info(f"Initializing LongT5 DataLoader with config: {config_path}")
        self.config = self._load_config(config_path)
        self.max_input_length = self.config['model']['max_input_tokens']
        self.max_output_length = self.config['model']['max_output_tokens']
        self.remove_structural_tokens = remove_structural_tokens

        self.tokenizer = None
        self._validate_paths()
        self.load_tokenizer()

        if self.remove_structural_tokens:
            logger.warning("⚠️ Structural tokens will be removed from inputs and outputs (ablation test enabled).")
        else:
            logger.info("✅ Structural tokens will be retained.")

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
        #self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["tokenizer_path"])
        self.tokenizer = T5TokenizerFast.from_pretrained(self.config["model"]["path"])
        #self.tokenizer.add_tokens(['[SEC]', '[SUBSEC]', '[SUBSUBSEC]', '[/SEC]', '[/SUBSEC]', '[/SUBSUBSEC]'])
        # ✅ Add logs here for debugging tokenizer
        logger.info(f"Tokenizer loaded from: {self.config['model']['path']}")
        logger.info(f"Special tokens in tokenizer vocab: {self.tokenizer.additional_special_tokens}")
        logger.info(f"Tokenizer size: {len(self.tokenizer)}")
        if self.remove_structural_tokens:
            assert all(tok not in self.tokenizer.get_vocab() for tok in ["[SEC]", "[SUBSEC]", "[SUBSUBSEC]"]), \
                "⚠️ Structural tokens still present in tokenizer during ablation!"

    def flatten_srs_to_text(self, srs_dict) -> str:
        def flatten(node):
            text = ''
            if isinstance(node, dict):
                for key, value in node.items():
                    if key.startswith("1."):
                        text += f"[SEC] {key}\n"
                    elif re.match(r"\d+\.\d+$", key):
                        text += f"[SUBSEC] {key}\n"
                    elif re.match(r"\d+\.\d+\.\d+$", key):
                        text += f"[SUBSUBSEC] {key}\n"
                    else:
                        text += f"{key}\n"
                    text += flatten(value)
            elif isinstance(node, list):
                text += '\n'.join(str(item) for item in node) + '\n'
            else:
                text += str(node) + '\n'
            return text
        return flatten(srs_dict)

    def strip_structural_tokens(self, text: str) -> str:
        return re.sub(r"\[(\/?SUBSUBSEC|\/?SUBSEC|\/?SEC)\]", "", text)

    def process_file(self, file_info: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_path = os.path.join("/home/gadde/Thesis", file_info['input'])
        output_path = os.path.join("/home/gadde/Thesis", file_info['output'])

        with open(input_path, 'r') as f:
            input_text = self.flatten_srs_to_text(json.load(f))
        with open(output_path, 'r') as f:
            output_text = self.flatten_srs_to_text(json.load(f))

        if self.remove_structural_tokens:
            input_text = self.strip_structural_tokens(input_text)
            output_text = self.strip_structural_tokens(output_text)

            # 🔍 Log for debugging
            logger.debug("Stripped input preview:\n" + input_text[:300])
            logger.debug("Stripped output preview:\n" + output_text[:300])

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

        labels = output_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100    # Ignore padding tokens in loss
    


        return (
            input_encoding["input_ids"],
            input_encoding["attention_mask"],
            #output_encoding["input_ids"]
            labels  # 👈 Return the fixed labels
        )

    def __getitem__(self, idx):
        if not hasattr(self, 'processed_items'):
            self.processed_items = self.load_dataset()
        item = self.processed_items[idx]
        # ⛳ Optional debug check
        assert item[2].shape[1] == self.max_output_length, f"Label length mismatch: {item[2].shape[1]}"
        
        return {
            'input_ids': item[0][:, :self.max_input_length],
            'attention_mask': item[1][:, :self.max_input_length],
            'labels': item[2][:, :self.max_output_length]
        }

    def load_dataset(self, split: str = "train") -> List[Tuple]:
        if not self.tokenizer:
            self.load_tokenizer()
        mapping_path = self.config["data"][f"{split}_mapping"]
        with open(mapping_path, 'r') as f:
            file_mapping = json.load(f)

        processed_items = []
        skipped = 0
        for file_info in tqdm(file_mapping, desc=f"Loading {split} dataset"):
            result = self.process_file(file_info)
            if result:
                processed_items.append(result)
            else:
                skipped += 1
        return processed_items

    @staticmethod
    def collate_fn(batch):
        input_ids, attention_masks, labels = zip(*batch)
        return {
            'input_ids': torch.cat(input_ids, dim=0),
            'attention_mask': torch.cat(attention_masks, dim=0),
            'labels': torch.cat(labels, dim=0)
        }

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
                worker_init_fn=self.seed_worker,
                generator=generator
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                collate_fn=self.collate_fn
            )

if __name__ == "__main__":
    try:
        logger.info("✅ Debug: Initializing and testing LongT5DataLoader")
        loader = LongT5DataLoader(remove_structural_tokens=True)  # 👈 Set this to False to disable ablation
        train_loader = loader.get_dataloader("train")
        for batch in train_loader:
            logger.info(f"✓ Test batch loaded: input shape {batch['input_ids'].shape}, labels shape {batch['labels'].shape}")
            break
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        raise