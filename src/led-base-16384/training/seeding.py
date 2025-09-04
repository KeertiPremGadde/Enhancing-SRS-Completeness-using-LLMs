import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger("Seeding")
logger.setLevel(logging.DEBUG)

def set_deterministic_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # Older versions of PyTorch don't have this

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    logger.debug(f"Worker {worker_id} seeded with {worker_seed}")

def get_torch_generator(seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return g
