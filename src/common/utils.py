import random
import numpy as np
import torch
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding = "utf-8") as f:
        config = yaml.safe_load(f)
    return config

def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
