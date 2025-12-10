from typing import Dict, Any
import torch
from pydantic import BaseModel
from pathlib import Path

DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}

CLT_WEIGHTS_FILENAME = "clt_weights.safetensors"
CLT_CFG_FILENAME = "clt_cfg.json"
# LatentCache_FILENAME = "latent_cache.safetensors"
# LatentCache_CFG_FILENAME = "latent_cache_cfg.json"
# PROMPTS_FOLDERNAME = "prompts"
# EXPLANATIONS_FOLDERNAME = "explanations"
# DICT_FOLDERNAME = "dict"

class DummyModel(BaseModel):
    cfg: Any

def activation_split_path(base_path: str | Path, context_size: int, split_idx: int, must_exist: bool = False) -> Path:
    path = Path(base_path) / f"ctx_{context_size}" / f"activations_split_{split_idx}.safetensors"
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Activation file not found at {path}")
    return path