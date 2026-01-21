
from pydantic import BaseModel
from typing import TypeVar
from typing import Dict, Any, Optional

T = TypeVar("T", bound=BaseModel)


class CLTConfig(BaseModel):
    # -----MISC------------------------------
    device : str
    dtype: str
    seed: int
    model_name: str

    # -----CLT parameters---------------------
    d_in: int
    d_latent: int
    n_layers: int
    jumprelu_bandwidth: float
    jumprelu_init_threshold: float
    normalize_decoder: bool
    dead_feature_window: int
    cross_layer_decoders: bool
    context_size: int
    functional_loss: Optional[str] = None

    # -----Efficient Decoder Parameterization-----
    # decoder_type: "full" (default), "lora", or "linear_transform"
    # - "full": separate decoder for each (source_layer, target_layer) pair
    # - "lora": base decoder + low-rank updates for cross-layer terms
    # - "linear_transform": base decoder + learned transformations for cross-layer
    decoder_type: str = "full"
    decoder_rank: int = 64  # rank for LoRA-style low-rank updates

    # -----Sparsity---------------------------
    l0_coefficient: float

    # -----DDP--------------------------------
    ddp: bool = False
    fsdp: bool = False
    feature_sharding: bool = False

    # one‑liner to get a json‑safe dict
    def to_dict(self, *, exclude_none: bool = True,**kw) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=exclude_none)

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "CLTConfig":
        """
        counterpart to `to_dict` – parses dtype string back to torch.dtype
        """
        return cls.model_validate(cfg_dict)

    @property
    def is_distributed(self) -> bool:
        return self.ddp or self.fsdp

    @property
    def is_sharded(self) -> bool:
        return self.feature_sharding

    @property
    def uses_process_group(self) -> bool:
        return self.is_distributed or self.is_sharded
