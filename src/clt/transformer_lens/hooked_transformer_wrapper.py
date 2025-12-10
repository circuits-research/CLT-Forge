import logging
from typing import (
    List,
    Optional,
    Tuple,
    Union,
    NamedTuple,
    TypeAlias
)

import torch
import torch.nn.functional as F
from typing_extensions import Literal

import transformer_lens.utils as utils
from transformer_lens import HookedTransformer

# Note - activation cache is used with run_with_cache, past_key_value_caching is used for
# generation.
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from transformer_lens.utils import (
    USE_DEFAULT_VALUE,
)

SingleLoss: TypeAlias = torch.Tensor
LossPerToken: TypeAlias = torch.Tensor
Loss: TypeAlias = Union[SingleLoss, LossPerToken]

class Output(NamedTuple):
    """Output Named Tuple."""
    logits: torch.Tensor
    loss: Loss

def _process_input_to_residual(
    self,
    input: Union[
        str,
        List[str],
        torch.Tensor,
        torch.Tensor,
    ],
    prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
    padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
    attention_mask: Optional[torch.Tensor] = None,
    past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
) -> Tuple[
    torch.Tensor,  # residual
    Optional[torch.Tensor],  # tokens
    Optional[torch.Tensor],  # shortformer_pos_embed
    Optional[torch.Tensor],  # attention_mask
]:
    """Process input and convert to initial residual stream."""
    with utils.LocallyOverridenDefaults(
        self, prepend_bos=prepend_bos, padding_side=padding_side
    ):
        return self.input_to_embed(
            input,
            prepend_bos=prepend_bos,
            padding_side=padding_side,
            attention_mask=attention_mask,
            past_kv_cache=past_kv_cache,
        )

def _run_single_transformer_block(
    self,
    residual: torch.Tensor,
    layer_idx: int,
    shortformer_pos_embed: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
) -> torch.Tensor:
    """Run a single transformer block."""
    assert 0 <= layer_idx < self.cfg.n_layers, f"Layer index {layer_idx} out of range [0, {self.cfg.n_layers})"
    
    block = self.blocks[layer_idx]
    residual = block(
        residual,
        past_kv_cache_entry=past_kv_cache[layer_idx] if past_kv_cache is not None else None,
        shortformer_pos_embed=shortformer_pos_embed,
        attention_mask=attention_mask,
    )  # [batch, pos, d_model]
    return residual

def _run_transformer_blocks(
    self,
    residual: torch.Tensor,
    shortformer_pos_embed: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
) -> torch.Tensor:
    """Run all transformer blocks sequentially."""
    for layer_idx in range(self.cfg.n_layers):
        residual = self._run_single_transformer_block(
            residual, layer_idx, shortformer_pos_embed, attention_mask, past_kv_cache
        )
    return residual

def _residual_to_output(
    self,
    residual: torch.Tensor,
    return_type: Optional[str] = "logits",
    loss_per_token: bool = False,
    tokens: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[
    None,
    torch.Tensor,
    Loss,
    Tuple[torch.Tensor, Loss],
]:
    """Convert final residual stream to output (logits/loss)."""
    if self.cfg.normalization_type is not None:
        residual = self.ln_final(residual)  # [batch, pos, d_model]
    
    if return_type is None:
        return None
    
    logits = self.unembed(residual)  # [batch, pos, d_vocab]
    if self.cfg.output_logits_soft_cap > 0.0:
        logits = self.cfg.output_logits_soft_cap * F.tanh(
            logits / self.cfg.output_logits_soft_cap
        )
    
    if return_type == "logits":
        return logits
    else:
        assert tokens is not None, "tokens must be passed in if return_type is 'loss' or 'both'"
        loss = self.loss_fn(logits, tokens, attention_mask, per_token=loss_per_token)
        if return_type == "loss":
            return loss
        elif return_type == "both":
            return Output(logits, loss)
        else:
            logging.warning(f"Invalid return_type passed in: {return_type}")
            return None

def patch_transformer_lens():
    """Call this function to add the new methods to HookedTransformer."""
    HookedTransformer._process_input_to_residual = _process_input_to_residual
    HookedTransformer._run_single_transformer_block = _run_single_transformer_block
    HookedTransformer._run_transformer_blocks = _run_transformer_blocks
    HookedTransformer._residual_to_output = _residual_to_output
