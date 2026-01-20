import math
import torch
import torch.nn as nn
from jaxtyping import Float
from pathlib import Path
from safetensors.torch import save_file, load_file
import json
from pydantic import BaseModel, ConfigDict
from typing import Union, Optional, Dict
from transformer_lens.hook_points import HookedRootModule

from clt.config import CLTConfig
from clt.utils import DTYPE_MAP, CLT_WEIGHTS_FILENAME, CLT_CFG_FILENAME
from clt.training.optim import JumpReLU
from clt import logger
from clt.load_model import load_model
import torch.distributed as dist

C_l0_COEF = 4

class LossMetrics(BaseModel):
    act_in: torch.Tensor
    act_out: torch.Tensor
    feature_acts: torch.Tensor
    hidden_pre: torch.Tensor
    act_pred: torch.Tensor
    mse_loss: torch.Tensor 
    l0_loss: torch.Tensor
    dead_feature_loss: torch.Tensor
    mse_loss_accross_layers: torch.Tensor
    l0_loss_accross_layers: torch.Tensor
    
    # l0_loss_replacement: torch.Tensor = torch.tensor(float('-inf'))
    # l0_accross_layers_replacement: Optional[torch.Tensor] = None
    # hybrid_loss: Optional[torch.Tensor] = torch.tensor(float('-inf')) # for wandb
    # pred_per: Optional[float] = torch.zeros(32)

    model_config = ConfigDict(arbitrary_types_allowed=True)

class CLT(nn.Module):
    """
    * pytorch module for a cross layer transcoder
    * can take an LLM as attribute and compute replacement model forward pass
    """

    def __init__(self, cfg: CLTConfig, rank: int = 0, world_size: int = 1) -> None:
        super().__init__()

        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.N_layers = cfg.n_layers
        self.d_in = cfg.d_in
        self.d_latent = cfg.d_latent
        self.local_d_latent = self.d_latent // world_size
        self.dtype = DTYPE_MAP[cfg.dtype]
        self.device = torch.device(cfg.device)

        init_device = self.device if not cfg.fsdp else torch.device("cpu")

        if self.cfg.is_sharded:
            torch.manual_seed(cfg.seed + rank) # Different seed per rank
        else:
            torch.manual_seed(cfg.seed) # Same seed for DDP/FSDP

        self.N_layers_out = torch.tensor(
            [cfg.n_layers - (i + 1) for i in range(self.N_layers)],
            dtype=torch.long,
            device=self.device,
        )
        self.max_layers_out = int(self.N_layers_out.max().item())

        # All parameters are created on init_device
        self.W_enc = nn.Parameter(torch.empty(self.N_layers, self.d_in, self.local_d_latent, dtype=self.dtype, device=init_device))
        self.b_enc = nn.Parameter(torch.zeros(self.N_layers, self.local_d_latent, dtype=self.dtype, device=init_device))

        if cfg.cross_layer_decoders:
            self.N_dec = self.N_layers * (self.N_layers + 1) // 2
            self.W_dec = nn.Parameter(torch.empty(self.N_dec, self.local_d_latent, self.d_in, dtype=self.dtype, device=init_device))
            self.b_dec = nn.Parameter(torch.zeros(self.N_dec, self.d_in, dtype=self.dtype, device=init_device))

            l_idx, k_idx = torch.triu_indices(self.N_layers, self.N_layers, offset=0,
                                            device=init_device)
            self.register_buffer('l_idx', l_idx, persistent=False)   # [K]
            self.register_buffer('k_idx', k_idx, persistent=False)   # [K]

            layer_mask = torch.zeros(self.N_layers, self.N_dec, device=init_device, dtype=self.dtype)
            for layer in range(self.N_layers):
                layer_mask[layer, l_idx == layer] = 1
            self.register_buffer('layer_mask', layer_mask)

        else: 
            self.W_dec = nn.Parameter(torch.empty(self.N_layers, self.local_d_latent, self.d_in, dtype=self.dtype, device=init_device))
            self.b_dec = nn.Parameter(torch.zeros(self.N_layers, self.d_in, dtype=self.dtype, device=init_device))

        self.log_threshold = nn.Parameter(
            torch.full((self.N_layers, self.local_d_latent), math.log(cfg.jumprelu_init_threshold), dtype=self.dtype, device=init_device)
        )
        self.bandwidth = cfg.jumprelu_bandwidth

        self.register_buffer('feature_count', 
            torch.zeros(
                self.N_layers, 
                self.local_d_latent, 
                dtype=torch.long, 
                device=init_device
            )
        )

        self._initialize()

        self.register_buffer('estimated_norm_scaling_factor_in', torch.ones(self.N_layers, device=self.device))
        self.register_buffer('estimated_norm_scaling_factor_out', torch.ones(self.N_layers, device=self.device))

    def _initialize(self) -> None:
        # Anthropic guidelines
        # encoder:  U(-1/n_features,  +1/n_features)
        enc_lim = 1.0 / self.d_latent**0.5
        for W in self.W_enc:
            nn.init.uniform_(W, -enc_lim, enc_lim)

        # decoder: U(-1/(n_layers*d_model), +1/(n_layers*d_model))
        dec_lim = 1.0 / (self.N_layers * self.d_in)**0.5
        nn.init.uniform_(self.W_dec, -dec_lim, dec_lim)

    def _initialize_b_enc(self, hidden_pre: Float[torch.Tensor, "..."], rate: float = 0.3) -> None: 
        """
        Initialize b_enc by examining a subset of the data and picking a constant per feature
        such that each feature activates at a certain rate.
        x: [B, N_layers, d_latent]
        """
        with torch.no_grad():
            # # Compute pre-activations without bias
            # hidden_pre = torch.einsum(
            #     "bnd,ndk->bnk",
            #     x,
            #     self.W_enc,
            # )  # [B, N_layers, d_latent]
            
            thresh = torch.exp(self.log_threshold).detach().cpu() 
            target_activation_rate = rate
            
            # For each layer and feature, find the bias that gives target activation rate
            B = hidden_pre.shape[0]
            bias_values = torch.zeros_like(self.b_enc).detach().cpu()
            
            for layer in range(self.N_layers):
                for feature in range(self.local_d_latent):
                    feature_pre_acts = hidden_pre[:, layer, feature]  # [B]
                    sorted_acts, _ = torch.sort(feature_pre_acts, descending=True)
                    target_idx = int(target_activation_rate * B) + 1
                    threshold_value = sorted_acts[target_idx]
                    required_bias = thresh[layer, feature] - threshold_value
                    
                    bias_values[layer, feature] = required_bias
            
            self.b_enc.data = bias_values.to(self.device)
            logger.info(f"Initialized b_enc with target activation rate {target_activation_rate:.6f}")
            
            # # Verify the initialization by computing actual activation rates
            # feat_act, _ = self.encode(x)            
            # activation_rates = (feat_act > 0).bfloat16().mean(dim=0)  # [N_layers, d_latent]
            # avg_activation_rate = activation_rates.mean().item()
            
            # print(f"Actual average activation rate: {avg_activation_rate * self.d_latent:.0f}")
            # print(f"Expected ~{self.d_latent * target_activation_rate:.0f} ")

    def encode(
        self,
        x: Float[torch.Tensor, "..."],
        layer: Optional[int] = None
    ) -> tuple[
        Float[torch.Tensor, "..."],
        Float[torch.Tensor, "..."],
    ]:
        """
        x: [B, N_layers, d_in] if layer is None, else [B, d_in]
        output: tuple([B, N_layers, d_latent], [B, N_layers, d_latent]) if layer is None, else [B, d_latent]
        """

        if layer is None: 
            hidden_pre = torch.einsum(
                "bnd,ndk->bnk",
                x,
                self.W_enc,
            ) + self.b_enc

            thresh = torch.exp(self.log_threshold) #shape [N_layers, d_latent]
        else: 
            assert 0 <= layer < self.N_layers, f"Layer {layer} out of range"
            hidden_pre = x @ self.W_enc[layer] + self.b_enc[layer]
            thresh = torch.exp(self.log_threshold[layer]) 
        
        feat_act = JumpReLU.apply(hidden_pre, thresh, self.bandwidth)
        return feat_act, hidden_pre

    def decode(
        self,
        z: Float[torch.Tensor, "..."],
        layer: Optional[int] = None
    ) -> Float[torch.Tensor, "..."]:
        """
        Decode latent features z back to model activations.
        Handles both cross-layer decoders and single-layer decoders.
        Correctly aggregates bias and supports distributed training.
        
        CRITICAL: In feature sharding, after all_reduce(SUM):
        - ALL ranks have identical 'out' tensor
        - b_dec is replicated (same on all ranks)
        - ALL ranks add b_dec locally → identical result
        - No broadcast needed (keeps gradient flow clean)
        """

        if layer is None:
            if self.cfg.cross_layer_decoders:
                B = z.shape[0]
                z_sel = z.index_select(1, self.l_idx)  # select source layers

                contrib = torch.einsum('bkd,kdf->bkf', z_sel, self.W_dec)  # [B, N_dec, d_in]

                out = torch.zeros(B, self.N_layers, self.d_in, dtype=contrib.dtype, device=contrib.device)
                out = out.index_add(1, self.k_idx, contrib)

                if self.cfg.is_sharded:
                    dist.all_reduce(out, op=dist.ReduceOp.SUM)
                
                # ALL ranks add replicated bias locally
                b_contrib = torch.zeros(1, self.N_layers, self.d_in, dtype=contrib.dtype, device=contrib.device)
                b_contrib = b_contrib.index_add(1, self.k_idx, self.b_dec.to(out.dtype).unsqueeze(0))
                out = out + b_contrib

            else:
                out = torch.einsum("bnk,nkd->bnd", z, self.W_dec)  # [B, N_layers, d_in]
                
                if self.cfg.is_sharded:
                    dist.all_reduce(out, op=dist.ReduceOp.SUM)
                
                # ALL ranks add replicated bias locally
                out = out + self.b_dec.to(out.dtype).unsqueeze(0)

        else:
            # Layer-specific decode
            assert 0 <= layer < self.N_layers, f"Layer {layer} out of range"
            if self.cfg.cross_layer_decoders:
                indices = (self.l_idx == layer).nonzero(as_tuple=True)[0]
                z_layer = z.unsqueeze(1).expand(-1, len(indices), -1)
                W_dec_layer = self.W_dec[indices]
                b_dec_layer = self.b_dec[indices]
                out = torch.einsum('bkd,kdf->bkf', z_layer, W_dec_layer) + b_dec_layer
            else:
                out = z @ self.W_dec[layer] + self.b_dec[layer]

        return out


    def forward_eval(
        self,
        x: Float[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "..."]:
        """
        x: [N, ..., d_in]
        Returns: z and reconstruction
        """
        z, _ = self.encode(x)
        recon = self.decode(z)
        return recon

    def forward(
        self,
        act_in:  torch.Tensor,
        act_out: torch.Tensor,
        l0_coef: float,
        df_coef: float,
        return_metrics: bool = True
    ):
        """
        Wrapper forward function for DDP.
        """

        # renormalize decoder, should normally not be used
        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        metrics = self.loss(act_in, act_out, l0_coef, df_coef)
        loss = metrics.mse_loss + metrics.l0_loss + metrics.dead_feature_loss

        return (loss, metrics) if return_metrics else loss

    def loss(self, act_in: torch.Tensor, act_out: torch.Tensor, l0_coef: float, df_coef: float) -> LossMetrics:
        feat_act, hidden_pre = self.encode(act_in)
        act_pred = self.decode(feat_act)

        ### MSE loss
        mse_loss_tensor = torch.nn.functional.mse_loss(act_out, act_pred, reduction="none")
        mse_loss_accross_layers = mse_loss_tensor.sum(dim=-1).mean(dim=0)
        mse_loss = mse_loss_accross_layers.sum()

        ### L0 regularization
        if self.cfg.cross_layer_decoders:
            squared_norms = (self.W_dec**2).sum(dim=2)
            feature_norms_local = torch.sqrt(torch.matmul(self.layer_mask, squared_norms))
        else: 
            feature_norms_local = self.W_dec.norm(dim=2)
        
        logger.info(f"Rank {self.rank}: feat_act shape = {feat_act.shape}")
        logger.info(f"Rank {self.rank}: feature_norms_local shape = {feature_norms_local.shape}")
        logger.info(f"Rank {self.rank}: W_dec shape = {self.W_dec.shape}")
        logger.info(f"Rank {self.rank}: W_dec requires_grad = {self.W_dec.requires_grad}")
        logger.info(f"Rank {self.rank}: feature_norms_local requires_grad = {feature_norms_local.requires_grad}")
        
        # Compute L0 loss local
        weighted_activations = feat_act * feature_norms_local
        tanh_weighted_activations = torch.tanh(C_l0_COEF * weighted_activations)
        l0_loss_accross_layers = l0_coef * tanh_weighted_activations.sum(dim=-1).mean(dim=0)
        l0_loss_local = l0_loss_accross_layers.sum()
        
        # SUM losses across ranks
        if self.cfg.is_sharded:
            dist.all_reduce(l0_loss_local, op=dist.ReduceOp.SUM)
        
        l0_loss = l0_loss_local
        logger.info(f"Rank {self.rank}: W_dec.requires_grad={self.W_dec.requires_grad}, has grad={self.W_dec.grad is not None}")
        logger.info(f"Rank {self.rank}: L0 loss = {l0_loss.item():.6f}")
        logger.info(f"Rank {self.rank}: feature_norms has grad = {hasattr(feature_norms_local, 'grad') and feature_norms_local.grad is not None}")
        
        ### Dead feature penalty 
        dead_feature_loss = df_coef * torch.relu(torch.exp(self.log_threshold) - hidden_pre) * feature_norms_local
        dead_feature_loss = dead_feature_loss.sum(dim=-1).mean(dim=0).sum()

        # SUM losses across ranks
        if self.cfg.is_sharded:
            dist.all_reduce(dead_feature_loss, op=dist.ReduceOp.SUM)

        ### Dead feature count local
        with torch.no_grad():
            firing = feat_act.sum(dim=0) > 0
            self.feature_count += 1
            self.feature_count[firing] = 0

        return LossMetrics(
            act_in=act_in,
            act_out=act_out,
            feature_acts=feat_act,
            hidden_pre=hidden_pre,
            act_pred=act_pred,
            mse_loss=mse_loss,
            l0_loss=l0_loss,
            dead_feature_loss=dead_feature_loss,
            mse_loss_accross_layers=mse_loss_accross_layers,
            l0_loss_accross_layers=l0_loss_accross_layers
        )
        
    @torch.no_grad()
    def get_dead_features(self) -> torch.Tensor:
        return self.feature_count > self.cfg.dead_feature_window # [N_layers, d_latent]

    def save_model(self, path_str: str, save_cfg: bool = True, rank: Optional[int] = None, state_dict_: Optional[Dict] = None):
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)
        
        state_dict = self.state_dict()
        prefix = f"rank{rank}_" if rank is not None else ""
        # Remove any keys that start with 'model.' (the attached transformer model)
        clt_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.')}

        weights_path = path / f"{prefix}{CLT_WEIGHTS_FILENAME}"
        save_file(clt_state_dict, weights_path)

        if save_cfg: 
            cfg_dict = self.cfg.to_dict()
            cfg_path = path / CLT_CFG_FILENAME

            with open(cfg_path, "w") as f:
                json.dump(cfg_dict, f)

        return cfg_path
    
    @classmethod
    def load_from_pretrained(cls, path: Union[str, Path], device: str, is_sharded: bool = False, rank: Optional[int] = None, world_size: Optional[int] = None, model_name: Optional[str] = "gpt2") -> "CLT":
        path = Path(path)

        if is_sharded:
            if rank is None or world_size is None:
                raise ValueError("Sharded CLT requires rank and world_size")
            prefix = f"rank{rank}_"
        else:
            rank, world_size, prefix = 0, 1, ""

        cfg_path = path / CLT_CFG_FILENAME
        weights_path = path / f"{prefix}{CLT_WEIGHTS_FILENAME}"

        with cfg_path.open("r") as f:
            cfg_dict = json.load(f)

        cfg_dict["device"] = device
        cfg = CLTConfig.from_dict(cfg_dict)

        if is_sharded != cfg.is_sharded:
            raise ValueError(
                f"Sharding mismatch when loading CLT checkpoint:\n"
                f"  argument is_sharded={is_sharded}\n"
                f"  checkpoint cfg.is_sharded={cfg.is_sharded}\n"
                f"These must match."
            )
 
        clt = cls(cfg, rank=rank, world_size=world_size)
        state_dict = load_file(weights_path, device=device)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.')}
        missing, unexpected = clt.load_state_dict(state_dict, strict=False)

        if missing or unexpected:
            raise RuntimeError(f"Incompatible checkpoint.\n  missing: {missing}\n  unexpected: {unexpected}")

        clt.to(torch.device(device))
        return clt
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=2, keepdim=True)