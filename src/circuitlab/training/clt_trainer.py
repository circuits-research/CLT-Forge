from typing import Dict, Callable, Optional
import wandb
import logging
from contextlib import nullcontext
import torch
from torch.cuda.amp import GradScaler
from torch.optim import Adam
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

from circuitlab.utils import DTYPE_MAP
from circuitlab.clt import CLT, LossMetrics
from circuitlab.training.activations_store import ActivationsStore
from circuitlab.training.optim import LearningRateScheduler
from circuitlab.config import CLTTrainingRunnerConfig
from circuitlab import logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CLTTrainer():
    """
    * Trainer class for fitting a clt using activations from an activations_store.
    """

    def __init__(
        self,
        clt: CLT,
        activations_store: ActivationsStore,
        cfg: CLTTrainingRunnerConfig,
        save_checkpoint_fn: Callable[["CLTTrainer", str], None],
        rank: int = 0,
        world_size: int = 1
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.is_main_process = rank == 0
        self.clt = clt
        self.activations_store = activations_store
        self.cfg = cfg
        self.save_checkpoint_fn = save_checkpoint_fn
        self.n_training_steps: int = 0

        self.checkpoint_thresholds = []
        if self.cfg.n_checkpoints > 0:
            self.checkpoint_thresholds = list(
                range(
                    0,
                    cfg.total_training_tokens,
                    cfg.total_training_tokens // self.cfg.n_checkpoints,
                )
            )[1:]

        self.lr_scheduler = LearningRateScheduler(
            warmup_type = cfg.lr_warm_up_type,
            base_lr = cfg.lr,
            total_training_steps = cfg.total_training_steps,
            warmup_steps = cfg.lr_warm_up_steps,
            lr_decay_steps = cfg.lr_decay_steps,
            final_lr_scale = cfg.final_lr_scale,
            decay_stable = cfg.decay_stable_steps
        )

        self.l0_scheduler = LearningRateScheduler(
            warmup_type = cfg.l0_warm_up_type,
            base_lr = cfg.l0_coefficient,
            total_training_steps = cfg.total_training_steps,
            warmup_steps = cfg.l0_warm_up_steps,
            lr_waiting_steps = cfg.l0_waiting_steps
        )

        self.optimizer = Adam(
            self.clt.parameters(),
            lr=cfg.lr,
            betas=(
                cfg.adam_beta1,
                cfg.adam_beta2,
            ),
        )

        #PRECISION: AutoCast and GradScaler configuration
        self.dtype = DTYPE_MAP[self.cfg.dtype]
        self.device_type = torch.device(self.cfg.device).type
        self.use_autocast = (
            self.dtype in {torch.float16, torch.bfloat16}
            and self.device_type == "cuda"
        )
        self.use_scaler = (
            self.dtype == torch.float16
            and self.device_type == "cuda"
        )
        self.autocast_dtype = self.dtype if self.use_autocast else None
        self.scaler = GradScaler() if self.use_scaler else None

        self.n_tokens: int = 0
        self.monitoring_l0 = None
        self.accumulation_step: int = 0

    def _initialize_b_enc(self, n_batches: int = 2):

        model = self._get_clt()

        def get_hidden_pre(acts_in):
            # 1. Access the underlying model
            with torch.no_grad():
                # 2. Handle FSDP parameter gathering
                if self.cfg.fsdp:
                    # Under FSDP, W_enc is sharded. We summon it to perform the full matmul.
                    with FSDP.summon_full_params(self.clt, recurse=False):
                        hidden = torch.einsum("bnd,ndk->bnk", acts_in, model.W_enc).detach().to("cpu")
                else:
                    # DDP and single-GPU both have the full W_enc on each device.
                    hidden = torch.einsum("bnd,ndk->bnk", acts_in, model.W_enc).detach().to("cpu")
                    
            return hidden

        batch_iter = range(n_batches)
        if self.is_main_process:
            batch_iter = tqdm(batch_iter, desc="Initializing b_enc")
        x = []
        for _ in batch_iter:
            # consume data synchronously
            acts_in, _ = (t.to(self.cfg.device) for t in next(self.activations_store.__iter__()))
            hidden_pre = get_hidden_pre(acts_in.to(model.W_enc.dtype))
            x.append(hidden_pre)

        x = torch.cat(x, dim=0)

        if self.cfg.ddp:
            if self.is_main_process:
                self.clt.module._initialize_b_enc(x)

            torch.distributed.barrier()
            torch.distributed.broadcast(self.clt.module.b_enc.data, src=0)

        elif self.cfg.fsdp:

            # Use FSDP context to access full parameters
            with FSDP.summon_full_params(self.clt):
                if self.is_main_process:
                    self.clt.module._initialize_b_enc(x)

            torch.distributed.barrier()

        else:
            self.clt._initialize_b_enc(x)

            if self.cfg.is_sharded:
                torch.distributed.barrier()

    def _synchronize_feature_sharding_gradients(self):
        """Manually performs all_reduce(AVG) on non-sharded parameters (b_enc, b_dec) in Feature Sharding mode."""

        if not self.cfg.is_sharded:
            raise ValueError("This function should not be used if feature_sharding is False.")

        b_dec_param = self.clt.b_dec

        # if b_enc_param.grad is not None:
        #     dist.all_reduce(b_enc_param.grad.data, op=dist.ReduceOp.AVG)

        # TODO: is it necessary, if so, why ?
        if b_dec_param.grad is not None:
            dist.all_reduce(b_dec_param.grad.data, op=dist.ReduceOp.AVG)

        dist.barrier()

    def fit(self): 
        """ fit a clt """
        
        # start_func_finetuning = True
        if self.cfg.from_pretrained_path is None:
            self._initialize_b_enc()
        
        while self.n_tokens < self.cfg.total_training_tokens: 
            *tokens_part, acts_in, acts_out = (
                t.to(device=self.cfg.device) 
                for t in next(self.activations_store.__iter__())
            )

            while self.n_tokens < self.cfg.total_training_tokens:

                *tokens_part, acts_in, acts_out = (
                    t.to(device=self.cfg.device)
                    for t in next(self.activations_store.__iter__())
                )
 
                if self.cfg.debug and self.cfg.is_sharded:
                    self.check_activations_across_ranks_are_equal()

                loss_metrics = self._compute_training_step_loss(
                    acts_in,
                    acts_out,
                    tokens_part[0] if len(tokens_part) > 0 else None
                )

                self.n_tokens += self.cfg.train_batch_size_tokens

                if self.accumulation_step == 0: 
                    self._log_train_step(loss_metrics)
                    self._checkpoint_if_needed()

                # if self.cfg.functional_loss is not None and self.fc_scheduler.get_lr() > 0 and start_func_finetuning:
                #     self._enable_functional_training()
                #     start_func_finetuning = False

                if self.cfg.checkpoint_l0 is not None and self.monitoring_l0 is not None:
                    if self.monitoring_l0 < self.cfg.checkpoint_l0[0]:

                        self.save_checkpoint_fn(
                            trainer=self,
                            checkpoint_name=f"middle_{self.n_tokens}",
                        )
                        if self.is_main_process:
                            self.cfg.checkpoint_l0.pop(0)

                if self.cfg.optimal_l0 is not None and self.monitoring_l0 is not None:
                    if self.monitoring_l0 < self.cfg.optimal_l0:
                        logger.info(
                            f"Stopping training at current l0 {self.monitoring_l0}"
                        )
                        break

                del acts_in, acts_out, tokens_part
                torch.cuda.empty_cache()
                
                if self.accumulation_step == 0:
                    self.n_training_steps += 1

            self.save_checkpoint_fn(
                trainer=self,
                checkpoint_name=f"final_{self.n_tokens}",
            )

            return self.clt

    def check_activations_across_ranks_are_equal(self, acts_in: torch.Tensor):

        # Use first value to check
        local_val = acts_in[0, 0, 0].detach().clone()
        gathered_vals = [torch.zeros_like(local_val) for _ in range(self.world_size)]
        dist.all_gather(gathered_vals, local_val)

        ref_val = gathered_vals[0]
        if not torch.isclose(local_val, ref_val, rtol=1e-5):
            err_msg = (f"CRITICAL DATA DESYNC AT STEP {self.n_training_steps}!\n"
                    f"Rank {self.rank} input: {local_val.item():.6f}\n"
                    f"Rank 0 input: {ref_val.item():.6f}\n"
                    f"The GPUs are processing different data batches. Check ActivationsStore barriers.")
            logger.error(err_msg)
            raise RuntimeError(err_msg)

    def get_grad_norms(self, clt_model):

        params = {
            "W_enc": clt_model.W_enc,
            "b_enc": clt_model.b_enc,
            "W_dec": clt_model.W_dec,
        }
        
        grad_norms = {}
        for name, param in params.items():
            if param.grad is not None:
                # 1. Calculate the squared norm of the local gradient shard
                local_sq_norm = param.grad.detach().data.norm()**2
                
                if self.cfg.fsdp or self.cfg.is_distributed:
                    # 2. All-Reduce: Sum the squared norms from all GPUs
                    # This mathematically equals the norm of the full gradient vector
                    dist.all_reduce(local_sq_norm, op=dist.ReduceOp.SUM)
                    
                grad_norms[name] = torch.sqrt(local_sq_norm).item()
            else:
                grad_norms[name] = 0.0
                
        return grad_norms

    def _log_debug_info(self, loss_metrics: LossMetrics):
        """Log activation and gradient norms across GPUs."""
        if self.n_training_steps % 100 != 0:
            return

        sparsity = (loss_metrics.feature_acts == 0).float().mean().item()
        if self.rank == 0:
            logger.info(f"Feature sparsity: {sparsity:.4f}")

        clt_model = self._get_clt()
        
        # Access W_dec value
        if self.cfg.fsdp:
            with FSDP.summon_full_params(self.clt, recurse=False):
                w_dec_0_0_0 = clt_model.W_dec[0,0,0].item()
        else:
            w_dec_0_0_0 = clt_model.W_dec[0,0,0].item()
        
        logger.info(f"Rank {self.rank}: W_dec[0,0,0] = {w_dec_0_0_0:.6f}")

        feat_act = loss_metrics.feature_acts  # [B, N_layers, local_d_latent]

        local_sq_norm_per_layer = feat_act.pow(2).sum(dim=(0, 2))  # [N_layers]

        if self.cfg.uses_process_group:
            # Gather the local SQUARED norms from all GPUs.
            all_sq_norms_list = [
                torch.zeros_like(local_sq_norm_per_layer)
                for _ in range(self.world_size)
            ]
            dist.all_gather(all_sq_norms_list, local_sq_norm_per_layer.contiguous())

            global_sq_norm_per_layer = sum(all_sq_norms_list)
            global_norm_per_layer = torch.sqrt(global_sq_norm_per_layer + 1e-12)
            final_logged_norm = global_norm_per_layer.mean().item()
            act_norms_per_layer = global_norm_per_layer

            if self.rank == 0:
                logger.info(f"\nStep {self.n_training_steps}")
                logger.info("Activation norms per GPU (Local Shard Norms):")
                for gpu_id in range(self.world_size):
                    # sqrt(sum(local_sq_norm) / N_layers)
                    local_norm_val = torch.sqrt(all_sq_norms_list[gpu_id].sum() / all_sq_norms_list[gpu_id].size(0) + 1e-12)
                    logger.info(f"  GPU {gpu_id}: {local_norm_val.item():.4f}")
                logger.info(f"Global Activation Norm (Synchronized): {final_logged_norm:.4f}")
        else:
            act_norms_per_layer = feat_act.norm(dim=(0, 2)).mean(dim=0)  # [N_layers]
            if self.rank == 0:
                logger.info(f"\nStep {self.n_training_steps}")
                logger.info(f"Activation norms: {act_norms_per_layer}")

        # Gradient norms per parameter type
        grad_norms = self.get_grad_norms(clt_model)
        
        if self.cfg.uses_process_group:

            grad_norm_tensor = torch.tensor([grad_norms["W_enc"], grad_norms["b_enc"], grad_norms["W_dec"]], device=self.cfg.device)
            all_grad_norms = [torch.zeros_like(grad_norm_tensor) for _ in range(self.world_size)]
            dist.all_gather(all_grad_norms, grad_norm_tensor.contiguous())

            if self.rank == 0:
                logger.info("Gradient norms per GPU:")
                for gpu_id in range(self.world_size):
                    norms = all_grad_norms[gpu_id]
                    logger.info(f"  GPU {gpu_id}: W_enc={norms[0]:.4f}, b_enc={norms[1]:.4f}, W_dec={norms[2]:.4f}")
        else:
            if self.rank == 0:
                logger.info(f"Gradient norms: W_enc={grad_norms['W_enc']:.4f}, b_enc={grad_norms['b_enc']:.4f}, W_dec={grad_norms['W_dec']:.4f}")

    def _compute_training_step_loss(self, act_in: torch.Tensor, act_out: torch.Tensor, tokens: Optional[torch.Tensor] = None) -> LossMetrics:
       
        if self.accumulation_step == 0:
            self.optimizer.zero_grad()

        #PRECISION: AutoCast vs GradScaler backprop
        if self.use_autocast:
            with torch.autocast(
                device_type = self.device_type,
                dtype = self.autocast_dtype,
            ):
                loss, loss_metrics = self.clt(
                    act_in,
                    act_out,
                    self.l0_scheduler.get_lr(),
                    df_coef=self.cfg.dead_penalty_coef
                )
        else:
            loss, loss_metrics = self.clt(
                act_in,
                act_out,
                self.l0_scheduler.get_lr(),
                df_coef=self.cfg.dead_penalty_coef,
            )

        # Scale loss by accumulation steps
        loss = loss / self.cfg.gradient_accumulation_steps

        self.accumulation_step = (self.accumulation_step + 1) % self.cfg.gradient_accumulation_steps

        # if self.n_training_steps % 100 == 0 and self.world_size > 1:
        #     loss_tensor = loss_metrics.mse_loss.detach()
        #     all_losses = [torch.zeros_like(loss_tensor) for _ in range(self.world_size)]
        #     dist.all_gather(all_losses, loss_tensor.contiguous())
        #     if self.rank == 0:
        #         loss_str = ", ".join([f"gpu{i}: {l.item():.2f}" for i, l in enumerate(all_losses)])
        #         logger.info(f"Step {self.n_training_steps} - {loss_str}")
        
        ctx = self.clt.no_sync() if (self.cfg.is_distributed and self.accumulation_step != 0) else nullcontext()
        with ctx:
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        if self.accumulation_step == 0:

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.clt.parameters(), 1.0)

                if self.cfg.is_sharded:
                    self._synchronize_feature_sharding_gradients()

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                torch.nn.utils.clip_grad_norm_(self.clt.parameters(), 1.0)

                if self.cfg.is_sharded:
                    self._synchronize_feature_sharding_gradients()

                self.optimizer.step()

        if self.cfg.debug:
            self._log_debug_info(loss_metrics)

        if self.accumulation_step == 0:
            self.update_optimizer_lr()
            self.l0_scheduler.step()

        return loss_metrics

    def update_optimizer_lr(self) -> float:
        current_lr = self.lr_scheduler.step()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    @torch.no_grad()
    def _log_train_step(self, loss_metrics: LossMetrics):
        if (
            self.cfg.log_to_wandb and
            (self.n_training_steps) % self.cfg.wandb_log_frequency == 0
        ):
            logging_dict = self._build_train_step_log_dict(loss_metrics)
            if self.is_main_process:
                wandb.log( # type: ignore[attr-defined]
                    logging_dict,
                    step=self.n_tokens * self.world_size,
                )

    def gather_feature_activations(self, feat_act: torch.Tensor) -> torch.Tensor:
        """
        Gather feature activations across feature-sharded processes.
        Input:  [B, N_layers, local_d_latent]
        Output: [B, N_layers, d_latent]
        """

        if not self.cfg.is_sharded:
            return feat_act

        gathered = [
            torch.zeros_like(feat_act)
            for _ in range(self.world_size)
        ]

        dist.all_gather(gathered, feat_act.contiguous())

        # Concatenate along latent dimension
        full_feat_act = torch.cat(gathered, dim=-1)

        return full_feat_act

    def _build_train_step_log_dict(self, loss_metrics: LossMetrics) -> Dict:
        act_out = loss_metrics.act_out
        feature_acts = loss_metrics.feature_acts
        act_pred = loss_metrics.act_pred
        loss = loss_metrics.mse_loss + loss_metrics.l0_loss # TODO, need to change this
        clt_model = self._get_clt()
        dead_features_per_layer = clt_model.get_dead_features().sum(dim=1)

        if self.cfg.is_sharded:
            feature_acts = self.gather_feature_activations(
                feature_acts
            )

            dist.all_reduce(dead_features_per_layer, op=dist.ReduceOp.SUM)

        # metrics for currents acts
        dead_features_average_count = dead_features_per_layer.float().mean()
        l0_across_layers = (feature_acts > 0).float().sum(-1).mean(0)
        l0 = l0_across_layers.mean()
        current_learning_rate = self.optimizer.param_groups[0]["lr"]
        per_token_l2_loss = (act_out - act_pred).pow(2).sum(dim=-1) # shape
        total_variance = (act_out - act_out.mean(0)).pow(2).sum(dim=-1) # shape
        explained_variance_across_layers = 1 - per_token_l2_loss.mean(0) / total_variance.mean(0)
        explained_variance = explained_variance_across_layers.mean()
        normalized_mse = 1 - explained_variance
        n_training_tokens = self.n_tokens * self.world_size

        # monitoring l0 to stop training, TODO: should be done somewhere else ?
        self.monitoring_l0 = l0.item()

        if self.is_main_process: 
            logger.info(
                f"Step: {self.n_training_steps} | "
                f"Loss: {loss:.4f} | "
                f"EV: {explained_variance:.4f} | "
                f"L0: {l0:.2f}"
            )

        # Load the dictionary
        log_dict = {
            # losses
            "losses/overall_loss": loss.item(),
            # metrics
            "metrics/total_variance": total_variance.mean().item(),
            "metrics/explained_variance": explained_variance.item(),
            "metrics/normalized_mse": normalized_mse.item(),
            "metrics/l0": l0.item(),
            "metrics/dead_features": dead_features_average_count.item(),
            # "losses/l0_loss_replacement": loss_metrics.l0_loss_replacement.item(),
            # "metrics/next_token_per": loss_metrics.pred_per if loss_metrics.pred_per is not None else 0.0,
            "details/current_learning_rate": current_learning_rate,
            "details/current_l0_coefficient": self.l0_scheduler.get_lr(),
            # "details/current_fl_coefficient": self.fc_scheduler.get_lr(),
            "details/n_training_tokens": n_training_tokens,
        }

        for l in range(len(l0_across_layers)):
            log_dict[f"dead_features/layer_{l}"] = dead_features_per_layer[l].item()
            log_dict[f"explained_variance/layer_{l}"] = explained_variance_across_layers[l].item()
            log_dict[f"sparsity/layer_{l}"] = l0_across_layers[l].item()
            # log_dict[f"sparsity_replacement/layer_{l}"] = loss_metrics.l0_across_layers_replacement[l].mean() if loss_metrics.l0_across_layers_replacement is not None else 0.0

        # # Log individual position accuracies
        # if loss_metrics.pred_per is not None:
        #     # Log individual position accuracies
        #     context_size = len(loss_metrics.pred_per)
        #     for pos in range(context_size):
        #         log_dict[f"metrics/next_token_per_pos_{pos}"] = loss_metrics.pred_per[pos].item()

        # # Create metrics dictionary for layer-wise tracking
        # layer_metrics = {
        #     "Explained Variance": explained_variance_across_layers,
        # }

        # # Update log_dict with layer metrics history, TODO: fixing memory problems in loading these metrics
        # log_dict = self._update_layer_metrics_history(log_dict, layer_metrics)

        log_dict["losses/raw_l0_loss"] = (
            loss_metrics.l0_loss / (self.l0_scheduler.get_lr())
        )
        log_dict["losses/l0_loss"] = loss_metrics.l0_loss
        log_dict["losses/mse_loss"] = loss_metrics.mse_loss
        log_dict["losses/dead_loss"] = loss_metrics.dead_feature_loss
        # log_dict["losses/hybrid_loss"] = loss_metrics.hybrid_loss

        return log_dict

    @torch.no_grad()
    def _checkpoint_if_needed(self):
        if (
            self.checkpoint_thresholds
            and self.n_tokens > self.checkpoint_thresholds[0]
        ):
            # CRITICAL: ALL ranks must call the save function
            self.save_checkpoint_fn(
                trainer=self,
                checkpoint_name=str(self.n_tokens),
            )
            self.checkpoint_thresholds.pop(0)

    def _update_layer_metrics_history(
        self,
        log_dict: Dict,
        metrics_dict: Dict[str, torch.Tensor],
    ) -> Dict:
        
        clt_model = self._get_clt()
        num_layers = clt_model.N_layers
        current_step = self.n_tokens * self.world_size

        # Initialize history trackers if they don't exist yet
        if not hasattr(self, "history_steps"):
            self.history_steps = []
            self.history_metrics: Dict[str, list[list[float]]] = {}

        self.history_steps.append(current_step)

        for metric_name, layer_values in metrics_dict.items():
            if metric_name not in self.history_metrics:
                self.history_metrics[metric_name] = [[] for _ in range(num_layers)]

            for i in range(num_layers):
                self.history_metrics[metric_name][i].append(layer_values[i].item())

        for metric_name, history in self.history_metrics.items():
            # Convert metric name to snake_case for log key
            plot_key = f"{metric_name.lower().replace(' ', '_')}_over_time"

            log_dict[plot_key] = wandb.plot.line_series(  # type: ignore[attr-defined]
                xs=self.history_steps,
                ys=history,
                keys=[f"Layer {i}" for i in range(num_layers)],
                title=metric_name,
                xname="Tokens"
            )

        return log_dict

    # def _enable_functional_training(self):
    #     """Enable functional training by configuring activations store for token return."""

    #     if self.cfg.ddp or self.cfg.fsdp:
    #         self.clt.module.attach_model_for_replacement(
    #             self.cfg.model_class_name,
    #             self.cfg.model_name,
    #             torch.device(self.cfg.device),
    #             self.cfg.model_from_pretrained_kwargs
    #         )
    #     else: 
    #         self.clt.attach_model_for_replacement(
    #             self.cfg.model_class_name, 
    #             self.cfg.model_name, 
    #             torch.device(self.cfg.device), 
    #             self.cfg.model_from_pretrained_kwargs
    #         )

    #     self.activations_store.shuffle = False
    #     self.activations_store.return_tokens = True
    #     self.activations_store.mix_with_previous_buffer = False
    #     self.activations_store.split = self.activations_store.rank
    #     self.activations_s

    def _get_clt(self) -> CLT:
        """Get the unwrapped CLT model."""
        if self.cfg.is_distributed:
            return self.clt.module
        else:
            return self.clt
