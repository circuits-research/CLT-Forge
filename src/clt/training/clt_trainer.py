from typing import Dict, Callable, Optional
import wandb
import logging
import torch
from torch.amp import GradScaler, autocast
from torch.optim import Adam

from clt.clt import CLT
from clt.training.activations_store import ActivationsStore
from clt.training.optim import LearningRateScheduler
from clt.clt import LossMetrics
from clt.config import CLTTrainingRunnerConfig
from clt import logger
import torch.distributed as dist

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
            cfg.lr_warm_up_type,
            cfg.lr,
            cfg.total_training_steps,
            cfg.lr_warm_up_steps, 
            lr_decay_steps = cfg.lr_decay_steps,
            final_lr_scale = cfg.final_lr_scale, 
            decay_stable = cfg.decay_stable_steps
        )

        self.l0_scheduler = LearningRateScheduler(
            cfg.l0_warm_up_type,
            cfg.l0_coefficient,
            cfg.total_training_steps,
            cfg.l0_warm_up_steps,
            lr_waiting_steps = cfg.l0_waiting_steps
        )

        # if self.cfg.functional_loss is not None:
        #     # for mypy
        #     if (
        #         cfg.fc_warm_up_type is None or
        #         cfg.fc_coefficient is None or
        #         cfg.total_training_steps is None or
        #         cfg.fc_warm_up_steps is None or
        #         cfg.fc_waiting_steps is None
        #     ):
        #         raise ValueError(
        #             "All functional loss scheduler parameters (fc_warm_up_type, fc_coefficient, "
        #             "total_training_steps, fc_warm_up_steps, fc_waiting_steps) must be set when functional_loss is enabled."
        #         )
            
        #     self.fc_scheduler = LearningRateScheduler(
        #         cfg.fc_warm_up_type,
        #         cfg.fc_coefficient,
        #         cfg.total_training_steps,
        #         cfg.fc_warm_up_steps,
        #         lr_waiting_steps=cfg.fc_waiting_steps
        #     )
            
        self.optimizer = Adam(
            self.clt.parameters(),
            lr=cfg.lr,
            betas=(
                cfg.adam_beta1,
                cfg.adam_beta2,
            ),
        )

        if self.cfg.use_mixed_precision and self.cfg.device != "cpu":
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None

        self.n_tokens: int = 0
        self.monitoring_l0 = None


    def _initialize_b_enc(self, n_batches: int = 10): 

        # Gather data for all modes
        x = []
        for _ in range(n_batches):
            # All ranks must call the iterator to consume data synchronously
            acts_in, _ = (t.to(self.cfg.device) for t in next(self.activations_store.__iter__()))
            
            with torch.no_grad():
                if self.cfg.ddp or self.cfg.fsdp:
                    # Note: Using .module is correct for DDP/FSDP to access the underlying parameters
                    hidden_pre = torch.einsum("bnd,ndk->bnk", acts_in, self.clt.module.W_enc).detach().to("cpu")
                else:
                    # Accessing unsharded self.clt.W_enc is correct for Feature Sharding
                    hidden_pre = torch.einsum("bnd,ndk->bnk", acts_in, self.clt.W_enc).detach().to("cpu")
            
            x.append(hidden_pre)
        x = torch.cat(x, dim=0)
                    
        if self.cfg.ddp:
            if self.is_main_process:
                self.clt.module._initialize_b_enc(x) 
                
            torch.distributed.barrier()
            torch.distributed.broadcast(self.clt.module.b_enc.data, src=0)
            
        elif self.cfg.fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            # Use FSDP context to access full parameters
            with FSDP.summon_full_params(self.clt):
                if self.is_main_process:
                    self.clt.module._initialize_b_enc(x)

            torch.distributed.barrier()
            
        else:  
            self.clt._initialize_b_enc(x)
            
            if self.world_size > 1:
                torch.distributed.barrier()



    def _synchronize_feature_sharding_gradients(self):
        """Manually performs all_reduce(AVG) on non-sharded parameters (b_enc, b_dec) in Feature Sharding mode."""
        
        # Only run in Feature Sharding mode (world_size > 1, ddp=False, fsdp=False)
        if self.world_size <= 1 or self.cfg.ddp or self.cfg.fsdp:
            return

        # Get references to the non-sharded parameters
        b_enc_param = self.clt.b_enc
        b_dec_param = self.clt.b_dec
        
        # # 1. Synchronize b_enc gradient (replicated parameter)
        # if b_enc_param.grad is not None:
        #     dist.all_reduce(b_enc_param.grad.data, op=dist.ReduceOp.AVG)
            
        # 2. Synchronize b_dec gradient (replicated parameter)
        if b_dec_param.grad is not None:
            dist.all_reduce(b_dec_param.grad.data, op=dist.ReduceOp.AVG)
            
        dist.barrier()

    def fit(self): 
            """ fit a clt """
            
            # start_func_finetuning = True
            if self.cfg.from_pretrained_path is None:
                self._initialize_b_enc()
                
            print(f"GPU {self.rank} - b_enc mean: {self.clt.b_enc.mean().item():.4f}, b_enc sum: {self.clt.b_enc.sum().item():.4f}")
            
            while self.n_tokens < self.cfg.total_training_tokens: 

                # get next batch
                *tokens_part, acts_in, acts_out = (
                    t.to(device=self.cfg.device) 
                    for t in next(self.activations_store.__iter__())
                )     

                # In Feature Sharding, all GPUs must process the exact same input batch.
                if self.world_size > 1:
                    # We sample the first value of the input activations (acts_in[0, 0, 0]).
                    local_val = acts_in[0, 0, 0].detach().clone() 

                    # Prepare list to gather values from all ranks
                    gathered_vals = [torch.zeros_like(local_val) for _ in range(self.world_size)]
                    
                    # Use all_gather to collect the reference value from every GPU
                    dist.all_gather(gathered_vals, local_val)

                    # Compare local value against Rank 0's reference value
                    ref_val = gathered_vals[0]
                    if not torch.isclose(local_val, ref_val, rtol=1e-5):
                        err_msg = (f"CRITICAL DATA DESYNC AT STEP {self.n_training_steps}!\n"
                                f"Rank {self.rank} input: {local_val.item():.6f}\n"
                                f"Rank 0 input: {ref_val.item():.6f}\n"
                                f"The GPUs are processing different data batches. Check ActivationsStore barriers.")
                        logger.error(err_msg)
                        raise RuntimeError(err_msg)   

                loss_metrics = self._compute_training_step_loss(
                    acts_in, 
                    acts_out, 
                    tokens_part[0] if len(tokens_part) > 0 else None
                )

                self.n_tokens += self.cfg.train_batch_size_tokens
                self.n_training_steps += 1
                if self.is_main_process:
                    self._log_train_step(loss_metrics)
                    self._run_and_log_evals()
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

            self.save_checkpoint_fn(
                trainer=self,
                checkpoint_name=f"final_{self.n_tokens}",
            )

            return self.clt

    def _log_debug_info(self, loss_metrics: LossMetrics):
        """Log activation and gradient norms across GPUs."""
        if self.n_training_steps % 100 != 0:
            return
        
        sparsity = (loss_metrics.feature_acts == 0).float().mean().item()
        if self.rank == 0:
            print(f"Feature sparsity: {sparsity:.4f}")

        print(f"Rank {self.rank}: W_dec[0,0,0] = {self.clt.W_dec[0,0,0].item():.6f}")
        if self.clt.W_dec.grad is not None:
            print(f"Rank {self.rank}: W_dec.grad[0,0,0] = {self.clt.W_dec.grad[0,0,0].item():.6f}")
        else:
            print(f"Rank {self.rank}: W_dec.grad is None")


        feat_act = loss_metrics.feature_acts  # [B, N_layers, local_d_latent]
        
        # Calculate the SQUARED NORM LOCALLY: ||x||^2_local = sum(x^2)
        local_sq_norm_per_layer = feat_act.pow(2).sum(dim=(0, 2))  # [N_layers]
        
        if self.world_size > 1:
            
            # Gather the local SQUARED norms from all GPUs.
            all_sq_norms_list = [
                torch.zeros_like(local_sq_norm_per_layer) 
                for _ in range(self.world_size)
            ]
            dist.all_gather(all_sq_norms_list, local_sq_norm_per_layer.contiguous())
            
            # Sum the squared norms to get the global squared norm.
            global_sq_norm_per_layer = sum(all_sq_norms_list)
            
            # Take the square root to get the global L2 norm
            global_norm_per_layer = torch.sqrt(global_sq_norm_per_layer + 1e-12)
            
            # average the global norm across layers to get the final logged metric.
            final_logged_norm = global_norm_per_layer.mean().item()
            
            # Set the activation norms for the subsequent gradient logging block
            act_norms_per_layer = global_norm_per_layer
            
            if self.rank == 0:
                print(f"\nStep {self.n_training_steps}")
                print(f"Activation norms per GPU (Local Shard Norms):")
                # Print the local norms 
                for gpu_id in range(self.world_size):
                    # Calculate the L2 norm for the local shard: sqrt(sum(local_sq_norm) / N_layers)
                    local_norm_val = torch.sqrt(all_sq_norms_list[gpu_id].sum() / all_sq_norms_list[gpu_id].size(0) + 1e-12)
                    print(f"  GPU {gpu_id}: {local_norm_val.item():.4f}")
                print(f"Global Activation Norm (Synchronized): {final_logged_norm:.4f}")
        else:
            # Single GPU case (original logic)
            act_norms_per_layer = feat_act.norm(dim=(0, 2)).mean(dim=0)  # [N_layers]
            if self.rank == 0:
                print(f"\nStep {self.n_training_steps}")
                print(f"Activation norms: {act_norms_per_layer}")
        
        # Gradient norms per parameter type
        if self.cfg.ddp or self.cfg.fsdp:
            W_enc = self.clt.module.W_enc
            b_enc = self.clt.module.b_enc
            W_dec = self.clt.module.W_dec
        else:
            W_enc = self.clt.W_enc
            b_enc = self.clt.b_enc
            W_dec = self.clt.W_dec
        
        grad_norms = {
            "W_enc": W_enc.grad.norm().item() if W_enc.grad is not None else 0.0,
            "b_enc": b_enc.grad.norm().item() if b_enc.grad is not None else 0.0,
            "W_dec": W_dec.grad.norm().item() if W_dec.grad is not None else 0.0,
        }
        
        if self.world_size > 1:

            grad_norm_tensor = torch.tensor([grad_norms["W_enc"], grad_norms["b_enc"], grad_norms["W_dec"]], device=self.cfg.device)
            all_grad_norms = [torch.zeros_like(grad_norm_tensor) for _ in range(self.world_size)]
            dist.all_gather(all_grad_norms, grad_norm_tensor.contiguous())
            
            if self.rank == 0:
                print(f"Gradient norms per GPU:")
                for gpu_id in range(self.world_size):
                    norms = all_grad_norms[gpu_id]
                    print(f"  GPU {gpu_id}: W_enc={norms[0]:.4f}, b_enc={norms[1]:.4f}, W_dec={norms[2]:.4f}")
        else:
            if self.rank == 0:
                print(f"Gradient norms: W_enc={grad_norms['W_enc']:.4f}, b_enc={grad_norms['b_enc']:.4f}, W_dec={grad_norms['W_dec']:.4f}")

    def _compute_training_step_loss(self, act_in: torch.Tensor, act_out: torch.Tensor, tokens: Optional[torch.Tensor] = None) -> LossMetrics:
    
        if self.n_training_steps < 5:
            print(f"GPU {self.rank} - act_in sum: {act_in.sum().item():.4f}, shape: {act_in.shape}")

        accumulation_steps = self.cfg.gradient_accumulation_steps
        
        # Only zero grad at the start or after optimizer step
        if self.n_training_steps % accumulation_steps == 0:
            self.optimizer.zero_grad()

        if self.scaler is not None:
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                loss, loss_metrics = self.clt(act_in, act_out, self.l0_scheduler.get_lr(), df_coef=self.cfg.dead_penalty_coef)
        else:
            loss, loss_metrics = self.clt(act_in, act_out, self.l0_scheduler.get_lr(), df_coef=self.cfg.dead_penalty_coef)
        
        if self.n_training_steps == 0 and self.rank == 0:
            print(f"feat_act shape: {loss_metrics.feature_acts.shape}", flush=True)
            print(f"act_pred shape: {loss_metrics.act_pred.shape}", flush=True)
        
        if self.n_training_steps % 100 == 0 and self.world_size > 1:
            loss_tensor = loss_metrics.mse_loss.detach()
            all_losses = [torch.zeros_like(loss_tensor) for _ in range(self.world_size)]
            dist.all_gather(all_losses, loss_tensor.contiguous())
            if self.rank == 0:
                loss_str = ", ".join([f"gpu{i}: {l.item():.2f}" for i, l in enumerate(all_losses)])
                print(f"Step {self.n_training_steps} - {loss_str}", flush=True)
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps
        
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
            
            # Only step and sync every accumulation_steps
            if (self.n_training_steps + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.clt.parameters(), 1.0)
                
                # GRADIENT SYNCHRONIZATION
                if self.world_size > 1 and not (self.cfg.ddp or self.cfg.fsdp):
                    self._synchronize_feature_sharding_gradients() 
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            scaled_loss.backward()
            
            # Only step and sync every accumulation_steps
            if (self.n_training_steps + 1) % accumulation_steps == 0:
                # GRADIENT SYNCHRONIZATION
                if self.world_size > 1 and not (self.cfg.ddp or self.cfg.fsdp):
                    self._synchronize_feature_sharding_gradients()
                            
                self.optimizer.step()

        # Call schedulers EVERY step (for proper step counting)
        self.update_optimizer_lr()
        self.l0_scheduler.step()
        
        self._log_debug_info(loss_metrics)
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
            (self.n_training_steps + 1) % self.cfg.wandb_log_frequency == 0
        ):
            wandb.log( # type: ignore[attr-defined]
                self._build_train_step_log_dict(loss_metrics),
                step=self.n_tokens * self.world_size,
            )

    def _build_train_step_log_dict(self, loss_metrics: LossMetrics) -> Dict: 
        act_in = loss_metrics.act_in
        act_out = loss_metrics.act_out
        feature_acts = loss_metrics.feature_acts
        act_pred = loss_metrics.act_pred
        loss = loss_metrics.mse_loss + loss_metrics.l0_loss # TODO, need to change this

        if self.cfg.ddp or self.cfg.fsdp:
            dead_features_per_layer = self.clt.module.get_dead_features().sum(dim=1)
        else: 
            dead_features_per_layer = self.clt.get_dead_features().sum(dim=1)
        
        dead_features_average_count = dead_features_per_layer.float().mean()

        # metrics for currents acts
        l0_across_layers = (feature_acts > 0).float().sum(-1).mean(0)
        l0 = l0_across_layers.mean()
        current_learning_rate = self.optimizer.param_groups[0]["lr"]
        per_token_l2_loss = (act_out - act_pred).pow(2).sum(dim=-1) # shape 
        total_variance = (act_out - act_out.mean(0)).pow(2).sum(dim=-1) # shape 
        explained_variance_across_layers = 1 - per_token_l2_loss.mean(0) / total_variance.mean(0)
        explained_variance = explained_variance_across_layers.mean()
        normalized_mse = 1 - explained_variance

        # monitoring l0 to stop training, TODO: should be done somewhere else ?
        self.monitoring_l0 = l0.item()

        logger.info(f"MSE Loss: {loss_metrics.mse_loss:.4f}")
        logger.info(f"L0 Loss: {loss_metrics.l0_loss:.4f}")

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
            # sparsity
            "details/current_learning_rate": current_learning_rate,
            "details/current_l0_coefficient": self.l0_scheduler.get_lr(),
            # "details/current_fl_coefficient": self.fc_scheduler.get_lr(),
            "details/n_training_tokens": self.n_tokens * self.world_size,
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

    def _run_and_log_evals(self): 
        pass 


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
        
        if self.cfg.ddp or self.cfg.fsdp:
            num_layers = self.clt.module.N_layers
        else: 
            num_layers = self.clt.N_layers
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
    #     self.activations_store._rebuild_buffers()
