import math
import torch 
from typing import Any

class LearningRateScheduler:
    def __init__(
        self,
        warmup_type: str,
        base_lr: float,
        total_training_steps: int,
        warmup_steps: int,
        lr_decay_steps: int = 1,
        final_lr_scale: float = 0.0,
        lr_waiting_steps: int = 0, 
        decay_stable: int = 0 # plateau at final lr (useful for replacement score finetuning)
    ):
        assert 0 <= final_lr_scale <= 1.0, "final_lr_scale must be between 0 and 1"
        assert warmup_steps >= 0 and lr_decay_steps > 0, "warmup_steps must be ≥ 0, lr_decay_steps > 0"
        assert lr_waiting_steps + warmup_steps <= total_training_steps - lr_decay_steps - decay_stable, "warm up and waiting too long"
        assert warmup_type in ["cosine", "linear"], "warmup_type must be either 'cosine' or 'linear'"
        
        self.warmup_type = warmup_type
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.final_lr_scale = final_lr_scale
        self.total_training_steps = total_training_steps
        self.lr_waiting_steps = lr_waiting_steps
        self.decay_stable = decay_stable

        self.current_step = 1
        self.lr = 0.0

    def _compute_lr(self, step: int) -> float:
        if step < self.lr_waiting_steps:
            # Stay at zero during waiting phase
            return 0.0
        elif step < self.lr_waiting_steps + self.warmup_steps:
            # Cosine warmup from 0 to base_lr
            warmup_step = step - self.lr_waiting_steps
            if self.warmup_type == "cosine":
                return self.base_lr * 0.5 * (1 - math.cos(math.pi * warmup_step / self.warmup_steps))
            elif self.warmup_type == "linear":
                return self.base_lr * warmup_step / self.warmup_steps
            else:
                raise ValueError(f"Unknown warmup_type: {self.warmup_type}")
        elif step < self.total_training_steps - (self.lr_decay_steps + self.decay_stable):
            return self.base_lr
        elif step < self.total_training_steps - self.decay_stable:
            # Linear decay from base_lr to final_lr
            decay_progress = (step - (self.total_training_steps - self.lr_decay_steps - self.decay_stable)) / self.lr_decay_steps
            scale = 1 - (1 - self.final_lr_scale) * decay_progress
            return self.base_lr * scale
        else:
            return self.base_lr * self.final_lr_scale

    def step(self) -> float:
        self.lr = self._compute_lr(self.current_step)
        self.current_step += 1
        return self.lr

    def get_lr(self) -> float:
        return self.lr

def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)

class Step(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        threshold_grad = torch.sum(
            -(1.0 / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return None, threshold_grad, None

class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth


        # # our own interpretation of allow the gradient to pass through STE to all model parameters
        # threshold_band = threshold + bandwidth / 2
        # x_grad = ((x > threshold_band) + rectangle((x - threshold) / bandwidth) * (threshold_band / bandwidth)) * grad_output
        # I should compute the gradient using STE also for W_enc and b_enc ? 
        
        x_grad = (x > threshold) * grad_output

        threshold_grad = torch.sum(
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None
