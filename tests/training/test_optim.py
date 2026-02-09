import torch
import pytest
from torch.autograd import gradcheck

from clt.training.optim import LearningRateScheduler, rectangle, Step, JumpReLU  # Replace with your actual module name

def test_learning_rate_scheduler_behavior():
    scheduler = LearningRateScheduler(
        warmup_type = "cosine", base_lr=1.0, total_training_steps=10, warmup_steps=3, lr_decay_steps=3, final_lr_scale=0.1, lr_waiting_steps=0
    )
    lrs = [0] + [scheduler.step() for _ in range(12)]

    assert lrs[0] == 0.0
    assert lrs[2] < scheduler.base_lr 
    assert lrs[3] == pytest.approx(scheduler.base_lr, abs=1e-5)  
    assert lrs[5] == pytest.approx(scheduler.base_lr, abs=1e-5)  
    assert lrs[9] < lrs[8]
    assert lrs[10] == pytest.approx(scheduler.base_lr * scheduler.final_lr_scale, abs=1e-5)

    # After training steps, should stay constant
    assert lrs[11] == pytest.approx(lrs[10], abs=1e-5)

def test_rectangle_function():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.49, 0.5, 1.0])
    expected = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.uint8)
    result = rectangle(x).to(torch.uint8)
    assert torch.equal(result, expected)

@pytest.mark.parametrize("func_class", [Step, JumpReLU])
def test_custom_autograd_functions_gradcheck(func_class):
    x = torch.tensor([[0.3, 0.7]], dtype=torch.float64, requires_grad=True)
    threshold = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
    bandwidth = 0.1
    assert gradcheck(func_class.apply, (x, threshold, bandwidth), eps=1e-6, atol=1e-4)

def test_step_forward_values():
    x = torch.tensor([[0.3, 0.7]])
    threshold = torch.tensor([0.5, 0.5])
    out = Step.apply(x, threshold, 0.1)
    expected = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(out, expected)


def test_jumprelu_forward_values():
    x = torch.tensor([[0.3, 0.7]])
    threshold = torch.tensor([0.5, 0.5])
    out = JumpReLU.apply(x, threshold, 0.1)
    expected = torch.tensor([[0.0, 0.7]])
    assert torch.allclose(out, expected)
