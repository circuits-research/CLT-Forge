import pytest
from tests.utils import build_clt_training_runner_cfg, FakeActivationsStore
from clt.config import CLTTrainingRunnerConfig, CLTConfig
from clt.clt import CLT
from sae_lens.load_model import load_model
import torch
from clt.training.clt_trainer import CLTTrainer
from clt.training.activations_store import ActivationsStore
from transformer_lens.hook_points import HookedRootModule
import wandb
from pathlib import Path

N_COUNTS = 0

# Get the directory of the current file
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "roneneldan/TinyStories-33M",
            "dataset_path": str(project_root / "data/NeelNanda_c4_10k_tokenized"),
            "d_in": 768,
            "cross_layer_decoders": False, 
            "disk": True,
        }, 
        {
            "model_name": "roneneldan/TinyStories-33M",
            "dataset_path": str(project_root / "data/NeelNanda_c4_10k_tokenized"),
            "d_in": 768,
            "cross_layer_decoders": True, 
            "disk": True,
        }
    ]
)
def cfg(request: pytest.FixtureRequest):
    return build_clt_training_runner_cfg(**request.param)

@pytest.fixture()
def model(cfg: CLTTrainingRunnerConfig) -> HookedRootModule:
    print("loading model")
    return load_model(
        cfg.model_class_name,
        cfg.model_name,
        device=torch.device(cfg.device),
        model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
    )

@pytest.fixture()
def clt(cfg: CLTTrainingRunnerConfig, model: HookedRootModule) -> CLT:
    n_layers = model.cfg.n_layers
    clt_config = cfg.create_sub_config(CLTConfig, n_layers=n_layers)
    return CLT(clt_config)

@pytest.fixture(params=["real"]) # could also use fack
def activations_store(request, cfg, model):
    if request.param == "fake":
        x = torch.randn(cfg.train_batch_size_tokens, model.cfg.n_layers, cfg.d_in)
        y = torch.randn_like(x)
        return FakeActivationsStore(x, y)
    elif request.param == "real":
        return ActivationsStore(model, cfg)
    print("loading model")

@pytest.fixture()
def dummy_save_fn():
    counter = {"count": 0}

    def save_fn(trainer, checkpoint_name):
        counter["count"] += 1

    return save_fn, counter

def test_clt_trainer_runs(
    cfg: CLTTrainingRunnerConfig,
    clt: CLT,
    activations_store,
    dummy_save_fn,
    monkeypatch,
):
    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)
    save_fn, counter = dummy_save_fn
    trainer = CLTTrainer(
        clt=clt,
        activations_store=activations_store,
        cfg=cfg,
        save_checkpoint_fn=save_fn,
    )

    clt_out = trainer.fit()
    assert isinstance(clt_out, CLT)

def _make_trainer(cfg, clt, activations_store, dummy_save_fn):
    save_fn, counter = dummy_save_fn
    return CLTTrainer(
        clt=clt,
        activations_store=activations_store,
        cfg=cfg,
        save_checkpoint_fn=save_fn,
    ), counter

# def test_clt_trainer_runs_with_functional_loss(
#     cfg: CLTTrainingRunnerConfig,
#     model: HookedRootModule,
#     activations_store,
#     dummy_save_fn,
#     monkeypatch,
# ):
    
#     n_layers = model.cfg.n_layers
#     cfg.functional_loss = "kl"
#     clt_config = cfg.create_sub_config(CLTConfig, n_layers=n_layers)
#     clt = CLT(clt_config)

#     monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)
#     # Enable functional loss in the config
#     cfg.functional_loss = "kl"  # or "argmax"
#     cfg.fc_coefficient = 1e-3
#     cfg.fc_warm_up_steps = 1
#     cfg.fc_waiting_steps = 0
#     save_fn, counter = dummy_save_fn
#     trainer = CLTTrainer(
#         clt=clt,
#         activations_store=activations_store,
#         cfg=cfg,
#         save_checkpoint_fn=save_fn,
#     )

#     clt_out = trainer.fit()
#     assert isinstance(clt_out, CLT)

def test_initialize_b_enc(
    cfg: CLTTrainingRunnerConfig,
    clt: CLT,
    activations_store,
    dummy_save_fn,
    monkeypatch,
):
    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)
    cfg.d_latent = 10_000
    trainer, _ = _make_trainer(cfg, clt, activations_store, dummy_save_fn)
    original_b_enc = clt.b_enc.data.clone()
    trainer._initialize_b_enc(n_batches=2_000)
    
    assert not torch.equal(original_b_enc, clt.b_enc.data), "b_enc should be modified during initialization"
    
    # Test that activation rate is reasonable after initialization
    acts_in, _ = next(activations_store.__iter__())
    acts_in = acts_in.to(cfg.device)
    
    feat_acts, _ = trainer.clt.encode(acts_in)
    activation_rate = (feat_acts > 0).float().mean(dim=0).mean().item()
    
    expected_rate = 0.1
    assert expected_rate * 0.25 < activation_rate < expected_rate * 4, f"Activation rate {activation_rate:.6f} should be reasonable, expected around {expected_rate:.6f}"

def test_single_step_updates_parameters(
    cfg,
    clt,
    activations_store,
    dummy_save_fn,
):
    trainer, _ = _make_trainer(cfg, clt, activations_store, dummy_save_fn)

    before = [p.detach().clone() for p in clt.parameters() if p.requires_grad]

    act_in, act_out = next(activations_store.__iter__())
    trainer._compute_training_step_loss(act_in, act_out)

    after = [p.detach() for p in clt.parameters() if p.requires_grad]

    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "parameters unchanged"

def test_update_optimizer_lr_syncs_scheduler(
    cfg,
    clt,
    activations_store,
    dummy_save_fn,
):
    trainer, _ = _make_trainer(cfg, clt, activations_store, dummy_save_fn)

    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    scheduled_lr = trainer.update_optimizer_lr()

    assert trainer.optimizer.param_groups[0]["lr"] == scheduled_lr
    # after *some* steps the LR should differ from the original config value
    stepped = False
    for _ in range(4):
        scheduled_lr = trainer.update_optimizer_lr()
        if scheduled_lr != initial_lr:
            stepped = True
            break
    assert stepped, "learning-rate did not change after multiple scheduler steps"

def test_l0_scheduler_progresses(
    cfg,
    clt,
    activations_store,
    dummy_save_fn,
):
    trainer, _ = _make_trainer(cfg, clt, activations_store, dummy_save_fn)

    start_val = trainer.l0_scheduler.get_lr()
    trainer.l0_scheduler.step()
    next_val = trainer.l0_scheduler.get_lr()

    assert next_val != start_val, "L₀ scheduler LR did not change after a step"

def test_build_train_step_log_dict_keys(
    cfg,
    clt,
    activations_store,
    dummy_save_fn,
):
    trainer, _ = _make_trainer(cfg, clt, activations_store, dummy_save_fn)

    act_in, act_out = next(activations_store.__iter__())
    metrics = trainer._compute_training_step_loss(act_in, act_out)
    log_dict = trainer._build_train_step_log_dict(metrics)

    # Static expected keys
    static_expected_keys = {
        "losses/overall_loss",
        "metrics/explained_variance",
        "metrics/l0",
        "metrics/dead_features",
        "details/current_learning_rate",
        "details/current_l0_coefficient",
        "details/n_training_tokens",
        "losses/l0_loss",
        "losses/raw_l0_loss",
        "losses/mse_loss",
    }

    missing_keys = [key for key in static_expected_keys if key not in log_dict]
    assert not missing_keys, f"Missing keys in log_dict: {missing_keys}"

    # Optional: type checks (just as extra validation)
    assert isinstance(log_dict["losses/overall_loss"], float), "Loss should be a float"
    assert isinstance(log_dict["details/current_learning_rate"], float), "LR should be a float"

def test_checkpoint_callback_count(
    cfg,
    clt,
    activations_store,
    dummy_save_fn,
    monkeypatch,
    capsys
):
    monkeypatch.setattr(wandb, "log", lambda *_, **__: None)
    save_fn, counter = dummy_save_fn
    with capsys.disabled():
        trainer, counter = _make_trainer(cfg, clt, activations_store, dummy_save_fn)
        trainer.fit()
    # act_in, act_out = next(activations_store.__iter__())

    assert trainer.l0_scheduler.current_step == int(cfg.total_training_tokens / cfg.train_batch_size_tokens) + 1 #current steps starts at one and do last one
    # assert torch.allclose(trainer.clt.loss(act_in, act_out, trainer.l0_scheduler.get_lr()).mse_loss, torch.tensor(0., device = cfg.device))
    assert counter["count"] == cfg.n_checkpoints