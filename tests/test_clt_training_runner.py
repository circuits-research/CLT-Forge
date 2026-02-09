import pytest
import torch
from tests.utils import build_clt_training_runner_cfg, FakeActivationsStore
from transformer_lens.hook_points import HookedRootModule
import wandb
from pathlib import Path

from clt.config import CLTTrainingRunnerConfig, CLTConfig
from clt.clt import CLT
from sae_lens.load_model import load_model
from clt.training.clt_trainer import CLTTrainer
from clt.clt_training_runner import CLTTrainingRunner
from clt.training.activations_store import ActivationsStore
from clt.utils import CLT_WEIGHTS_FILENAME, CLT_CFG_FILENAME

N_COUNTS = 0

# Get the directory of the current file
current_file = Path(__file__).resolve()
project_root = current_file.parent

# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "roneneldan/TinyStories-33M",
            "dataset_path": str(project_root / "data/NeelNanda_c4_10k_tokenized"),
            "d_in": 768,
            "cross_layer_decoders": True,
            "disk": True
        }, 
        {
            "model_name": "roneneldan/TinyStories-33M",
            "dataset_path": str(project_root / "data/NeelNanda_c4_10k_tokenized"),
            "d_in": 768,
            "cross_layer_decoders": False,
            "disk": True
        }
    ]
)
def cfg(request: pytest.FixtureRequest):
    return build_clt_training_runner_cfg(**request.param)

@pytest.fixture()
def model(cfg: CLTTrainingRunnerConfig) -> HookedRootModule:
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

@pytest.fixture(params=["fake", "real"])
def activations_store(request, cfg, model):
    if request.param == "fake":
        x = torch.randn(cfg.train_batch_size_tokens, model.cfg.n_layers, cfg.d_in)
        y = torch.randn_like(x)
        return FakeActivationsStore(x, y)
    elif request.param == "real":
        return ActivationsStore(model, cfg)

@pytest.fixture()
def dummy_save_fn():
    counter = {"count": 0}

    def save_fn(trainer, checkpoint_name):
        counter["count"] += 1

    return save_fn, counter

def test_clt_training_runner_init(cfg: CLTTrainingRunnerConfig):
    runner = CLTTrainingRunner(cfg)

    assert isinstance(runner.clt, CLT)
    assert runner.device.type in ["cuda", "cpu"]
    assert runner.model.cfg.n_layers == runner.clt.cfg.n_layers

def test_clt_training_runner_from_pretrained(tmp_path: Path, cfg: CLTTrainingRunnerConfig):
    model_path = tmp_path / "pretrained"
    model_path.mkdir(parents=True, exist_ok=True)
    dummy_clt = CLT(cfg.create_sub_config(CLTConfig, n_layers=4))
    dummy_clt.save_model(str(model_path))

    cfg.from_pretrained_path = str(model_path)

    runner = CLTTrainingRunner(cfg)
    assert isinstance(runner.clt, CLT)
    assert runner.clt.cfg.n_layers == 4

def test_clt_training_runner_run(monkeypatch, cfg: CLTTrainingRunnerConfig):
    runner = CLTTrainingRunner(cfg)
    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)
    clt_out = runner.run()
    assert isinstance(clt_out, CLT)

# def test_clt_training_runner_save_checkpoint(monkeypatch, tmp_path: Path, cfg: CLTTrainingRunnerConfig):
#     cfg.checkpoint_path = str(tmp_path)
#     runner = CLTTrainingRunner(cfg)

#     monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)
#     trainer = CLTTrainer(
#         clt=runner.clt,
#         activations_store=runner.activations_store,
#         cfg=cfg,
#         save_checkpoint_fn=lambda *_: None,
#     )

#     checkpoint_name = "test_ckpt"
#     runner.save_checkpoint(trainer, checkpoint_name)

#     ckpt_dir = Path(cfg.checkpoint_path) / checkpoint_name
#     assert (ckpt_dir / CLT_WEIGHTS_FILENAME).exists()
#     assert (ckpt_dir / CLT_CFG_FILENAME).exists()