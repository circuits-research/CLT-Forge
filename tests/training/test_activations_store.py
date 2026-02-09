import pytest
import torch
from clt.training.activations_store import ActivationsStore
from clt.config import CLTTrainingRunnerConfig
from sae_lens.load_model import load_model
from transformer_lens.hook_points import HookedRootModule
from tests.utils import build_clt_training_runner_cfg
from pathlib import Path

# Get the directory of the current file
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "gpt2-small",
            "dataset_path": str(project_root / "data/NeelNanda_c4_10k_tokenized"),
            "disk": True,
            "d_in": 768,
        }
    ]
)
def cfg(request: pytest.FixtureRequest) -> CLTTrainingRunnerConfig:
    return build_clt_training_runner_cfg(**request.param)

@pytest.fixture()
def model(cfg: CLTTrainingRunnerConfig) -> HookedRootModule:
    return load_model(
        cfg.model_class_name,
        cfg.model_name,
        device=torch.device(cfg.device),
        model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
    )

def test_init(cfg, model):
    store = ActivationsStore(model, cfg)
    assert store.N_layers == model.cfg.n_layers
    assert store.context_size == cfg.context_size
    assert store.device.type == cfg.device

def test_next_token_batch_shape(cfg, model):
    store = ActivationsStore(model, cfg)
    batch = store._next_token_batch()
    assert batch.ndim == 2
    assert batch.shape[0] == cfg.store_batch_size_prompts
    assert batch.shape[1] == cfg.context_size
    assert batch.device == torch.device(cfg.device)

def test_batchify_shape(cfg, model):
    store = ActivationsStore(model, cfg)
    raw_iter = store._iterate_raw_dataset_tokens()
    batches = list(store._batchify(raw_iter))
    for b in batches:
        assert b.ndim == 2
        assert b.shape[1] == cfg.context_size

def test_activations_output_shape(cfg, model):
    store = ActivationsStore(model, cfg)
    batch = store._next_token_batch()
    ins, outs = store._activations(batch)
    assert ins.shape == outs.shape
    BCLD = ins.shape
    assert len(BCLD) == 3  # (B * C, L, d)
    assert BCLD[1] == model.cfg.n_layers
    assert BCLD[2] == cfg.d_in

def test_fresh_activation_batches(cfg, model):
    store = ActivationsStore(model, cfg)
    x, y = store._fresh_activation_batches()
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == y.shape
    assert x.ndim == 3  # (B*C, L, d)
    assert x.shape[0] == cfg.store_batch_size_prompts * cfg.context_size * (cfg.n_batches_in_buffer // 2)
    assert x.shape[1] == model.cfg.n_layers
    assert x.shape[2] == cfg.d_in

def test_iterator_yields_valid_batches(cfg, model):
    store = ActivationsStore(model, cfg)
    batch_iter = iter(store)
    x, y = next(batch_iter)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == y.shape
    assert x.ndim == 3  # shape: [B, L, d] where B is train_batch_size_tokens
    assert x.shape[0] == cfg.train_batch_size_tokens

def test_token_column_validation(cfg, model):
    # this test assumes a valid dataset; to test failure, you’d mock the dataset
    store = ActivationsStore(model, cfg)
    first = next(store._iterate_raw_dataset_tokens())
    assert isinstance(first, torch.Tensor)
    assert first.ndim == 1

def test_rebuild_buffers_preserves_shapes(cfg, model):
    store = ActivationsStore(model, cfg)
    store._rebuild_buffers()
    assert store._storage_in.shape == store._storage_out.shape
    assert store._storage_in.ndim == 3  # shape: [N, L, d]

def test_norm_scaling_factor_application(cfg, model):
    cfg.train_batch_size_tokens = 32 * 32
    cfg.context_size = 32
    cfg.store_batch_size_prompts = 32
    cfg.n_batches_in_buffer = 32 # 32*32*32 tokens in buffer 

    store = ActivationsStore(model, cfg)
    assert store.estimated_norm_scaling_factor_in.shape[0] == store.N_layers
    assert store.estimated_norm_scaling_factor_out.shape[0] == store.N_layers

    act_batch, _ = next(iter(store))

    norm_per_layer = act_batch.norm(dim=-1).mean(dim=0)

    # Should be approximately 1.0 per layer
    assert torch.allclose(norm_per_layer, (cfg.d_in ** 0.5) * torch.ones_like(norm_per_layer), rtol=0.1, atol=0.1), \
        f"Norms not close to 1.0: {norm_per_layer}"

def test_generate_and_save(cfg, tmp_path: Path, model): 
    # generate
    store = ActivationsStore(model, cfg)
    store.generate_and_save_activations(
        path = str(tmp_path), 
        split_count = 3,
        number_of_tokens = 4_000
    )

    # load 
    cfg.cached_activations_path = tmp_path
    cfg.train_batch_size_tokens = 400
    cfg.n_train_batch_per_buffer = 2
    store = ActivationsStore(model, cfg)
    store.cached_act_in.shape 

    for _ in range(10):
        A, _, _ = store._load_buffer_from_cached(return_tokens = True)
        assert A.shape == (cfg.n_train_batch_per_buffer * cfg.train_batch_size_tokens, model.cfg.n_layers, cfg.d_in)

    ctx_dir = tmp_path / "ctx_4"
    
    expected_files = [
        ctx_dir / "activations_split_0.safetensors",
        ctx_dir / "activations_split_1.safetensors",
        ctx_dir / "activations_split_2.safetensors",
    ]

    assert ctx_dir.exists() and ctx_dir.is_dir()
    actual_files = sorted(ctx_dir.glob("*.safetensors"))
    assert len(actual_files) == 3, f"Expected 3 files, found {len(actual_files)}: {actual_files}"
    assert sorted(actual_files) == sorted(expected_files), f"File names mismatch:\nExpected: {expected_files}\nGot: {actual_files}"
