import pytest 
from tests.utils import build_clt_training_runner_cfg
from clt.config import CLTTrainingRunnerConfig, CLTConfig
from clt.clt import CLT
from pathlib import Path
import os 
import torch 
from pydantic import ValidationError

C_l0_COEF = 4

current_file = Path(__file__).resolve()
project_root = current_file.parent

@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": str(project_root / "data/NeelNanda_c4_10k_tokenized"),
            "cross_layer_decoders": True,
            "disk": True
        }, 
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": str(project_root / "data/NeelNanda_c4_10k_tokenized"),
            "cross_layer_decoders": False,
            "disk": True
        }
    ]
)
def cfg(request: pytest.FixtureRequest):
    """
    Pytest fixture to create a mock instance of CLTTrainingRunnerConfig.
    """
    params = request.param
    return build_clt_training_runner_cfg(**params)

@pytest.fixture(params=[1])
def clt(cfg: CLTTrainingRunnerConfig, request):
    n_layers = request.param
    clt_config = cfg.create_sub_config(
        CLTConfig, 
        n_layers=n_layers
    )
    clt = CLT(clt_config)
    return clt

def test_clt_init(cfg: CLTTrainingRunnerConfig): 
    n_layers = 1  # normally inferred in the training runner
    clt_config = cfg.create_sub_config(
        CLTConfig, 
        n_layers=n_layers
    )
    clt = CLT(clt_config)

    assert clt.cfg == clt_config

    if cfg.cross_layer_decoders: 
        N_dec = n_layers * (n_layers + 1) // 2
        assert clt.W_enc.shape == (n_layers, cfg.d_in, cfg.d_latent)
        assert clt.W_dec.shape == (N_dec, cfg.d_latent, cfg.d_in)
        assert clt.b_enc.shape == (n_layers, cfg.d_latent,)
        assert clt.b_dec.shape == (N_dec, cfg.d_in,)
    else : 
        assert clt.W_enc.shape == (n_layers, cfg.d_in, cfg.d_latent)
        assert clt.W_dec.shape == (n_layers, cfg.d_latent, cfg.d_in)
        assert clt.b_enc.shape == (n_layers, cfg.d_latent,)
        assert clt.b_dec.shape == (n_layers, cfg.d_in,)

def test_encode(clt: CLT): 
    acts_in = torch.randn(10, clt.cfg.n_layers, clt.cfg.d_in, device = clt.cfg.device)
    z, _ = clt.encode(acts_in)

    assert z.shape == (10, clt.cfg.n_layers, clt.cfg.d_latent)

def test_forward(clt: CLT): 
    acts_in = torch.randn(10, clt.cfg.n_layers, clt.cfg.d_in, device = clt.cfg.device)

    z, _ = clt.encode(acts_in)
    acts_pred = clt.decode(z)
    acts_out = clt.forward_eval(acts_in)

    assert torch.allclose(acts_out, acts_pred)
    acts_out_loop = torch.zeros_like(acts_in)

    if clt.cfg.cross_layer_decoders:
        for layer in range(clt.N_layers):         
            thr   = torch.exp(clt.log_threshold[layer])
            z_l   = acts_in[:, layer] @ clt.W_enc[layer] + clt.b_enc[layer]
            z_l   = z_l * (z_l > thr)
            idx_previous = 0
            for k in range(layer, clt.N_layers):
                idx = ((clt.l_idx == layer) & (clt.k_idx == k)).nonzero(as_tuple=True)[0].item()
                assert idx >= idx_previous
                acts_out_loop[:, k] += z_l @ clt.W_dec[idx] + clt.b_dec[idx]
                idx_previous = idx
    else:
        for layer in range(clt.N_layers):         
            thr   = torch.exp(clt.log_threshold[layer])
            z_l   = acts_in[:, layer] @ clt.W_enc[layer] + clt.b_enc[layer]
            z_l= z_l * (z_l > thr)
            acts_out_loop[:, layer] += z_l @ clt.W_dec[layer] + clt.b_dec[layer]

    assert torch.allclose(acts_pred, acts_out_loop)
    assert torch.allclose(acts_out, acts_out_loop, rtol=1e-5, atol=1e-6)

def test_loss(clt: CLT): 
    acts_in = torch.randn(10, clt.cfg.n_layers, clt.cfg.d_in, device = clt.cfg.device)
    acts_out = torch.randn(10, clt.cfg.n_layers, clt.cfg.d_in, device = clt.cfg.device)
    loss_metrics = clt.loss(acts_in, acts_out, clt.cfg.l0_coefficient, df_coef = 0)

    acts_out_loop = torch.zeros_like(acts_in, device=clt.cfg.device)
    feature_hidden = torch.zeros(10, clt.cfg.n_layers, clt.cfg.d_latent, device = clt.cfg.device)
    feature_acts = torch.zeros(10, clt.cfg.n_layers, clt.cfg.d_latent, device = clt.cfg.device)
    l0_loss_accross_layers = torch.zeros(clt.cfg.n_layers, device=feature_acts.device)

    if clt.cfg.cross_layer_decoders:
        for layer in range(clt.N_layers):         
            thr = torch.exp(clt.log_threshold[layer])
            feature_hidden[:, layer, :] = acts_in[:, layer, :] @ clt.W_enc[layer] + clt.b_enc[layer]
            feature_acts[:, layer, :] = feature_hidden[:, layer, :] * (feature_hidden[:, layer, :] > thr)

            for k in range(layer, clt.N_layers):
                idx = ((clt.l_idx == layer) & (clt.k_idx == k)).nonzero(as_tuple=True)[0].item()
                acts_out_loop[:, k] += feature_acts[:, layer, :] @ clt.W_dec[idx] + clt.b_dec[idx]

        # Compute feature norms using loops
        feature_norms = torch.zeros(clt.cfg.n_layers, clt.cfg.d_latent, device=feature_acts.device)
        for layer in range(clt.cfg.n_layers):
            for feat in range(clt.cfg.d_latent):
                norm_sum = 0
                for dec_idx in range(clt.N_dec):
                    if clt.l_idx[dec_idx] == layer:
                        norm_sum += (clt.W_dec[dec_idx, feat, :] ** 2).sum()
                feature_norms[layer, feat] = torch.sqrt(norm_sum)
    else:
        for layer in range(clt.N_layers):         
            thr   = torch.exp(clt.log_threshold[layer])
            feature_hidden[:, layer, :]   = acts_in[:, layer, :] @ clt.W_enc[layer] + clt.b_enc[layer]
            feature_acts[:, layer, :] = feature_hidden[:, layer, :] * (feature_hidden[:, layer, :] > thr)
            acts_out_loop[:, layer] += feature_acts[:, layer, :] @ clt.W_dec[layer] + clt.b_dec[layer]

        # Simple case: feature_norms = clt.W_dec.norm(dim=2)
        feature_norms = torch.zeros(clt.cfg.n_layers, clt.cfg.d_latent, device=feature_acts.device)
        for layer in range(clt.cfg.n_layers):
            for feat in range(clt.cfg.d_latent):
                feature_norms[layer, feat] = clt.W_dec[layer, feat, :].norm()

    mse_loss_tensor = torch.nn.functional.mse_loss(acts_out, acts_out_loop, reduction="none")
    mse_loss_accross_layers = mse_loss_tensor.sum(dim=-1).mean(dim=0)
    mse_loss = mse_loss_accross_layers.sum()

    # Compute weighted activations and l0 loss using loops
    for layer in range(clt.cfg.n_layers):
        layer_l0 = 0
        for batch in range(feature_acts.shape[0]):
            for feat in range(feature_acts.shape[2]):
                weighted_act = feature_acts[batch, layer, feat] * feature_norms[layer, feat]
                tanh_weighted_act = torch.tanh(C_l0_COEF * weighted_act)
                layer_l0 += tanh_weighted_act
        layer_l0 = layer_l0 / feature_acts.shape[0] # should do mean over batch
        l0_loss_accross_layers[layer] = clt.cfg.l0_coefficient * layer_l0

    l0_loss = l0_loss_accross_layers.sum()

    assert torch.allclose(loss_metrics.act_in, acts_in)
    assert torch.allclose(loss_metrics.act_out, acts_out)
    assert torch.allclose(loss_metrics.feature_acts, feature_acts, rtol=1e-5, atol=1e-6)
    assert torch.allclose(loss_metrics.hidden_pre, feature_hidden, rtol=1e-5, atol=1e-6)
    assert torch.allclose(loss_metrics.mse_loss, mse_loss, rtol=1e-5, atol=1e-6)
    assert torch.allclose(loss_metrics.l0_loss, l0_loss, rtol=1e-5, atol=1e-6)
    assert torch.allclose(loss_metrics.mse_loss_accross_layers, mse_loss_accross_layers, rtol=1e-5, atol=1e-6)
    assert torch.allclose(loss_metrics.l0_loss_accross_layers, l0_loss_accross_layers, rtol=1e-5, atol=1e-6)

def test_clt_save_and_load_from_pretrained(clt: CLT, tmp_path: Path) -> None:
    clt_state_dict = clt.state_dict()

    model_path = str(tmp_path)
    assert os.path.exists(model_path)
    clt.save_model(model_path)

    try:
        clt_loaded = clt.load_from_pretrained(model_path, device="cpu")
    except ValidationError as e:
        print(f"Validation error details: {e}")
        print(f"Error dict: {e.errors()}")
        raise

    clt_loaded_state_dict = clt_loaded.state_dict()

    # check state_dict matches the original
    for key in clt_state_dict:
        assert torch.allclose(
            clt_state_dict[key],
            clt_loaded_state_dict[key],
        )

    acts_in = torch.randn(10, clt.cfg.n_layers, clt.cfg.d_in, device=clt.cfg.device)
    acts_out_1 = clt.forward_eval(acts_in)
    acts_out_2 = clt_loaded.forward_eval(acts_in)
    assert torch.allclose(acts_out_1, acts_out_2)

def test_set_decoder_norm_to_unit_norm(clt: CLT):
    with torch.no_grad():
        clt.set_decoder_norm_to_unit_norm()
        norm_after = torch.norm(clt.W_dec.data, dim=2) 
        assert torch.allclose(norm_after, torch.ones_like(norm_after), rtol=1e-4, atol=1e-6)