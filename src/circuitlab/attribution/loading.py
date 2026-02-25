import torch
from pathlib import Path
from circuitlab import logger
from circuitlab.clt import CLT
from circuitlab.vendor.circuit_tracer.circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuitlab.vendor.circuit_tracer.circuit_tracer import ReplacementModel

def load_circuit_tracing_clt_from_local(
    clt_checkpoint: str,
    device: str = "cuda",
    debug: bool = False,
) -> CrossLayerTranscoder:
    """
    Creates a Circuit Tracing CLT from a local CLT checkpoint.
    """

    def log(msg):
        if debug:
            logger.debug(msg)

    path = Path(clt_checkpoint)
    log(f"Loading CLT checkpoint from: {clt_checkpoint}")

    clt = CLT.load_from_pretrained(path, device=device)

    state_dict_local = {
        k: v for k, v in clt.state_dict().items()
        if not k.startswith("model.")
    }

    W_dec = state_dict_local["W_dec"]
    b_dec = state_dict_local["b_dec"]
    W_enc = state_dict_local["W_enc"]
    b_enc = state_dict_local["b_enc"]
    log_threshold = state_dict_local["log_threshold"]
    estimated_norm_scaling_factor_in = state_dict_local["estimated_norm_scaling_factor_in"]
    estimated_norm_scaling_factor_out = state_dict_local["estimated_norm_scaling_factor_out"]

    n_layers = b_enc.shape[0]
    l_idx, k_idx = torch.triu_indices(n_layers, n_layers, offset=0, device=device)
    d_model = W_enc.shape[1]
    d_transcoder = W_enc.shape[2]
    dtype = W_dec.dtype

    feature_input_hook = "ln2.hook_normalized"
    feature_output_hook = "hook_mlp_out"

    log("\nModel dimensions:")
    log(f"  Layers: {n_layers}")
    log(f"  Model dimension: {d_model}")
    log(f"  Transcoder dimension: {d_transcoder}")
    log(f"  Data type: {dtype}")
    log(f"  W_dec dtype: {W_dec.dtype}")
    log(f"  W_enc dtype: {W_enc.dtype}")

    # 1. merge b_dec
    b_dec_circuit = torch.zeros((n_layers, b_dec.shape[1]), device=device)
    for i in range(b_dec.shape[0]):
        b_dec_circuit[k_idx[i]] += b_dec[i]

    state_dict = {"b_dec": b_dec_circuit, "b_enc": b_enc}
    state_dict["W_enc"] = W_enc.transpose(1, 2)

    # 2. split W_dec
    for i in range(n_layers):
        layer_weights = W_dec[l_idx == i]
        state_dict[f"W_dec.{i}"] = layer_weights.transpose(0, 1)

    # 3. threshold
    threshold = torch.exp(log_threshold).unsqueeze(1)
    state_dict["activation_function.threshold"] = threshold

    # 4. scaling
    log("\nApplying normalization scaling:")
    log(f"  Input scaling shape: {estimated_norm_scaling_factor_in.shape}")
    log(f"  Output scaling shape: {estimated_norm_scaling_factor_out.shape}")

    state_dict["W_enc"] *= estimated_norm_scaling_factor_in.view(-1, 1, 1)

    for i in range(n_layers):
        state_dict[f"W_dec.{i}"] /= estimated_norm_scaling_factor_out[i:].view(1, -1, 1)

    state_dict["b_dec"] /= estimated_norm_scaling_factor_out.view(-1, 1)

    instance = CrossLayerTranscoder(
        n_layers,
        d_transcoder,
        d_model,
        activation_function="jump_relu",
        lazy_decoder=False,
        lazy_encoder=False,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        dtype=dtype,
        device=device,
    )

    instance.load_state_dict(state_dict, assign=True)

    log(f"\nSuccessfully loaded CLT with {len(state_dict)} parameters")
    log(f"  Instance W_enc dtype: {instance.W_enc.dtype}")
    log(f"  Instance b_enc dtype: {instance.b_enc.dtype}")
    log(f"  Instance b_dec dtype: {instance.b_dec.dtype}")

    return instance


def test_clt_performance_on_prompt(
    inputs: str,
    clt: CrossLayerTranscoder,
    model: ReplacementModel,
    debug: bool = False,
):
    def log(msg):
        if debug:
            logger.debug(msg)

    tokens = model.ensure_tokenized(inputs) 

    mlp_in_cache, mlp_in_hooks, _ = model.get_caching_hooks(
        lambda name: model.feature_input_hook in name
    )
    mlp_out_cache, mlp_out_hooks, _ = model.get_caching_hooks(
        lambda name: model.feature_output_hook in name
    )

    _ = model.run_with_hooks(tokens, fwd_hooks=mlp_in_hooks + mlp_out_hooks)

    mlp_in_cache = torch.cat(list(mlp_in_cache.values()), dim=0)
    mlp_out_cache = torch.cat(list(mlp_out_cache.values()), dim=0)

    attribution_data = model.transcoders.compute_attribution_components(mlp_in_cache)
    error_vectors = mlp_out_cache - attribution_data["reconstruction"]

    for layer_idx in range(error_vectors.shape[0]):
        layer_errors = error_vectors[layer_idx, 1:]
        layer_outputs = mlp_out_cache[layer_idx, 1:]

        activations = attribution_data["activation_matrix"].to_dense()[layer_idx, 1:]

        l0 = (activations > 0).float().sum(1)
        log(f"Layer {layer_idx}: L0 = {l0.mean().item():.2f}")

        mse = torch.mean(layer_errors ** 2)
        variance = torch.var(layer_outputs)
        normalized_mse = mse / variance if variance > 0 else float("inf")

        log(
            f"Layer {layer_idx}: MSE={mse:.6f}, Var={variance:.6f}, Norm MSE={normalized_mse:.6f}"
        )

def compare_reconstruction_with_local_clt_class(
    clt_checkpoint: str,
    inputs: str,
    clt: CrossLayerTranscoder,
    model: ReplacementModel,
    model_name: str,
    debug: bool = False,
):
    def log(msg):
        if debug:
            logger.debug(msg)

    tokens = model.ensure_tokenized(inputs)

    mlp_in_cache, mlp_in_hooks, _ = model.get_caching_hooks(
        lambda name: model.feature_input_hook in name
    )
    mlp_out_cache, mlp_out_hooks, _ = model.get_caching_hooks(
        lambda name: model.feature_output_hook in name
    )

    _ = model.run_with_hooks(tokens, fwd_hooks=mlp_in_hooks + mlp_out_hooks)

    mlp_in_cache = torch.cat(list(mlp_in_cache.values()), dim=0)
    mlp_out_cache = torch.cat(list(mlp_out_cache.values()), dim=0)

    attribution_data = model.transcoders.compute_attribution_components(mlp_in_cache)

    reconstruction_ct = attribution_data["reconstruction"][:, 1:, :]
    activations_ct = attribution_data["activation_matrix"].to_dense()[:, 1:, :].transpose(0, 1)

    local_clt = CLT.load_from_pretrained(path=clt_checkpoint, device=str(clt.device))

    log(f"mlp_in_cache norm: {mlp_in_cache.norm()}")

    mlp_in_cache = mlp_in_cache[:, 1:, :]
    mlp_out_cache = mlp_out_cache[:, 1:, :]

    input_for_clt = mlp_in_cache.transpose(0, 1) * local_clt.estimated_norm_scaling_factor_in.view(1, -1, 1)
    activations = local_clt.encode(input_for_clt)[0]

    tolerance = 1e-5

    log(f"local activations l0: {(activations > 0).float().sum()}")
    log(f"circuit activations l0: {(activations_ct > 0).float().sum()}")

    if torch.allclose(activations, activations_ct, atol=tolerance, rtol=tolerance):
        log("activations match within tolerance")
    else:
        max_diff = torch.max(torch.abs(activations - activations_ct))
        log(f"activations mismatch - max diff: {max_diff:.8f}")

    reconstruction_local = local_clt.decode(activations) / local_clt.estimated_norm_scaling_factor_out.view(1, -1, 1)
    reconstruction_local = reconstruction_local.transpose(0, 1)

    if reconstruction_ct.shape != reconstruction_local.shape:
        log(f"Shape mismatch: {reconstruction_ct.shape} vs {reconstruction_local.shape}")
        return False

    if torch.allclose(reconstruction_ct, reconstruction_local, atol=tolerance, rtol=tolerance):
        log("Reconstructions match within tolerance")
        return True
    else:
        max_diff = torch.max(torch.abs(reconstruction_ct - reconstruction_local))
        log(f"Reconstruction mismatch - max diff: {max_diff:.8f}")
        return False
