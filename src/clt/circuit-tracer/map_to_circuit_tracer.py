import torch
from pathlib import Path
from clt.clt import CLT
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer import ReplacementModel

def load_circuit_tracing_clt_from_local(
    clt_checkpoint: str, 
    device: str = "cuda"
) -> CrossLayerTranscoder: 
    
    """
    Loads to a Circuit Tracing CLT from a local CLT checkpoint.
    """

    path = Path(clt_checkpoint)
    print(f"Loading CLT checkpoint from: {clt_checkpoint}")

    clt = CLT.load_from_pretrained(path, device=device)

    state_dict_local = {
        k: v for k, v in clt.state_dict().items()
        if not k.startswith("model.")
    }

    W_dec = state_dict_local["W_dec"] # shape [K, d_transcoder, d_model]
    b_dec = state_dict_local["b_dec"] # shape [K, d_model]
    W_enc = state_dict_local["W_enc"] # shape [n_layers, d_model, d_transcoder]
    b_enc = state_dict_local["b_enc"] # shape [n_layers, d_transcoder]
    log_threshold = state_dict_local["log_threshold"] # shape [n_layers, d_transcoder]
    estimated_norm_scaling_factor_in = state_dict_local["estimated_norm_scaling_factor_in"] # shape [N_layers]
    estimated_norm_scaling_factor_out = state_dict_local["estimated_norm_scaling_factor_out"] # shape [N_layers]

    n_layers = b_enc.shape[0]
    l_idx, k_idx = torch.triu_indices(n_layers, n_layers, offset=0, device=device)
    d_model = W_enc.shape[1]
    d_transcoder = W_enc.shape[2]
    dtype = W_dec.dtype
    feature_input_hook = "ln2.hook_normalized" 
    feature_output_hook = "hook_mlp_out"

    print("\n Model dimensions:")
    print(f"   • Layers: {n_layers}")
    print(f"   • Model dimension: {d_model}")
    print(f"   • Transcoder dimension: {d_transcoder}")
    print(f"   • Data type: {dtype}")
    print(f"   • W_dec original dtype: {W_dec.dtype}")
    print(f"   • W_enc original dtype: {W_enc.dtype}")

    # 1. merge b_dec
    b_dec_circuit = torch.zeros((n_layers, b_dec.shape[1]), device=device)
    for i in range(b_dec.shape[0]):
        # layer_idx = k_idx[i].item() if hasattr(k_idx[i], 'item') else k_idx[i]
        b_dec_circuit[k_idx[i]] += b_dec[i]

    state_dict = {"b_dec": b_dec_circuit, "b_enc": b_enc}
    state_dict["W_enc"] = W_enc.transpose(1, 2) # shape [n_layers, d_transcoder, d_model]

    # 2. split W_dec 
    for i in range(n_layers):
        layer_weights = W_dec[l_idx == i] # shape [n_layers-i, d_transcoder, d_model]
        state_dict[f"W_dec.{i}"] = layer_weights.transpose(0, 1) # shape [d_transcoder, n_layers-i, d_model]

    # 3. log_threshold 
    threshold = torch.exp(log_threshold).unsqueeze(1) # shape [n_layers, 1, d_transcoder]
    state_dict["activation_function.threshold"] = threshold

    # 4. scaling
    print("\n⚖️  Applying normalization scaling:")
    print(f"   • Input scaling factors shape: {estimated_norm_scaling_factor_in.shape}")
    print(f"   • Output scaling factors shape: {estimated_norm_scaling_factor_out.shape}")
    
    # Apply input scaling (multiply)
    state_dict["W_enc"] = state_dict["W_enc"] * estimated_norm_scaling_factor_in.view(-1, 1, 1)
    state_dict["b_enc"] = state_dict["b_enc"]
    state_dict["activation_function.threshold"] = state_dict["activation_function.threshold"]
    
    # Apply output scaling (divide)
    for i in range(n_layers):
        state_dict[f"W_dec.{i}"] = state_dict[f"W_dec.{i}"] / estimated_norm_scaling_factor_out[i:].view(1, -1, 1)
    state_dict["b_dec"] = state_dict["b_dec"] / estimated_norm_scaling_factor_out.view(-1, 1)
        
    # Create instance and load state dict
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
    
    print(f"\n Successfully loaded CLT with {len(state_dict)} parameters")
    print(f"   • Instance W_enc dtype: {instance.W_enc.dtype}")
    print(f"   • Instance b_enc dtype: {instance.b_enc.dtype}")
    print(f"   • Instance b_dec dtype: {instance.b_dec.dtype}")
    return instance

# path = Path(clt_checkpoint)
# cfg_path = path / CLT_CFG_FILENAME
# weights_path = path / CLT_WEIGHTS_FILENAME

# print(f"Loading CLT checkpoint from: {clt_checkpoint}")

# if not weights_path.exists():
#     raise FileNotFoundError(f"Weights file not found: {weights_path}")
# if not cfg_path.exists():
#     raise FileNotFoundError(f"Config file not found: {cfg_path}")

# state_dict_local = load_file(weights_path, device=device)
# state_dict_local = {k: v for k, v in state_dict_local.items() if not k.startswith('model.')}

# should be ran everytime to double check
def test_clt_performance_on_prompt(inputs: str, clt: CrossLayerTranscoder, model: ReplacementModel): 
        
        if isinstance(inputs, str):
            tokens = model.ensure_tokenized(inputs)
        else:
            tokens = inputs.squeeze()

        assert isinstance(tokens, torch.Tensor), "Tokens must be a tensor"
        assert tokens.ndim == 1, "Tokens must be a 1D tensor"

        mlp_in_cache, mlp_in_caching_hooks, _ = model.get_caching_hooks(
            lambda name: model.feature_input_hook in name
        )

        mlp_out_cache, mlp_out_caching_hooks, _ = model.get_caching_hooks(
            lambda name: model.feature_output_hook in name
        )
        _ = model.run_with_hooks(tokens, fwd_hooks=mlp_in_caching_hooks + mlp_out_caching_hooks)

        mlp_in_cache = torch.cat(list(mlp_in_cache.values()), dim=0)
        mlp_out_cache = torch.cat(list(mlp_out_cache.values()), dim=0)

        attribution_data = model.transcoders.compute_attribution_components(mlp_in_cache)
        error_vectors = mlp_out_cache - attribution_data["reconstruction"] # shape [n_layers, n_pos, d_model]
                
        for layer_idx in range(error_vectors.shape[0]):
            layer_errors = error_vectors[layer_idx, 1:]  # shape [n_pos, d_model]
            layer_outputs = mlp_out_cache[layer_idx, 1:]  # shape [n_pos, d_model]

            activations = attribution_data["activation_matrix"].to_dense()[layer_idx, 1:]

            # Compute average l0 per layer
            l0 = (activations > 0).float().sum(1)
            print(f"• Layer {layer_idx}: L0 = {l0.mean().item():.2f}")
            
            mse = torch.mean(layer_errors ** 2)
            variance = torch.var(layer_outputs)
            
            # Compute normalized MSE
            normalized_mse = mse / variance if variance > 0 else float('inf')
            
            print(f"   • Layer {layer_idx}: MSE = {mse:.6f}, Variance = {variance:.6f}, Normalized MSE = {normalized_mse:.6f}")
        
def compare_reconstruction_with_local_clt_class(clt_checkpoint: str, inputs: str, clt: CrossLayerTranscoder, model: ReplacementModel, model_name: str):

    ### get error vectors with circuit-tracing clt class
    if isinstance(inputs, str):
        tokens = model.ensure_tokenized(inputs)
    else:
        tokens = inputs.squeeze()

    assert isinstance(tokens, torch.Tensor), "Tokens must be a tensor"
    assert tokens.ndim == 1, "Tokens must be a 1D tensor"

    mlp_in_cache, mlp_in_caching_hooks, _ = model.get_caching_hooks(
        lambda name: model.feature_input_hook in name
    )

    mlp_out_cache, mlp_out_caching_hooks, _ = model.get_caching_hooks(
        lambda name: model.feature_output_hook in name
    )
    _ = model.run_with_hooks(tokens, fwd_hooks=mlp_in_caching_hooks + mlp_out_caching_hooks)

    mlp_in_cache = torch.cat(list(mlp_in_cache.values()), dim=0)
    mlp_out_cache = torch.cat(list(mlp_out_cache.values()), dim=0)

    attribution_data = model.transcoders.compute_attribution_components(mlp_in_cache)


    reconstruction_circuit_tracing = attribution_data["reconstruction"][:, 1:, :] # shape [n_layers, n_pos, d_model]
    activations_circuit = attribution_data["activation_matrix"].to_dense()[:, 1:, :].transpose(0,1) # shape [n_layers, n_pos, n_features]

    ### get error vectors with local clt class
    local_clt = CLT.load_from_pretrained(
        path=clt_checkpoint,
        device=str(clt.device)
    )

    # Debug: print input shapes and norms
    print(f"mlp_in_cache norm: {mlp_in_cache.norm()}")

    mlp_in_cache = mlp_in_cache[:, 1:, :]
    mlp_out_cache = mlp_out_cache[:, 1:, :]

    input_for_clt = mlp_in_cache.transpose(0, 1) * local_clt.estimated_norm_scaling_factor_in.view(1, -1, 1)
    activations = local_clt.encode(input_for_clt)[0]

    tolerance = 1e-5
    print(f"local activations l0: {(activations > 0).float().sum()}")
    print(f"circuit activations l0: {(activations_circuit > 0).float().sum()}")
    matches = torch.allclose(activations, activations_circuit, atol=tolerance, rtol=tolerance)
    if matches:
        print("✅ activations match within tolerance")
    else:
        max_diff = torch.max(torch.abs(activations - activations_circuit))
        print(f"❌ activations mismatch - max difference: {max_diff:.8f}")

    # Decode without manual normalization 
    reconstruction_local = local_clt.decode(activations) / local_clt.estimated_norm_scaling_factor_out.view(1, -1, 1)
    reconstruction_local = reconstruction_local.transpose(0, 1)

    if reconstruction_circuit_tracing.shape != reconstruction_local.shape:
        print(f"Warning: Shape mismatch - circuit tracing: {reconstruction_circuit_tracing.shape}, local: {reconstruction_local.shape}")
        return False
    
    tolerance = 1e-5
    matches = torch.allclose(reconstruction_circuit_tracing, reconstruction_local, atol=tolerance, rtol=tolerance)
    
    if matches:
        print("✅ Reconstructions match within tolerance")
    else:
        max_diff = torch.max(torch.abs(reconstruction_circuit_tracing - reconstruction_local))
        print(f"❌ Reconstruction mismatch - max difference: {max_diff:.8f}")
    
    return matches
