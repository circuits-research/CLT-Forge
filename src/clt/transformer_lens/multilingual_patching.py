from pathlib import Path

def patch_official_model_names():
    """Add multilingual TinyStories models to TransformerLens OFFICIAL_MODEL_NAMES."""
    
    try:
        from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES, MODEL_ALIASES
        
        multilingual_models = [
            "CausalNLP/tinystories-multilingual-20", 
            "CausalNLP/tinystories-multilingual-50",
            "CausalNLP/tinystories-multilingual-70",
            "CausalNLP/tinystories-multilingual-90",
            "CausalNLP/gpt2-hf_multilingual-20", 
            "CausalNLP/gpt2-hf_multilingual-50",
            "CausalNLP/gpt2-hf_multilingual-70",
            "CausalNLP/gpt2-hf_multilingual-90",
        ]
        
        for model in multilingual_models:
            if model not in OFFICIAL_MODEL_NAMES:
                OFFICIAL_MODEL_NAMES.append(model)
                print(f"Added {model} to OFFICIAL_MODEL_NAMES")
            if model not in MODEL_ALIASES:
                MODEL_ALIASES[model] = [model]
                print(f"Added {model} to MODEL_ALIASES")

    except ImportError:
        print("TransformerLens not found. Make sure it's installed.")
    except Exception as e:
        print(f"Error patching OFFICIAL_MODEL_NAMES: {e}")

def patch_convert_hf_model_config():
    """Patch convert_hf_model_config to ensure CausalNLP models are handled correctly."""
    
    try:
        from transformer_lens import loading_from_pretrained
        import os
        from transformers import AutoConfig
        
        original_convert_hf_model_config = loading_from_pretrained.convert_hf_model_config
        
        def patched_convert_hf_model_config(model_name: str, **kwargs):

            if (Path(model_name) / "config.json").exists():
                official_model_name = model_name
            else:
                official_model_name = loading_from_pretrained.get_official_model_name(model_name)

            if "CausalNLP" in official_model_name:

                huggingface_token = os.environ.get("HF_TOKEN", "")
                hf_config = AutoConfig.from_pretrained(
                    official_model_name,
                    token=huggingface_token if len(huggingface_token) > 0 else None,
                    **kwargs,
                )
                
                cfg_dict = {
                    "d_model": hf_config.n_embd,
                    "d_head": hf_config.n_embd // hf_config.n_head,
                    "n_heads": hf_config.n_head,
                    "d_mlp": hf_config.n_embd * 4,
                    "n_layers": hf_config.n_layer,
                    "n_ctx": hf_config.n_positions,
                    "eps": hf_config.layer_norm_epsilon,
                    "d_vocab": hf_config.vocab_size,
                    "act_fn": hf_config.activation_function,
                    "use_attn_scale": True,
                    "use_local_attn": False,
                    "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
                    "normalization_type": "LN",
                    "original_architecture": "GPT2LMHeadModel", 
                    "tokenizer_name": official_model_name,
                }
                
                if kwargs.get("trust_remote_code", False):
                    cfg_dict["trust_remote_code"] = True
                    
                return cfg_dict
            else:
                return original_convert_hf_model_config(model_name, **kwargs)

        loading_from_pretrained.convert_hf_model_config = patched_convert_hf_model_config
        print("Successfully patched convert_hf_model_config for CausalNLP models")
        
    except ImportError:
        print("TransformerLens not found. Make sure it's installed.")
    except Exception as e:
        print(f"Error patching convert_hf_model_config: {e}")


# patch_official_model_names()
# patch_convert_hf_model_config()
