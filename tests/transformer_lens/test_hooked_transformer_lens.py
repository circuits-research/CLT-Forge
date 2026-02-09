import pytest
import torch
from transformer_lens import HookedTransformer
from featflow.transformer_lens.hooked_transformer_wrapper import patch_transformer_lens

### Just tests to check our own understanding of transformer lens, useful for future extensions to replacement score finetuning

@pytest.mark.parametrize("model_name", ["gpt2", "roneneldan/TinyStories-33M"])
def test_decomposed_forward_equivalence(model_name):
    """Test that decomposed forward methods give same result as standard forward and run_with_cache."""
    print(f"\n=== Testing {model_name} ===")
    model = HookedTransformer.from_pretrained(model_name, device="cpu")
    patch_transformer_lens()
    
    test_prompts = [
        "Hello",
        "Bonjour",
        "Hallo",
        "The cat",
        "Le chat",
        "Die Katze",
        "Once upon a time",
        "Il était une fois",
        "Es war einmal",
        "What is the meaning of life?",
        "Quel est le sens de la vie?",
        "Was ist der Sinn des Lebens?",
        "In a small village nestled between rolling hills and a sparkling river, there lived a curious little girl named Emma who loved to explore.",
        "Dans un petit village niché entre des collines ondulantes, vivait une petite fille curieuse.",
        "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once.",
    ]
    
    test_inputs = []
    for prompt in test_prompts:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        test_inputs.append((prompt, tokens))
    
    for prompt, input_tokens in test_inputs:
        print(f"\nTesting prompt: '{prompt}'")
        print(f"Token shape: {input_tokens.shape}")
        
        # Standard forward pass
        expected_logits = model(input_tokens, return_type="logits")
        cache_logits, cache = model.run_with_cache(input_tokens)
        
        # Using our decomposed way
        residual, tokens, shortformer_pos_embed, attention_mask = model._process_input_to_residual(input_tokens)
        residual = model._run_transformer_blocks(residual, shortformer_pos_embed, attention_mask)
        our_logits = model._residual_to_output(residual, return_type="logits")
        
        assert torch.allclose(our_logits, expected_logits, atol=1e-6), f"Expected vs our mismatch for: {prompt}"
        assert torch.allclose(our_logits, cache_logits, atol=1e-6), f"Cache vs our mismatch for: {prompt}"
        assert torch.allclose(expected_logits, cache_logits, atol=1e-6), f"Expected vs cache mismatch for: {prompt}"
        assert our_logits.shape == expected_logits.shape == cache_logits.shape
        
        # Generate next token predictions and compare
        next_token_logits_expected = expected_logits[0, -1, :]  # Last position logits
        next_token_logits_our = our_logits[0, -1, :]
        next_token_logits_cache = cache_logits[0, -1, :]
        
        top5_expected = torch.topk(next_token_logits_expected, 5)
        top5_our = torch.topk(next_token_logits_our, 5)
        top5_cache = torch.topk(next_token_logits_cache, 5)
        
        expected_tokens = [model.to_string(token_id) for token_id in top5_expected.indices]
        our_tokens = [model.to_string(token_id) for token_id in top5_our.indices]
        cache_tokens = [model.to_string(token_id) for token_id in top5_cache.indices]
        
        print(f"  Top 5 next tokens (expected): {expected_tokens}")
        print(f"  Top 5 next tokens (our method): {our_tokens}")
        print(f"  Top 5 next tokens (cache): {cache_tokens}")
        
        assert torch.equal(top5_expected.indices, top5_our.indices), f"Top token predictions differ for: {prompt}"
        assert torch.equal(top5_expected.indices, top5_cache.indices), f"Top token predictions differ for: {prompt}"
        
        # Test actual text generation by sampling
        temperature = 0.7
        probs_expected = torch.softmax(next_token_logits_expected / temperature, dim=-1)
        probs_our = torch.softmax(next_token_logits_our / temperature, dim=-1)
        
        torch.manual_seed(42)
        sampled_expected = torch.multinomial(probs_expected, 1)
        torch.manual_seed(42)
        sampled_our = torch.multinomial(probs_our, 1)
        
        assert torch.equal(sampled_expected, sampled_our), f"Sampled tokens differ for: {prompt}"
        
        # Generate a short continuation to see actual model behavior
        if len(prompt) < 50: 
            print("  Generating continuation...")
            
            current_tokens = input_tokens.clone()
            for _ in range(5):
                logits = model(current_tokens, return_type="logits")
                next_token = torch.argmax(logits[0, -1, :])
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            expected_continuation = model.to_string(current_tokens[0])
            
            current_tokens = input_tokens.clone()
            for _ in range(5):
                residual, tokens, shortformer_pos_embed, attention_mask = model._process_input_to_residual(current_tokens)
                residual = model._run_transformer_blocks(residual, shortformer_pos_embed, attention_mask)
                logits = model._residual_to_output(residual, return_type="logits")
                next_token = torch.argmax(logits[0, -1, :])
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            our_continuation = model.to_string(current_tokens[0])
            
            print(f"  Expected continuation: '{expected_continuation}'")
            print(f"  Our method continuation: '{our_continuation}'")
            
            assert expected_continuation == our_continuation, f"Generated text differs for: {prompt}"

    assert False
