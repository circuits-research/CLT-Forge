from circuitlab import logger

import torch
from typing import List, Dict, Any, Tuple

from circuitlab.vendor.circuit_tracer.circuit_tracer import ReplacementModel

def _decode_top_tokens(model, probs, top_k: int):
    top_tokens = torch.topk(probs, top_k)
    token_strings = [
        model.tokenizer.decode([token_id]) for token_id in top_tokens.indices
    ]
    return top_tokens, token_strings

def run_intervention(
    model: ReplacementModel,
    features: List[Tuple[int, int, int, float]], # features: list of (pos, layer, feature_idx, intervention_value)
    input_string: str,
    top_tokens_count: int = 4,
    freeze_attention: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Apply interventions on a set of features.
    """

    def log(msg):
        if debug:
            logger.info(msg)

    # Convert to expected format (layer, pos, feature_idx, value)
    features_to_intervene = [
        (layer, pos, feature_idx, value)
        for pos, layer, feature_idx, value in features
    ]

    input_tokens = model.ensure_tokenized(input_string)

    intervened_logits, _ = model.feature_intervention(
        input_tokens,
        interventions=features_to_intervene,
        freeze_attention=freeze_attention,
    )

    original_logits = model(input_tokens)

    original_probs = torch.softmax(original_logits[0, -1], dim=-1)
    intervened_probs = torch.softmax(intervened_logits[0, -1], dim=-1)

    top_tokens, top_token_strings = _decode_top_tokens(
        model, intervened_probs, top_tokens_count
    )

    baseline_top_tokens, baseline_token_strings = _decode_top_tokens(
        model, original_probs, top_tokens_count
    )

    baseline_probs = []
    prob_differences = []

    for token_id in top_tokens.indices:
        baseline_prob = original_probs[token_id].item()
        intervened_prob = intervened_probs[token_id].item()

        baseline_probs.append(baseline_prob)
        prob_differences.append(intervened_prob - baseline_prob)

    return {
        "tokens": top_token_strings,
        "probabilities": top_tokens.values.cpu().tolist(),
        "baseline_probabilities": baseline_probs,
        "probability_differences": prob_differences,
        "n_features": len(features),
        "top_baseline_token_strings": baseline_token_strings,
        "intervened_probs": intervened_probs.cpu(),
        "original_probs": original_probs.cpu(),
    }

def run_intervention_per_feature(  # useful for the visual interface
    model: ReplacementModel,
    input_string: str,
    result: Dict[str, Any],
    intervention_values: List[float] = [5.0, -5.0, -10.0],
    top_tokens_count: int = 4,
    freeze_attention: bool = True,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Applies independent interventions for each feature and returns the top predicted tokens
    """

    def log(msg):
        if debug:
            logger.info(msg)

    input_tokens = model.ensure_tokenized(input_string)
    data = result

    feature_indices = data["feature_indices"]
    feature_mask = data["feature_mask"]

    logit_probabilities = data["logit_probabilities"]
    logit_tokens = data["logit_tokens"]

    baseline_top_prob = logit_probabilities.max().item()
    baseline_top_idx = logit_tokens[logit_probabilities.argmax().item()]

    n_features = len(feature_indices)
    active_feature_indices = feature_indices[feature_mask[:n_features]]

    log(f"Intervening on {len(active_feature_indices)} features")

    results: List[Dict[str, Any]] = []

    for feature_data in active_feature_indices:
        layer = int(feature_data[1])
        pos = int(feature_data[0])
        feature_idx = int(feature_data[2])

        feature_result: Dict[str, Any] = {
            "feature_info": {
                "layer": layer,
                "position": pos,
                "feature_idx": feature_idx,
            },
            "interventions": [],
        }

        for value in intervention_values:
            features_to_intervene = [(layer, pos, feature_idx, value)]

            try:
                intervened_logits, _ = model.feature_intervention(
                    input_tokens,
                    interventions=features_to_intervene,
                    freeze_attention=freeze_attention,
                )

                intervened_probs = torch.softmax(
                    intervened_logits[0, -1], dim=-1
                )

                top_tokens, token_strings = _decode_top_tokens(
                    model, intervened_probs, top_tokens_count
                )

                new_prob = intervened_probs[baseline_top_idx].item()
                prob_change = new_prob - baseline_top_prob

                feature_result["interventions"].append(
                    {
                        "intervention_value": value,
                        "tokens": token_strings,
                        "probabilities": top_tokens.values.cpu().tolist(),
                        "baseline_token": int(baseline_top_idx),
                        "baseline_prob_original": baseline_top_prob,
                        "baseline_prob_after_intervention": new_prob,
                        "baseline_prob_change": prob_change,
                    }
                )

            except Exception as e:
                log(
                    f"Intervention failed for "
                    f"layer={layer}, pos={pos}, feature={feature_idx}, value={value}: {e}"
                )

                feature_result["interventions"].append(
                    {
                        "intervention_value": value,
                        "tokens": [],
                        "probabilities": [],
                        "baseline_token": None,
                        "baseline_prob_original": 0.0,
                        "baseline_prob_after_intervention": 0.0,
                        "baseline_prob_change": 0.0,
                    }
                )

        results.append(feature_result)

    return results
