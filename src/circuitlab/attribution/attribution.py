from circuitlab import logger

from circuitlab.attribution.loading import (
    load_circuit_tracing_clt_from_local,
    test_clt_performance_on_prompt,
    compare_reconstruction_with_local_clt_class,
)
from circuitlab.attribution.intervention import (
    run_intervention,
    run_intervention_per_feature,
)

from circuitlab.vendor.circuit_tracer.circuit_tracer import ReplacementModel, attribute
from circuitlab.vendor.circuit_tracer.circuit_tracer.graph import prune_graph, compute_graph_scores

import os
import torch
from typing import List, Dict, Any

class AttributionRunner:
    def __init__(
        self,
        clt_checkpoint: str,
        model_name: str = "gpt2",
        device: str = "cuda",
        debug: bool = False,
    ):
        self.debug = debug

        def log(msg):
            if self.debug:
                logger.info(msg)

        self.log = log

        self.log("Loading CLT...")
        self.clt = load_circuit_tracing_clt_from_local(
            clt_checkpoint, device=device, debug=debug
        )

        self.log("Loading model...")
        self.model = ReplacementModel.from_pretrained_and_transcoders(
            model_name=model_name,
            transcoders=self.clt,
        )

        self.clt_checkpoint = clt_checkpoint
        self.model_name = model_name

    def _build_result(self, graph, prune_result, input_string) -> Dict[str, Any]:
        sparse_adjacency = prune_result.edge_mask.float()

        active_feature = torch.stack(
            [
                graph.active_features[:, 1],
                graph.active_features[:, 0],
                graph.active_features[:, 2],
            ],
            dim=1,
        )

        token_string = [self.model.tokenizer.decode(t) for t in graph.input_tokens]
        logit_token_strings = [self.model.tokenizer.decode(t) for t in graph.logit_tokens]

        return {
            "adjacency_matrix": graph.adjacency_matrix.cpu(),
            "feature_indices": active_feature.cpu(),
            "sparse_pruned_adj": sparse_adjacency.cpu(),
            "feature_mask": prune_result.node_mask.cpu(),
            "edge_mask": prune_result.edge_mask.cpu(),
            "logit_tokens": graph.logit_tokens.cpu(),
            "logit_probabilities": graph.logit_probabilities.cpu(),
            "input_tokens": graph.input_tokens.cpu(),
            "input_string": input_string,
            "token_string": token_string,
            "logit_token_strings": logit_token_strings,
        }

    def run(
        self,
        input_string: str,
        folder_name: str,
        graph_name: str = "attribution_graph.pt",
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
        max_feature_nodes: int = 8192,
        batch_size: int = 256,
        offload: str = "cpu",
        verbose: bool = True,
        feature_threshold: float = 0.8,
        edge_threshold: float = 0.95,
        run_interventions: bool = True,
        intervention_values: List[float] = [0, -5.0, -10.0],
    ):
        self.log(f"Running attribution for prompt: {input_string[:50]}...")

        if self.debug:
            self.log("Running CLT validation checks...")

            test_clt_performance_on_prompt(
                input_string, self.clt, self.model, debug=self.debug
            )

            compare_reconstruction_with_local_clt_class(
                self.clt_checkpoint,
                input_string,
                self.clt,
                self.model,
                self.model_name,
                debug=self.debug,
            )

        graph = attribute(
            prompt=input_string,
            model=self.model,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
        )

        replacement_score, completeness_score = compute_graph_scores(graph)

        self.log(f"Replacement score: {replacement_score:.4f}")
        self.log(f"Completeness score: {completeness_score:.4f}")

        prune_result = prune_graph(
            graph=graph,
            node_threshold=feature_threshold,
            edge_threshold=edge_threshold,
        )

        if self.debug:
            self.log(f"Sparse adjacency shape: {prune_result.edge_mask.shape}")

        n_features = graph.active_features.shape[0]
        self.log(f"Number of features before pruning: {n_features}")
        self.log(f"Number of feature after pruning (not counting error nodes): {prune_result.node_mask[:n_features].sum().item()}")

        result = self._build_result(graph, prune_result, input_string)

        if run_interventions:
            self.log("Running interventions...")

            intervention_results = self.run_intervention_per_feature(
                input_string=input_string,
                result=result,
                intervention_values=intervention_values,
            )

            result["intervention_top_tokens"] = intervention_results

        # save final results
        os.makedirs(folder_name, exist_ok=True)
        save_path = os.path.join(folder_name, graph_name)
        torch.save(result, save_path)

        self.log(f"Saved attribution graph to {save_path}")

        return result

    def run_intervention_per_feature(
        self,
        input_string: str,
        result: Dict[str, Any],
        intervention_values: List[float],
    ):
        return run_intervention_per_feature(
            model=self.model,
            input_string=input_string,
            result=result,
            intervention_values=intervention_values,
            debug=self.debug,
        )

    def run_intervention(
        self,
        input_string: str,
        features,
        **kwargs,
    ):
        return run_intervention(
            model=self.model,
            input_string=input_string,
            features=features,
            debug=self.debug,
            **kwargs,
        )
