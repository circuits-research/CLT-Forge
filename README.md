<p align="center">
  <img src="./images/image.png" alt="CLT banner" width="800"/>
</p>

[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/framework-pytorch-red)]()
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CircuitLab** is a Python library for training [Cross-Layer Transcoders](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (CLTs) at scale. It will soon include an automatic intepretability pipeline and a visual interface. 

We believe that a major limitation in the development of CLTs, and more broadly attribution graph methods, is the significant engineering effort required to train, analyze, and iterate on them. This library aims to reduce that overhead by providing a clean, scalable, and extensible framework.

## Features

This library currently implements L1-regularized [JumpReLU](https://arxiv.org/pdf/2407.14435) CLTs with the following design principles:

- Follows [Anthropic](https://transformer-circuits.pub/2025/january-update/index.html)'s training guidelines
- Supports feature sharding across GPUs (as well as DDP and FSDP)  
- Includes activation caching and compression/quantization of the activations  
- Adopts a structure similar to [SAE Lens](https://github.com/jbloomAus/SAELens) (code design, activation-store, etc.) and uses [Transformer Lens](https://github.com/TransformerLensOrg/TransformerLens)

## Stay-Tuned

We also plan to release within the same package (March 2026):
- An automatic interpretability pipeline  
- A visual interface for exploring features and attribution graphs  
  - Similar in spirit (but simpler) to [Neuronpedia](https://github.com/hijohnnylin/neuronpedia)
  - Including attention attribution support (as in [SparseAttention](https://arxiv.org/abs/2512.05865))

## Quick Start

Training happens in **two steps**:

1.  **Precompute activations** (should be parallelized across indepedent jobs)
2.  **Train the CLT model** on the cached activations (should run on a single multi-gpu node)

------------------------------------------------------------------------

### 1. Generate and cache activations

``` python
from circuitlab import ActivationsStore, clt_training_runner_config, load_model

# Load model
model = load_model("meta-llama/Llama-3.2-1B", device="cuda")

# Create config
cfg = clt_training_runner_config()

# Create activation store
store = ActivationsStore(model, cfg)

# Generate and cache activations
store.generate_and_save_activations(
    path=cfg.cached_activations_path,
    use_compression=True,  # optional
)
```

------------------------------------------------------------------------

### 2. Train the CLT

``` python
from circuitlab import CLTTrainingRunner

# Train
trainer = CLTTrainingRunner(cfg)
trainer.run()
```

------------------------------------------------------------------------

## ⚙️ Notes
-   We provide screenshot examples of training metrics in the [output](./outputs) folder and sample training scripts in [runners](./runners/)
-   Compression is optional but recommended for large-scale runs (e.g. 1B +) with 4-8x memory reduction
-   Training with bf16 is fine (autocasting with activations and weights in bf16 but gradient states in 32) but requires higher lr (around 1.5-2x bigger)
-   For Llama 1B, on a full 8 gpu H100 node, we reach an expansion factor of 42 with micro-batch size 512
-   We provide a sample script to map model weigths to [circuit-tracer](https://github.com/safety-research/circuit-tracer) in the [file](./src/circuitlab/circuit-tracer/map_to_circuit_tracer.py)
-   There has been recent criticism of the faithfulness of CLTs (see [post](https://www.lesswrong.com/posts/6CS2NDmoLCFcEJMor/cross-layer-transcoders-are-incentivized-to-learn-unfaithful)). We are also currently studying this phenomenon.

## Citation

```bibtex
@misc{CircuitLab,
  title   = {CircuitLab: A Scalable Framework for Cross-Layer Transcoders Training and Attribution-Graph Visualization},
  author  = {Draye, Florent, Harasse Abir, T },
  year    = {2026},
  url     = {https://github.com/CLT-Training/CLT/}
}
