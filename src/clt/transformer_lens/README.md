# transformer_lens

This folder contains: 

- hooked_transformer_wrapper which breaks down the forward function of transformer_lens to replace the activations with the CLT activations and get a differentiable replacement score.
- multilingual_patching and sparse_patching which add models to transformer_lens to train CLTs on them (related to other papers), to avoid changing transformer_lens source_code.
