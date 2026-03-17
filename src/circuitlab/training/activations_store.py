
from __future__ import annotations
from typing import Iterator, Optional, Union, Any
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import save_file, load
import datasets
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformer_lens.hook_points import HookedRootModule
from sae_lens.tokenization_and_batching import concat_and_batch_sequences

from circuitlab.training.compressed_activations_store import (
    CompressionConfig,
    CompressedActivationsStore,
)
from circuitlab.config import CLTTrainingRunnerConfig, AutoInterpConfig
from circuitlab.utils import DummyModel, activation_split_path
from circuitlab import logger
from circuitlab.utils import DTYPE_MAP
from concurrent.futures import ThreadPoolExecutor, Future

class ActivationsStore:
    """
    * Streams activations by: 
        - Generating and saving activations to disk (or)
        - Generating activations on the fly

    COMMENTS: 
        - We add the BOS in front of each sequence, but remove it from activations, thus window becomes one token shorter.
    """

    # keep mypy happy
    cached_act_in: torch.Tensor
    cached_act_out: torch.Tensor
    cached_tokens: torch.Tensor
    cache_ptr: int
    leftover_activations: Optional[dict[str, Any]]

    def __init__(self, 
            model: Union[HookedRootModule, DummyModel], 
            cfg: Union[CLTTrainingRunnerConfig, AutoInterpConfig],
            rank: int = 0, 
            world_size: int = 1,
            estimated_norm_scaling_factor_in: Optional[torch.Tensor] = None,
            estimated_norm_scaling_factor_out: Optional[torch.Tensor] = None
        ) -> None:

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.dtype = DTYPE_MAP[cfg.dtype]

        if isinstance(cfg, AutoInterpConfig): 
            self.shuffle = False
            self.return_tokens = True
            self.mix_with_previous_buffer = False
        elif isinstance(cfg, CLTTrainingRunnerConfig): 
            self.shuffle = True
            self.return_tokens = False
            self.mix_with_previous_buffer = False
        else:
            raise TypeError("cfg must be either CLTTrainingRunnerConfig or AutoInterpConfig")
        
        model.cfg.use_hook_mlp_in = True # probably should remove
        self.buffer_counter = 0
        if cfg.n_batches_in_buffer < 2 or cfg.n_batches_in_buffer % 2:
            raise ValueError("n_batches_in_buffer must be an even integer ≥ 2")

        self.buffer_batches = cfg.n_batches_in_buffer
        self.half_buffer_batches = cfg.n_batches_in_buffer // 2
        self.context_size = cfg.context_size
        self.n_train_batch_per_buffer = cfg.n_train_batch_per_buffer

        self.N_layers = model.cfg.n_layers
        self.hook_names_in  = [f"blocks.{i}.ln2.hook_normalized"  for i in range(self.N_layers)]
        self.hook_names_out = [f"blocks.{i}.hook_mlp_out" for i in range(self.N_layers)]

        if self.cfg.cached_activations_path is None: 
            if self.cfg.is_multilingual_split_dataset: 
                # helper for multilingual dataset as in harrasse2026tracingmultilingualrepresentationsllms
                logger.info("Loading entire dataset...")
                self.raw_ds = load_dataset_multilingual(cfg.dataset_path).shuffle(seed=42)
                logger.info(f"First sample sequence: {self.raw_ds[0]}")
                self.doc_languages = [self.raw_ds[i]["language"] for i in range(len(self.raw_ds))]
            else:
                self.raw_ds = load_dataset_auto(cfg.dataset_path, split=cfg.split, disk=cfg.disk)
            logger.info("Loaded dataset")
        
            if "tokens" not in self.raw_ds.column_names:
                if "input_ids" in self.raw_ds.column_names:
                    logger.info("tokens column not found — using input_ids instead.")
                    self.raw_ds = self.raw_ds.rename_column("input_ids", "tokens")
                else:
                    raise ValueError(
                        f"Dataset {cfg.dataset_path} must contain a pre-tokenised tokens or input_ids column."
                    )
                
            first_tok = self.raw_ds[0]["tokens"]

            if isinstance(first_tok, torch.Tensor) and first_tok.ndim != 1:
                raise ValueError("Each 'tokens' entry must be a 1‑D tensor.")
            if isinstance(first_tok, (list, tuple)) and any(isinstance(x, list) for x in first_tok):
                raise ValueError("Nested sequences detected; expected a flat list of ints.")

            self._reset_token_iterator()
        else: 
            self._load_cached_activations()

        self._storage_in : Optional[torch.Tensor] = None
        self._storage_out: Optional[torch.Tensor] = None
        self._storage_tokens: Optional[torch.Tensor] = None
        self._yield_iter: Optional[
            Union[
                Iterator[tuple[torch.Tensor, torch.Tensor]],
                Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            ]
        ] = None

        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_future: Optional[Future] = None

        # Global schedule state: which rank is the active source for the current buffer.
        self.active_src_rank = 0

        self.estimated_norm_scaling_factor_in = estimated_norm_scaling_factor_in
        self.estimated_norm_scaling_factor_out = estimated_norm_scaling_factor_out

        self._skip_norm_application = True
        self.set_norm_scaling_factor_if_needed()
        self._skip_norm_application = False

        # Build this rank's current local buffer.
        self._rebuild_buffers()

        # Start background build of this rank's next local buffer.
        if self.cfg.cached_activations_path is not None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
            self._start_prefetch()

        assert self.cfg.train_batch_size_tokens % self.cfg.context_size == 0, "ctx size must divide train_batch_size_tokens"
    # ───────────────────  token pipeline  ───────────────────

    def _iterate_raw_dataset_tokens(self) -> Iterator[torch.Tensor]:
        """
        Yield each row's token vector as a 1‑D torch.Tensor on **CPU**.
        """
        dataset_len = len(self.raw_ds)
        if self.cfg.is_distributed:
            shard_size = dataset_len // self.world_size
            start = self.rank * shard_size
            end = dataset_len if self.rank == self.world_size - 1 else start + shard_size
        else:
            start = 0
            end = dataset_len
        self.runtime_doc_languages = []

        for i in range(start, end):
            toks = self.raw_ds[i]["tokens"]
            tokenizer = getattr(self.model, "tokenizer", None)
            bos_id = None if tokenizer is None else tokenizer.bos_token_id

            if not isinstance(toks, torch.Tensor):
                toks = torch.tensor(toks, dtype=torch.long)
            
            if bos_id is not None and len(toks) > 0 and toks[0].item() == bos_id:
                toks = toks[1:]
            
            doc_len = len(toks)
            truncated_len = (doc_len // (self.context_size)) * (self.context_size) # TODO before -1 to both

            if truncated_len > 0:
                toks = toks[:truncated_len]

                if self.cfg.is_multilingual_split_dataset: 
                    n_sequences = truncated_len // (self.context_size)
                    for _ in range(n_sequences):
                        self.runtime_doc_languages.append(self.doc_languages[i])

            yield toks

    def _reset_token_iterator(self) -> None:
        tokenizer = getattr(self.model, "tokenizer", None)
        bos_id = None if tokenizer is None else tokenizer.bos_token_id

        base_iter = concat_and_batch_sequences(
            tokens_iterator=self._iterate_raw_dataset_tokens(),
            context_size=self.context_size,
            begin_batch_token_id=None, # we dont want to prepend bos here, we add in run_with_cache
            begin_sequence_token_id=None,
            sequence_separator_token_id=bos_id,
        )

        self._token_iter = iter(self._batchify(base_iter))

    def _batchify(self, iterator: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
        batch = []
        for item in iterator:
            if item.shape[0] != self.context_size:
                continue  # skip last sequence that might be too short

            batch.append(item)
            if len(batch) == self.cfg.store_batch_size_prompts:
                yield torch.stack(batch)
                batch = []

    def _next_token_batch(self) -> torch.Tensor:
        try:
            batch = next(self._token_iter)
        except StopIteration:
            self._reset_token_iterator()
            batch = next(self._token_iter)

        return batch.to(device=self.device, dtype=torch.long)

    # ───────────────────  activation  ───────────────────

    @torch.no_grad()
    def _activations(self, batch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns batch of activations without BOS activations from batch of tokens"""

        # Manually add BOS token to ensure we have context_size + 1 tokens
        bos_token = self.model.tokenizer.bos_token_id
        bos_column = torch.full((batch_tokens.shape[0], 1), bos_token, dtype=batch_tokens.dtype, device=batch_tokens.device)
        batch_tokens_with_bos = torch.cat([bos_column, batch_tokens], dim=1)
        
        cache = self.model.run_with_cache(
            batch_tokens_with_bos,
            names_filter=self.hook_names_in+self.hook_names_out,
            prepend_bos=False, # We already added BOS manually
        )[1]
        
        def stack(names: list[str]) -> torch.Tensor:
            missing = [n for n in names if n not in cache]
            if missing:
                raise KeyError(f"The following hooks were not found in the cache: {missing}")
            acts_list: list[torch.Tensor] = [cache[n].flatten(2) for n in names]
            acts = torch.stack(acts_list, dim=0).permute(1, 2, 0, 3)
            acts = acts[:, 1:, :, :]  # Remove BOS token activations at pos 0

            assert acts.shape[1] == self.context_size
            # Flatten B and C → (B * C, N_layers, d)
            B, C, N_layers, d = acts.shape
            return acts.reshape(B * C, N_layers, d).cpu()

        return stack(self.hook_names_in), stack(self.hook_names_out)

    def _skip_batches(self):
        """Skip batches by advancing the token iterator without computing activations"""
        n_batches = self.cfg.train_batch_size_tokens // (self.context_size * self.cfg.store_batch_size_prompts)
        assert self.cfg.train_batch_size_tokens % (self.context_size * self.cfg.store_batch_size_prompts) == 0
        tokens = []
        for _ in range(n_batches):
            token_batch = self._next_token_batch()
            tokens.append(token_batch.cpu())
        
        # Return flattened tokens similar to _fresh_activation_batches
        return torch.cat([t.reshape(-1) for t in tokens], dim=0)

    def _fresh_activation_batches(self, return_tokens: bool = False, mix_with_previous_buffer: bool = True):
        n_batches = self.buffer_batches if not mix_with_previous_buffer else self.half_buffer_batches
        ins, outs, toks = [], [], []

        for i in range(n_batches):
            token_batch = self._next_token_batch()
            act_in, act_out = self._activations(token_batch)
            ins.append(act_in)
            outs.append(act_out)
            if return_tokens:
                assert token_batch.shape[1] == self.context_size, "wrong token batch size"
                toks.append(token_batch.cpu()) 

        if return_tokens:
            toks = torch.cat([t.reshape(-1) for t in toks], dim=0)  # flatten to [B * ctx]
            return torch.cat(ins, 0), torch.cat(outs, 0), toks
        else:
            return torch.cat(ins, 0), torch.cat(outs, 0)

    # def _rebuild_buffers(self) -> None:

    #     # Because all 

    #     return_tokens = self.return_tokens
    #     use_cached = self.cfg.cached_activations_path is not None

    #     if use_cached:
    #         result = self._load_buffer_from_cached(return_tokens=return_tokens)
    #     else:
    #         result = self._fresh_activation_batches(
    #             return_tokens=return_tokens,
    #             mix_with_previous_buffer=self.mix_with_previous_buffer
    #         )
        
    #     if return_tokens:
    #         new_in, new_out, new_tokens = result
    #     else:
    #         new_in, new_out = result

    #     if self.mix_with_previous_buffer and self._storage_in is not None:
    #         all_in = torch.cat([self._storage_in, new_in], dim=0)
    #         all_out = torch.cat([self._storage_out, new_out], dim=0)
    #         if self.return_tokens:
    #             all_tokens = torch.cat([self._storage_tokens, new_tokens], dim=0)
    #     else:
    #         all_in, all_out = new_in, new_out
    #         if self.return_tokens:
    #             all_tokens = new_tokens

    #     # all ranks use the same seed for shuffling the current buffer.
    #     if self.cfg.is_sharded:
    #         self._sync_buffer_counter()

    #     # print("Shuffle")
    #     # if self.shuffle:
    #     #     g = torch.Generator()
    #     #     g.manual_seed(42 + self.buffer_counter)
            
    #     #     perm = torch.randperm(all_in.size(0), generator=g, device="cpu")
    #     #     all_in, all_out = all_in[perm], all_out[perm]
    #     #     if self.return_tokens:
    #     #         all_tokens = all_tokens[perm]
        
    #     self.buffer_counter += 1

    #     if self.cfg.uses_process_group:
    #         dist.barrier()

    #     if self.mix_with_previous_buffer:
    #         split = all_in.size(0) // 2
    #         self._storage_in, self._storage_out = all_in[:split], all_out[:split]
    #         if self.return_tokens:
    #             self._storage_tokens = all_tokens[:split]
    #     else: 
    #         split = 0
        
    #     if self.return_tokens:
    #         dataset = TensorDataset(all_in[split:], all_out[split:], all_tokens[split:])
    #     else:
    #         dataset = TensorDataset(all_in[split:], all_out[split:])
        
    #     g = torch.Generator()
    #     g.manual_seed(42 + self.buffer_counter)

    #     loader = DataLoader(
    #         dataset,
    #         batch_size=self.cfg.train_batch_size_tokens,
    #         drop_last=True,
    #         pin_memory=True, 
    #         generator=g, 
    #         shuffle=True
    #     )

    #     self._yield_iter = iter(loader)

    def _sync_buffer_counter(self) -> None:
        counter = torch.tensor(
            self.buffer_counter,
            dtype=torch.int64,
            device=self.device,
        )
        dist.broadcast(counter, src=0)
        self.buffer_counter = counter.item()

    # ───────────────────  normalization  ───────────────────

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, n_batches_for_norm_estimate: int = 10):
        
        norms_per_layer_in = []
        norms_per_layer_out = []

        self.estimated_norm_scaling_factor_in = torch.ones(self.N_layers, device=self.device).float()
        self.estimated_norm_scaling_factor_out = torch.ones(self.N_layers, device=self.device).float()

        for _ in tqdm(range(n_batches_for_norm_estimate), 
                      desc="Estimating norm scaling factor", 
                      disable=(self.rank != 0)): 
            
            # cache_ptr is aligned across all nodes.
            acts_in, acts_out = next(iter(self))

            if self.rank == 0:
                norms_per_layer_in.append(acts_in.norm(dim=-1).mean(dim=0).float())
                norms_per_layer_out.append(acts_out.norm(dim=-1).mean(dim=0).float())

        del acts_in, acts_out
            
        if self.rank == 0:

            mean_norm_per_layer_in = torch.stack(norms_per_layer_in, dim=0).mean(dim=0)
            mean_norm_per_layer_out = torch.stack(norms_per_layer_out, dim=0).mean(dim=0)
            
            # The scaling factor is calculated to normalize the norm of the activations 
            # to the square root of the input dimension (d_in ** 0.5)
            self.estimated_norm_scaling_factor_in = (self.cfg.d_in ** 0.5) / mean_norm_per_layer_in
            self.estimated_norm_scaling_factor_out = (self.cfg.d_in ** 0.5) / mean_norm_per_layer_out

            logger.info(f"Estimated norm scaling factor in: {self.estimated_norm_scaling_factor_in}")
            logger.info(f"Estimated norm scaling factor out: {self.estimated_norm_scaling_factor_out}")

        return self.estimated_norm_scaling_factor_in.to(self.dtype), self.estimated_norm_scaling_factor_out.to(self.dtype)  

    @torch.no_grad()
    def set_norm_scaling_factor_if_needed(self):

        if self.estimated_norm_scaling_factor_in is None or self.estimated_norm_scaling_factor_out is None:
            
            # ensures all ranks consume batches (important for feature_sharding to keep same pointer)
            self.estimated_norm_scaling_factor_in, self.estimated_norm_scaling_factor_out = self.estimate_norm_scaling_factor(self.cfg.n_batches_for_norm_estimate)
            
            if self.cfg.uses_process_group:
                dist.barrier() 

                tensor_in = self.estimated_norm_scaling_factor_in.to(self.device)
                tensor_out = self.estimated_norm_scaling_factor_out.to(self.device)

                dist.broadcast(tensor_in, src=0)
                dist.broadcast(tensor_out, src=0)

                self.estimated_norm_scaling_factor_in = tensor_in
                self.estimated_norm_scaling_factor_out = tensor_out

    def apply_norm_scaling_factor_in(self, activations: torch.Tensor) -> torch.Tensor:
        if self.estimated_norm_scaling_factor_in is None:
            raise ValueError(
                "estimated_norm_scaling_factor_in is not set, call set_norm_scaling_factor_if_needed() first"
            )
        scaling = self.estimated_norm_scaling_factor_in.view(1, -1, 1)
        activations *= scaling # in-place operation not copying tensor
        return activations

    def apply_norm_scaling_factor_out(self, activations: torch.Tensor) -> torch.Tensor:
        if self.estimated_norm_scaling_factor_out is None:
            raise ValueError(
                "estimated_norm_scaling_factor_out is not set, call set_norm_scaling_factor_if_needed() first"
            )
        scaling = self.estimated_norm_scaling_factor_out.view(1, -1, 1)
        activations *= scaling # in-place operation not copying tensor
        return activations

    def remove_norm_scaling_factor_in(self, activations: torch.Tensor) -> torch.Tensor:
        if self.estimated_norm_scaling_factor_in is None:
            raise ValueError(
                "estimated_norm_scaling_factor_in is not set, call set_norm_scaling_factor_if_needed() first"
            )
        scaling = self.estimated_norm_scaling_factor_in.view(1, -1, 1)
        activations /= scaling # in-place operation not copying tensor
        return activations

    def remove_norm_scaling_factor_out(self, activations: torch.Tensor) -> torch.Tensor:
        if self.estimated_norm_scaling_factor_out is None:
            raise ValueError(
                "estimated_norm_scaling_factor_out is not set, call set_norm_scaling_factor_if_needed() first"
            )
        scaling = self.estimated_norm_scaling_factor_out.view(1, -1, 1)
        activations /=  scaling
        return activations

    # ------------------ Generate activations, save to and load from disk ------------------

    def generate_and_save_activations(self, path: str, split_count: int = 10, number_of_tokens: Optional[int] = None, split_begin_idx: int = 0, split_end_idx: Optional[int] = None, use_compression: bool = False,  compression_config: Optional['CompressionConfig'] = None,):
        """
        path - directory where splits will be saved with name activations_ctx_{self.context_size}_split_{split_idx}.safetensors
        """

        compression_store = None
        if use_compression:
            
            if compression_config is None:
                compression_config = CompressionConfig(
                    quantization="int8",
                    compression="zstd",
                    compression_level=3,
                )

            compression_store = CompressedActivationsStore(compression_config)
            logger.info(f"[ActivationsStore] Using compression: {compression_config.quantization} + {compression_config.compression}")
        # Set default value for split_end_idx
        if split_end_idx is None:
            split_end_idx = split_count

        save_path = Path(path)
        os.makedirs(save_path / f"ctx_{self.context_size}", exist_ok=True)

        buffer_size = self.cfg.store_batch_size_prompts * self.cfg.context_size * self.buffer_batches

        # Infer full token count if not provided
        if number_of_tokens is None:
            number_of_tokens = sum(len(example["tokens"]) for example in self.raw_ds)
            number_of_tokens -= number_of_tokens % buffer_size  # truncate to full buffers
            logger.info(f"[ActivationsStore] Using full dataset: {number_of_tokens} tokens")

        total_buffers = number_of_tokens // buffer_size
        usable_buffers = (total_buffers // split_count) * split_count  # drop leftovers
        buffers_per_split = usable_buffers // split_count
        split_token_count = buffers_per_split * buffer_size

        logger.info(f"[ActivationsStore] Saving {usable_buffers * buffer_size} tokens "
            f"in {split_count} splits of {buffers_per_split} buffers each")

        # Skip to the correct starting position in the dataset
        start_buffer_idx = split_begin_idx * buffers_per_split
        logger.info(f"[ActivationsStore] Skipping first {start_buffer_idx} buffers to reach split {split_begin_idx}")
        for skip_idx in range(start_buffer_idx):
            for _ in range(self.buffer_batches):
                self._next_token_batch()
            if skip_idx % 500 == 0:
                logger.info(f"[ActivationsStore] Skipped buffer {skip_idx + 1}/{start_buffer_idx}")

        buffer_idx = start_buffer_idx
        for split_idx in range(split_begin_idx, split_end_idx):
            acts_in = torch.empty((split_token_count, self.N_layers, self.cfg.d_in), dtype=self.dtype)
            acts_out = torch.empty((split_token_count, self.N_layers, self.cfg.d_in), dtype=self.dtype)
            tokens = torch.empty((split_token_count,), dtype=torch.long)

            for i in range(buffers_per_split):
                act_in_buffer, act_out_buffer, token_buffer = self._fresh_activation_batches(
                    return_tokens=True,
                    mix_with_previous_buffer=False
                )
                idx = i * buffer_size
                acts_in[idx:idx + buffer_size] = act_in_buffer
                acts_out[idx:idx + buffer_size] = act_out_buffer
                tokens[idx:idx + buffer_size] = token_buffer

                if buffer_idx % 50 == 0 or buffer_idx == usable_buffers - 1:
                    logger.info(f"[ActivationsStore] Processed buffer {buffer_idx + 1}/{usable_buffers}")

                buffer_idx += 1

            if compression_store is not None:
                # Save compressed (new format)
                split_path = save_path / f"ctx_{self.context_size}" / f"activations_ctx_{self.context_size}_split_{split_idx}.bin"
                file_size = compression_store.save_compressed(
                    path=split_path,
                    act_in=acts_in,
                    act_out=acts_out,
                    tokens=tokens,
                )
                compression_ratio = (acts_in.numel() * acts_in.element_size() * 2) / file_size  # 2 for in+out
                logger.info(f"[ActivationsStore] Saved compressed split {split_idx + 1}/{split_count}: "
                            f"{file_size / 1024 / 1024:.2f} MB ({compression_ratio:.2f}x compression)")
            else:
                # Save uncompressed (original format)
                split_path = activation_split_path(save_path, self.context_size, split_idx, must_exist=False)
                save_file({"act_in": acts_in, "act_out": acts_out, "tokens": tokens}, split_path)
                logger.info(f"[ActivationsStore] Saved split {split_idx + 1}/{split_count} to {split_path}")
            
        logger.info(f"[ActivationsStore] Finished saving all {split_count} splits to {save_path}")

    def _load_buffer_from_cached(self, return_tokens:bool = False) -> tuple[torch.Tensor, ...]:
        
        total_activations = self.cached_act_in.shape[0]

        if self.n_train_batch_per_buffer is None or self.cfg.train_batch_size_tokens is None:
            raise ValueError("n_train_batch_per_buffer and train_batch_size_tokens must not be None here")
        
        if self.cache_ptr + self.n_train_batch_per_buffer * self.cfg.train_batch_size_tokens > total_activations:
            self.leftover_activations = {
                "act_in": self.cached_act_in[self.cache_ptr:],
                "act_out": self.cached_act_out[self.cache_ptr:],
                "tokens": self.cached_tokens[self.cache_ptr:]
            }
            self._load_cached_activations()

            # TODO: FIND A SOLUTION FOR THIS, CURRENTLY IGNORING THE LEFTOVER BUFFER, CAT IS TOO SLOW

            # self.cached_act_in = torch.cat([self.leftover_activations["act_in"], self.cached_act_in], dim=0)
            # self.cached_act_out = torch.cat([self.leftover_activations["act_out"], self.cached_act_out], dim=0)
            # self.cached_tokens = torch.cat([self.leftover_activations["tokens"], self.cached_tokens], dim=0)

            if self.cached_act_in.shape[0] < self.n_train_batch_per_buffer * self.cfg.train_batch_size_tokens: 
                raise ValueError(
                    f"Buffer size {self.n_train_batch_per_buffer * self.cfg.train_batch_size_tokens} greater than split size {self.cached_act_in.shape[0]}"
                )

            self.leftover_activations = None

        start = self.cache_ptr
        end = start + self.n_train_batch_per_buffer * self.cfg.train_batch_size_tokens

        self.cache_ptr = end

        if return_tokens:
            return self.cached_act_in[start:end], self.cached_act_out[start:end], self.cached_tokens[start:end]
        else: 
            return self.cached_act_in[start:end], self.cached_act_out[start:end]

    def _load_cached_activations(self) -> None:
        if not hasattr(self, "split"):
            self.split = self.rank + 1201
            
        if self.cfg.cached_activations_path is None:
            raise ValueError("cached_activations_path must not be None here")
        
        base_path = Path(self.cfg.cached_activations_path) / f"ctx_{self.context_size}"
        compressed_path = base_path / f"activations_ctx_{self.context_size}_split_{self.split}.bin"

        if compressed_path.exists():

            compression_config = CompressionConfig(quantization="int8", compression="zstd")
            compression_store = CompressedActivationsStore(compression_config)
            
            loaded_in, loaded_out, loaded_tokens, metadata = compression_store.load_compressed(compressed_path)
            
            self.cached_act_in = loaded_in.to(self.dtype).cpu()
            self.cached_act_out = loaded_out.to(self.dtype).cpu()
            self.cached_tokens = loaded_tokens.cpu() if loaded_tokens is not None else None

            if self.rank == 0: 
                logger.info(f"[ActivationsStore] Loaded compressed split {self.split} "
                            f"with {self.cached_act_in.shape[0]} samples from {compressed_path}")
        else:
        
            activations_path = activation_split_path(self.cfg.cached_activations_path, self.context_size, self.split)
        
            # If it doesn't exist, restart from 0
            if not os.path.exists(activations_path):
                logger.info(f"[ActivationsStore] No split at {activations_path}, restarting from split {self.rank}")
                self.split = self.rank
                activations_path = activation_split_path(self.cfg.cached_activations_path, self.context_size, self.split)
                if not os.path.exists(activations_path):
                    raise FileNotFoundError(f"No cached activations found at {activations_path}")

            # if self.cfg.is_sharded:
            #     # first process loads to RAM
            #     for r in range(self.world_size):
            #         if self.rank == r:
            #             tensors = load(open(activations_path, 'rb').read())
            #             self.cached_act_in = tensors["act_in"].to(dtype=self.dtype, device="cpu")
            #             self.cached_act_out = tensors["act_out"].to(dtype=self.dtype, device="cpu")
            #             self.cached_tokens = tensors["tokens"].to(device="cpu")
            #         dist.barrier()

            # first process loads to RAM
            tensors = load(open(activations_path, 'rb').read())
            self.cached_act_in = tensors["act_in"].to(dtype=self.dtype, device="cpu")
            self.cached_act_out = tensors["act_out"].to(dtype=self.dtype, device="cpu")
            self.cached_tokens = tensors["tokens"].to(device="cpu")

            logger.info(
                f"[rank {self.rank}] Loaded uncompressed split {self.split} "
                f"with {self.cached_act_in.shape[0]} samples from {activations_path}"
            )
                
        self.split += self.world_size
        self.cache_ptr = 0

    def _advance_active_src_rank(self) -> None:
        if self.cfg.uses_process_group:
            self.active_src_rank = (self.active_src_rank + 1) % self.world_size
        else:
            self.active_src_rank = self.rank

    def _broadcast_batch_tensor(
        self,
        tensor: Optional[torch.Tensor],
        src: int,
    ) -> Optional[torch.Tensor]:
        """
        Broadcast one batch tensor from src to all ranks on GPU.
        Non-src ranks pass tensor=None.
        """
        if not self.cfg.uses_process_group:
            return tensor

        dtype_to_code = {
            torch.float32: 0,
            torch.float16: 1,
            torch.bfloat16: 2,
            torch.int64: 3,
            torch.int32: 4,
            torch.uint8: 5,
            torch.long: 3,
        }
        code_to_dtype = {v: k for k, v in dtype_to_code.items()}

        if self.rank == src:
            if tensor is None:
                raise ValueError("Source rank must provide a tensor to broadcast.")
            shape = list(tensor.shape)
            ndim = len(shape)
            dtype_code = dtype_to_code[tensor.dtype]
        else:
            ndim = 0
            dtype_code = 0
            shape = []

        max_ndim = 8
        meta = torch.full((2 + max_ndim,), -1, dtype=torch.int64, device=self.device)
        if self.rank == src:
            meta[0] = ndim
            meta[1] = dtype_code
            for i, s in enumerate(shape):
                meta[2 + i] = s

        dist.broadcast(meta, src=src)

        ndim = int(meta[0].item())
        dtype = code_to_dtype[int(meta[1].item())]
        recv_shape = tuple(int(meta[2 + i].item()) for i in range(ndim))

        if self.rank == src and tensor is not None:
            out = tensor.to(self.device, non_blocking=True)
        else:
            out = torch.empty(recv_shape, dtype=dtype, device=self.device)

        dist.broadcast(out, src=src)
        return out

    def _build_buffer_payload(self) -> dict[str, Any]:
        t0 = time.time()
        logger.info(f"[rank {self.rank}] [prefetch] build started")

        return_tokens = self.return_tokens
        use_cached = self.cfg.cached_activations_path is not None

        if use_cached:
            result = self._load_buffer_from_cached(return_tokens=return_tokens)
        else:
            result = self._fresh_activation_batches(
                return_tokens=return_tokens,
                mix_with_previous_buffer=self.mix_with_previous_buffer
            )

        if return_tokens:
            new_in, new_out, new_tokens = result
        else:
            new_in, new_out = result

        if self.mix_with_previous_buffer and self._storage_in is not None:
            all_in = torch.cat([self._storage_in, new_in], dim=0)
            all_out = torch.cat([self._storage_out, new_out], dim=0)
            if self.return_tokens:
                all_tokens = torch.cat([self._storage_tokens, new_tokens], dim=0)
        else:
            all_in, all_out = new_in, new_out
            if self.return_tokens:
                all_tokens = new_tokens

        if self.mix_with_previous_buffer:
            split = all_in.size(0) // 2
            storage_in = all_in[:split]
            storage_out = all_out[:split]
            storage_tokens = all_tokens[:split] if self.return_tokens else None
        else:
            split = 0
            storage_in = None
            storage_out = None
            storage_tokens = None

        if self.return_tokens:
            dataset = TensorDataset(all_in[split:], all_out[split:], all_tokens[split:])
        else:
            dataset = TensorDataset(all_in[split:], all_out[split:])

        g = torch.Generator()
        g.manual_seed(42 + self.buffer_counter)

        loader = DataLoader(
            dataset,
            batch_size=self.cfg.train_batch_size_tokens,
            drop_last=True,
            pin_memory=True,
            generator=g,
            shuffle=self.shuffle,
        )

        logger.info(f"[rank {self.rank}] [prefetch] build finished in {time.time() - t0:.2f}s")

        return {
            "yield_iter": iter(loader),
            "storage_in": storage_in,
            "storage_out": storage_out,
            "storage_tokens": storage_tokens,
        }

    def prefetch_status(self) -> dict[str, Any]:
        if self._prefetch_future is None:
            return {"status": "not_started"}

        return {
            "done": self._prefetch_future.done(),
            "running": self._prefetch_future.running(),
            "cancelled": self._prefetch_future.cancelled(),
            "exception": repr(self._prefetch_future.exception()) if self._prefetch_future.done() else None,
        }

    def _install_buffer_payload(self, payload: dict[str, Any]) -> None:
        self._yield_iter = payload["yield_iter"]
        self._storage_in = payload["storage_in"]
        self._storage_out = payload["storage_out"]
        self._storage_tokens = payload["storage_tokens"]

    def _rebuild_buffers(self) -> None:
        payload = self._build_buffer_payload()
        self._install_buffer_payload(payload)

    def _start_prefetch(self) -> None:
        if self._prefetch_executor is None:
            return
        if self._prefetch_future is None:
            logger.info(f"[rank {self.rank}] [prefetch] submitting next buffer build")
            self._prefetch_future = self._prefetch_executor.submit(self._build_buffer_payload)

    def _use_prefetched_buffer(self) -> None:
        """
        Promote this rank's prefetched next local buffer to current,
        then start prefetching the following one.
        """
        if self._prefetch_future is None:
            self._rebuild_buffers()
        else:
            payload = self._prefetch_future.result()
            self._install_buffer_payload(payload)
            self._prefetch_future = None

        self.buffer_counter += 1
        self._start_prefetch()

    def _broadcast_has_batch(self, has_batch: Optional[bool], src: int) -> bool:
        if not self.cfg.uses_process_group:
            if has_batch is None:
                raise ValueError("has_batch must be provided when not distributed")
            return has_batch

        if self.rank == src:
            if has_batch is None:
                raise ValueError("Source rank must provide has_batch")
            flag = torch.tensor([1 if has_batch else 0], dtype=torch.int64, device=self.device)
        else:
            flag = torch.empty(1, dtype=torch.int64, device=self.device)

        dist.broadcast(flag, src=src)
        return bool(flag.item())
            
    # ───────────────────  public iterator  ───────────────────

    def __iter__(self):
        while True:
            if self._yield_iter is None:
                self._rebuild_buffers()

            src = self.active_src_rank

            if self.rank == src:
                try:
                    batch = next(self._yield_iter)
                    has_batch = True
                except StopIteration:
                    batch = None
                    has_batch = False
            else:
                batch = None
                has_batch = None

            has_batch = self._broadcast_has_batch(has_batch, src)

            if not has_batch:
                if self.rank == src:
                    if self.cfg.cached_activations_path is not None:
                        self._use_prefetched_buffer()
                    else:
                        self._rebuild_buffers()
                        self.buffer_counter += 1

                self._advance_active_src_rank()
                continue

            if self.rank == src:
                if self.return_tokens:
                    token_batch_cpu, act_in_cpu, act_out_cpu = batch[2], batch[0], batch[1]

                    token_batch = token_batch_cpu.to(self.device, non_blocking=False)
                    act_in = act_in_cpu.to(self.device, non_blocking=False)
                    act_out = act_out_cpu.to(self.device, non_blocking=False)

                    token_batch = self._broadcast_batch_tensor(token_batch, src)
                    act_in = self._broadcast_batch_tensor(act_in, src)
                    act_out = self._broadcast_batch_tensor(act_out, src)
                else:
                    act_in_cpu, act_out_cpu = batch[0], batch[1]

                    act_in = act_in_cpu.to(self.device, non_blocking=False)
                    act_out = act_out_cpu.to(self.device, non_blocking=False)

                    act_in = self._broadcast_batch_tensor(act_in, src)
                    act_out = self._broadcast_batch_tensor(act_out, src)
            else:
                if self.return_tokens:
                    token_batch = self._broadcast_batch_tensor(None, src)
                    act_in = self._broadcast_batch_tensor(None, src)
                    act_out = self._broadcast_batch_tensor(None, src)
                else:
                    act_in = self._broadcast_batch_tensor(None, src)
                    act_out = self._broadcast_batch_tensor(None, src)

            if not self._skip_norm_application:
                act_in = self.apply_norm_scaling_factor_in(act_in)
                act_out = self.apply_norm_scaling_factor_out(act_out)

            if self.return_tokens:
                yield token_batch, act_in, act_out
            else:
                yield act_in, act_out

def load_dataset_auto(
    path_or_name: str,
    split: str = "train",
    disk: bool = False
):     
    if disk:
        return load_from_disk(path_or_name)

    return load_dataset(path_or_name, split=split)

def load_dataset_multilingual(
    path_or_name: str
): 
    ds_dict = load_dataset(path_or_name)
    datasets_with_lang = []
    for split_name, ds in ds_dict.items():
        ds = ds.add_column("language", [split_name] * len(ds))
        datasets_with_lang.append(ds)
    return datasets.concatenate_datasets(datasets_with_lang)
    
# def load_dataset_auto(path_or_name: str, split: str = "train", is_multilingual_split_dataset: bool = False):
#     if os.path.exists(path_or_name):
#         logger.info("Loading from disk")

#         # return load_from_disk(path_or_name)

#         return load_dataset(
#             path_or_name,
#             split="train",
#             cache_dir=os.path.expanduser("~/.cache/huggingface/datasets"),
#             keep_in_memory=False
#         )
#     else:
#         # For the multilingual dataset
#         if is_multilingual_split_dataset:

#             logger.info(f"Loading dataset {path_or_name}...")
#             start_time = time.time()
            
#             def progress_callback(info):
#                 elapsed = time.time() - start_time
#                 logger.info(f"Loading... {elapsed:.1f}s elapsed")
            
#             ds_dict = load_dataset(path_or_name)
#             elapsed = time.time() - start_time
#             logger.info(f"Dataset loaded successfully in {elapsed:.1f}s: {list(ds_dict.keys())}")
            
#             # Add language column based on split name before concatenation
#             datasets_with_lang = []
#             for split_name, ds in ds_dict.items():
#                 ds = ds.add_column("language", [split_name] * len(ds))
#                 datasets_with_lang.append(ds)
            
#             # Concatenate all splits into one Dataset
#             return datasets.concatenate_datasets(datasets_with_lang)
#         else:
#             logger.info("Loading from hub")
#             return load_dataset(path_or_name, split=split)
            
def _filter_buffer_acts(
    buffer: tuple[torch.Tensor, torch.Tensor | None],
    exclude_tokens: torch.Tensor | None,
) -> torch.Tensor:
    """
    Filter out activations for tokens that are in exclude_tokens.
    """

    activations, tokens = buffer
    if tokens is None or exclude_tokens is None:
        return activations

    mask = torch.isin(tokens, exclude_tokens)
    return activations[~mask]
