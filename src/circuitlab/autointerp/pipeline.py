# TODO I commented out for mypy and ruff

# import torch
# import os
# import json
# import heapq
# from pathlib import Path
# from safetensors.torch import load_file
# from typing import List, Optional, Dict
# from safetensors.torch import save_file
# import random 

# from circuitlab.config import AutoInterpConfig
# from circuitlab.utils import LatentCache_CFG_FILENAME, PROMPTS_FOLDERNAME, EXPLANATIONS_FOLDERNAME, DICT_FOLDERNAME
# from circuitlab.clt import CLT
# from circuitlab.load_model import load_model
# from circuitlab.training.activations_store import ActivationsStore
# from circuitlab import logger
# from circuitlab.autointerp.client import run_client
# from circuitlab.autointerp.prompt import generate_prompt
# from circuitlab.autointerp.prompt_multilingual import generate_prompt_multilingual
# from circuitlab.transformer_lens.multilingual_patching import patch_official_model_names, patch_convert_hf_model_config

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TOP_K_DEFAULT = 100
# N_TOP_ACTIVATING_TOKENS_SHOWN = 4

# class AutoInterp:
#     def __init__(self, cfg: AutoInterpConfig):
#         self.cfg = cfg
#         self.total_autointerp_tokens = cfg.total_autointerp_tokens
#         self.device = torch.device(self.cfg.device)
#         self.ctx = cfg.context_size 

#         patch_official_model_names()
#         patch_convert_hf_model_config()

#         self.model = load_model(
#             self.cfg.model_class_name,
#             self.cfg.model_name,
#             device=self.device,
#             model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
#         )

#         self.clt = CLT.load_from_pretrained(
#             self.cfg.clt_path, self.cfg.device
#         )

#         self.clt = self.clt.to(self.device)

#         self.activations_store = ActivationsStore(
#             self.model,
#             self.cfg,
#             estimated_norm_scaling_factor_in=self.clt.estimated_norm_scaling_factor_in, 
#             estimated_norm_scaling_factor_out=self.clt.estimated_norm_scaling_factor_out
#         )

#     def run(self, chunk_list: Optional[List[int]] = None):
#         n_tokens = 0
#         chunk_id = 0
#         current_chunk_acts = []
#         current_chunk_tokens = []
#         current_chunk_languages: list[str] = []  # Track languages for current chunk
#         current_chunk_size = 0
#         global_sequence_count = 0  # Track global sequence position across all chunks
        
#         save_dir = Path(self.cfg.latent_cache_path or "")
#         save_dir.mkdir(parents=True, exist_ok=True)
        
#         # Calculate chunk size based on total tokens and desired number of chunks
#         chunk_size = self.cfg.total_autointerp_tokens // self.cfg.n_chunks
#         print("number of chunks:", self.cfg.n_chunks)
#         print("chunk size:", chunk_size)
        
#         # Remove leftover chunk files from previous runs that might have exceeded current n_chunks
#         if save_dir.exists():
#             for layer_dir in save_dir.glob("layer_*"):
#                 if layer_dir.is_dir():
#                     for chunk_file in layer_dir.glob("chunk_*.safetensors"):
#                         chunk_id_str = chunk_file.stem.split("_")[-1]
#                         try:
#                             old_chunk_id = int(chunk_id_str)
#                             if old_chunk_id >= self.cfg.n_chunks:
#                                 chunk_file.unlink()
#                                 print(f"Removed old chunk file: {chunk_file}")
#                         except ValueError:
#                             pass 

#         while n_tokens < self.cfg.total_autointerp_tokens:
            
#             # Check if we should process this chunk or skip it
#             should_process_chunk = chunk_list is None or chunk_id in chunk_list
            
#             if should_process_chunk:
#                 tokens, acts_in, _ = next(iter(self.activations_store))
#                 acts_in_gpu = acts_in.to(self.cfg.device).to(self.activations_store.dtype)
#                 # acts_out_gpu = acts_out.to(self.cfg.device).to(self.activations_store.dtype)
#                 feat_act, _ = self.clt.encode(acts_in_gpu)
#                 # put feat_act to 0 if smaller than 1
#                 # feat_act[feat_act < 1] = 0
#                 # reconstruction = self.clt.decode(feat_act)

#                 # mse = ((reconstruction - acts_out_gpu) ** 2).sum(dim=-1).mean(0)
#                 # variance = ((acts_out_gpu - acts_out_gpu.mean(dim=0, keepdim=True)) ** 2).sum(dim=-1).mean(0)
#                 # print(f"Normalized MSE: {mse / variance}", flush=True)

#                 # feat_act_value = feat_act[feat_act > 0]
#                 # print(f"Mean activation: {feat_act_value.tolist()}")

#                 # feat_act_last_layer = feat_act[:, -1, :]
#                 # feat_act_last_layer_value = feat_act_last_layer[feat_act_last_layer > 0]

#                 # print(f"Mean activation: {feat_act_last_layer_value.tolist()}")

#                 # print(f"Mean l0: {(feat_act > 0).float().sum(dim=-1).mean()}")

#                 current_chunk_size += feat_act.shape[0]
                
#                 reshaped_acts, reshaped_tokens = self._reshape_latents(feat_act.cpu(), tokens.cpu())
                
#                 if reshaped_acts.shape[0] > 0:
#                     current_chunk_acts.append(reshaped_acts)
#                     current_chunk_tokens.append(reshaped_tokens)
                    
#                     # Collect language info if available
#                     if ("CausalNLP" in self.cfg.model_name and 
#                         hasattr(self.activations_store, 'runtime_doc_languages') and
#                         self.activations_store.runtime_doc_languages):
                        
#                         # Get corresponding languages for this batch of sequences
#                         n_sequences = reshaped_acts.shape[0]
#                         # Use global sequence count to correctly index into runtime_doc_languages
#                         batch_languages = self.activations_store.runtime_doc_languages[global_sequence_count:global_sequence_count + n_sequences]
#                         current_chunk_languages.extend(batch_languages)
                
#                 # Update global sequence count for both processed and skipped chunks
#                 n_sequences = reshaped_acts.shape[0] if reshaped_acts.shape[0] > 0 else 0
#                 global_sequence_count += n_sequences
                
#                 n_tokens += feat_act.shape[0]
#                 del acts_in_gpu, feat_act, reshaped_acts, reshaped_tokens
#             else:
#                 # Skip processing by advancing token iterator only
#                 skipped_tokens = self.activations_store._skip_batches()
#                 n_tokens += len(skipped_tokens)
#                 current_chunk_size += len(skipped_tokens)
                
#                 # Update global sequence count for skipped sequences
#                 # Calculate number of sequences that would have been created from skipped tokens
#                 n_skipped_sequences = len(skipped_tokens) // self.ctx
#                 global_sequence_count += n_skipped_sequences
            
#             print(n_tokens, "processed", flush=True)
            
#             if current_chunk_size >= chunk_size or n_tokens >= self.cfg.total_autointerp_tokens:
#                 if current_chunk_acts:
#                     print(f"chunk{chunk_id} saved", flush=True)
#                     combined_acts = torch.cat(current_chunk_acts, dim=0)
#                     combined_tokens = torch.cat(current_chunk_tokens, dim=0)

#                     print(f"Chunk {chunk_id}, sparsity: {(combined_acts > 0).float().sum(-1).mean(0).mean(1)}")
                    
#                     # Save by layer
#                     for layer in range(combined_acts.shape[2]):
#                         layer_dir = save_dir / f"layer_{layer}"

#                         layer_dir.mkdir(exist_ok=True)
                        
#                         layer_data = {
#                             "activations": combined_acts[:, :, layer, :].half(),
#                             "tokens": combined_tokens
#                         }
                        
#                         chunk_file = layer_dir / f"chunk_{chunk_id}.safetensors"
#                         save_file(layer_data, chunk_file)
                        
#                         # Save language info separately as JSON if available
#                         if ("CausalNLP" in self.cfg.model_name and current_chunk_languages):
#                             lang_file = layer_dir / f"chunk_{chunk_id}_languages.json"
#                             with open(lang_file, 'w') as f:
#                                 json.dump({"languages": current_chunk_languages}, f)
                    
#                     logger.info(f"Saved chunk {chunk_id} with {combined_acts.shape[0]} sequences")
#                     del combined_acts, combined_tokens
#                 else:
#                     print(chunk_id, "skipped", flush=True)
                
#                 chunk_id += 1
#                 current_chunk_acts = []
#                 current_chunk_tokens = []
#                 current_chunk_languages = []  # Reset language tracking
#                 current_chunk_size = 0
                
#                 # Early exit if we've processed all desired chunks
#                 if chunk_list is not None and chunk_id > max(chunk_list):
#                     logger.info("Finished processing all chunks in chunk_list. Stopping early.")
#                     break
            
#             torch.cuda.empty_cache()

#         cfg_path = Path(self.cfg.latent_cache_path or "") / LatentCache_CFG_FILENAME
#         with open(cfg_path, "w") as f:
#             json.dump(self.cfg.to_dict(), f)
#         logger.info(f"Latent cache config saved to: {cfg_path}")

#     def _reshape_latents(self, feat_acts, tokens):
#         N = feat_acts.shape[0]
#         ctx = self.ctx
#         if N % ctx != 0:
#             excess = N % ctx
#             feat_acts = feat_acts[:-excess]
#             tokens = tokens[:-excess]

#         if feat_acts.shape[0] == 0:
#             return feat_acts, tokens

#         N_seq = feat_acts.shape[0] // ctx
#         feat_acts = feat_acts.view(N_seq, ctx, *feat_acts.shape[1:])
#         tokens = tokens.view(N_seq, ctx)
#         return feat_acts, tokens

#     def save_feature_frequency(self):
#         feature_frequency = {}
#         for layer in range(self.model.cfg.n_layers):
#             layer_dir = Path(self.cfg.latent_cache_path) / f"layer_{layer}"

#             # Find all chunk files
#             chunk_files = sorted(layer_dir.glob("chunk_*.safetensors"))
#             if not chunk_files:
#                 print(f"No chunks found for layer {layer}", flush=True)
#                 continue

#             chunk_files = chunk_files[:1]

#             print(f"Processing layer {layer}: {len(chunk_files)} chunks", flush=True)

#             # Accumulate statistics across all chunks
#             total_sum = None
#             total_count = 0

#             for chunk_file in chunk_files:
#                 data = load_file(chunk_file)
#                 activations = data["activations"]  # [N_seq, ctx, d_latent]

#                 # Sum of active features
#                 active_sum = (activations > 0).float().sum(dim=(0, 1))  # [d_latent]

#                 if total_sum is None:
#                     total_sum = active_sum
#                 else:
#                     total_sum += active_sum

#                 # Count total positions
#                 total_count += activations.shape[0] * activations.shape[1]

#             # Compute mean frequency
#             feature_frequency[layer] = (total_sum / total_count).cpu().tolist()

#         # Save the results
#         with open(os.path.join(self.cfg.latent_cache_path, "feature_frequency.json"), "w") as f:
#             json.dump(feature_frequency, f)
#         print(f"Feature frequency saved to {os.path.join(self.cfg.latent_cache_path, 'feature_frequency.json')}", flush=True)


#     def get_top_activating_sequences(self, layer: int, top_k: int = TOP_K_DEFAULT, index_list: Optional[List[int]] = None, compute_averages: bool = True, compute_language_distributions: bool = True):
#         if self.cfg.latent_cache_path is None: 
#             raise ValueError("latent_cache_path must not be none here.")

#         layer_dir = Path(self.cfg.latent_cache_path) / f"layer_{layer}"
#         chunk_files = sorted(layer_dir.glob("chunk_*.safetensors"))
        
#         if not chunk_files:
#             raise FileNotFoundError(f"No chunk files found in {layer_dir}")
        
#         d_latent: Optional[int] = None
#         global_top_sequences: Dict[int, List[Dict]] = {}
#         feature_stats: Dict[int, Dict[str, float]] | None = {} if compute_averages else None
#         # Track general language distributions across all chunks for each feature
#         general_language_distributions: Dict[int, Dict[str, int]] | None = {} if compute_language_distributions else None
        
#         # Use heaps for O(log k) insertions instead of O(k) sorting
#         use_heap = top_k < 200  # Heap is more efficient for smaller k
        
#         for chunk_file in chunk_files:

#             print(f"chunk file: {chunk_file}", flush=True)
#             data = load_file(chunk_file)
#             activations = data["activations"]  # [N_seq, ctx, d_latent]
#             tokens = data["tokens"]  # [N_seq, ctx]
            
#             # Load language data from separate JSON file if available
#             languages = None
#             lang_file = chunk_file.parent / f"{chunk_file.stem}_languages.json"
#             if lang_file.exists():
#                 with open(lang_file, 'r') as f:
#                     lang_data = json.load(f)
#                     languages = lang_data.get("languages", None)
            
#             print(f"finished loading: {chunk_file}", flush=True)

#             if d_latent is None:
#                 d_latent = activations.shape[-1]
#                 for feat_idx in range(d_latent):
#                     global_top_sequences[feat_idx] = []
#                     if compute_averages and feature_stats is not None:
#                         feature_stats[feat_idx] = {'total_activation': 0.0, 'count': 0}
#                     if compute_language_distributions and general_language_distributions is not None:
#                         general_language_distributions[feat_idx] = {}
            
#             max_vals = activations.max(dim=1).values  # [N_seq, d_latent]
            
#             # Pre-filter features if index_list is provided
#             if index_list is not None:
#                 feat_indices = torch.tensor(index_list, device=max_vals.device)
#                 filtered_max_vals = max_vals[:, feat_indices]
#                 filtered_activations = activations[:, :, feat_indices]
#                 effective_indices = index_list
#             else:
#                 filtered_max_vals = max_vals
#                 filtered_activations = activations
#                 effective_indices = list(range(d_latent))
            
#             if compute_averages and feature_stats is not None:
#                 positive_mask = filtered_max_vals > 0
#                 for i, feat_idx in enumerate(effective_indices):
#                     feat_positive = positive_mask[:, i]
#                     if feat_positive.any():
#                         feature_stats[feat_idx]['total_activation'] += filtered_max_vals[feat_positive, i].sum().item()
#                         feature_stats[feat_idx]['count'] += feat_positive.sum().item()
            
#             # Track general language distribution for all sequences with positive activation
#             if (compute_language_distributions and general_language_distributions is not None 
#                 and languages is not None):
#                 positive_mask = filtered_max_vals > 0
#                 for i, feat_idx in enumerate(effective_indices):
#                     feat_positive = positive_mask[:, i]
#                     if feat_positive.any():
#                         # Get languages for sequences with positive activation for this feature
#                         positive_seq_indices = feat_positive.nonzero().flatten()
#                         for seq_idx in positive_seq_indices:
#                             seq_idx_int = seq_idx.item()
#                             if seq_idx_int < len(languages):
#                                 lang = languages[seq_idx_int]
#                                 if lang not in general_language_distributions[feat_idx]:
#                                     general_language_distributions[feat_idx][lang] = 0
#                                 general_language_distributions[feat_idx][lang] += 1
            
#             for i, feat_idx in enumerate(effective_indices):
#                 feat_max_vals = filtered_max_vals[:, i]
                
#                 current_size = len(global_top_sequences[feat_idx])
#                 available_slots = max(0, top_k * 2 - current_size)  # Allow 2x buffer
                
#                 if available_slots > 0:

#                     n_take = min(available_slots, feat_max_vals.shape[0])
#                     if n_take < feat_max_vals.shape[0]:
#                         top_indices = torch.topk(feat_max_vals, n_take).indices
#                     else:
#                         top_indices = torch.arange(feat_max_vals.shape[0])
                    
#                     # Create sequence data only for top sequences with positive activation
#                     for seq_idx in top_indices:
#                         seq_idx = seq_idx.item()
#                         max_val = feat_max_vals[seq_idx].item()
#                         # Skip sequences with non-positive max activation
#                         if max_val > 0:
#                             seq_data = {
#                                 'activations': filtered_activations[seq_idx, :, i],
#                                 'tokens': tokens[seq_idx, :],
#                                 'max_val': max_val,
#                                 'original_seq_idx': seq_idx  # Store original sequence index for language mapping
#                             }
                            
#                             # Add language info if available
#                             if languages is not None and seq_idx < len(languages):
#                                 seq_data['language'] = languages[seq_idx]

#                             global_top_sequences[feat_idx].append(seq_data)
                
#                 if len(global_top_sequences[feat_idx]) >= top_k * 2:
#                     if use_heap:
#                         # Convert to heap and maintain top-k (ensure scalar values)
#                         heap_data = [(-float(item['max_val']), idx, item) for idx, item in enumerate(global_top_sequences[feat_idx])]
#                         heapq.heapify(heap_data)
#                         global_top_sequences[feat_idx] = [heapq.heappop(heap_data)[2] for _ in range(min(top_k, len(heap_data)))]
#                     else:
#                         # Use torch.topk for larger k values
#                         all_vals = torch.tensor([float(item['max_val']) for item in global_top_sequences[feat_idx]])
#                         top_indices = torch.topk(all_vals, top_k).indices
#                         global_top_sequences[feat_idx] = [global_top_sequences[feat_idx][idx] for idx in top_indices.tolist()]
                                    
#         # Clean up chunk data after processing
#         del activations, tokens, data, max_vals
#         if index_list is not None:
#             del feat_indices, filtered_max_vals, filtered_activations
        
#         # Final sort, filter non-positive values, and limit to top_k
#         if d_latent is not None:
#             for feat_idx in range(d_latent):
#                 if index_list is not None and feat_idx not in index_list:
#                     continue
                
#                 # Filter out sequences with non-positive max_val and sort
#                 positive_sequences = [seq for seq in global_top_sequences[feat_idx] if seq['max_val'] > 0]
#                 positive_sequences.sort(key=lambda x: x['max_val'], reverse=True)
#                 global_top_sequences[feat_idx] = positive_sequences[:top_k]
                
#         self._stored_top_sequences = global_top_sequences

#         # Convert general language distributions to percentages
#         general_lang_percentages = None
#         if compute_language_distributions and general_language_distributions is not None and d_latent is not None:
#             general_lang_percentages = {}
#             for feat_idx in range(d_latent):
#                 if index_list is not None and feat_idx not in index_list:
#                     continue
                
#                 lang_counts = general_language_distributions[feat_idx]
#                 total_count = sum(lang_counts.values())
#                 if total_count > 0:
#                     general_lang_percentages[feat_idx] = {
#                         lang: count / total_count for lang, count in lang_counts.items()
#                     }
#                 else:
#                     general_lang_percentages[feat_idx] = {}

#         if compute_averages and feature_stats is not None and d_latent is not None:
#             feature_averages = {}
#             for feat_idx in range(d_latent):
#                 stats = feature_stats[feat_idx]
#                 avg_activation = stats['total_activation'] / stats['count'] if stats['count'] > 0 else 0.0
#                 feature_averages[feat_idx] = avg_activation

#             self._stored_feature_averages = feature_averages
        
#         # Store general language distributions for later use
#         if general_lang_percentages is not None:
#             self._stored_general_language_distributions = general_lang_percentages
        
#         # Return values based on what was computed
#         result = [global_top_sequences]
        
#         if compute_averages and feature_stats is not None:
#             result.append(feature_averages)
        
#         if compute_language_distributions and general_lang_percentages is not None:
#             result.append(general_lang_percentages)
        
#         return tuple(result) if len(result) > 1 else result[0]
    
#     def generate_prompts_for_layer(self, layer: int, top_k: int = TOP_K_DEFAULT, index_list: Optional[List[int]] = None):
#         print("starting top activation list", flush=True)
#         compute_lang_dist = "causal" in self.cfg.model_name
#         result = self.get_top_activating_sequences(layer, top_k, index_list, compute_language_distributions=compute_lang_dist)
#         top_sequences = result[0]  # First element is always top_sequences
#         d_latent = len(top_sequences)
        
#         if self.cfg.latent_cache_path is None: 
#             raise ValueError("latent_cache_path must not be none here.")
#         prompt_dir = Path(self.cfg.latent_cache_path) / PROMPTS_FOLDERNAME / f"layer{layer}"
#         prompt_dir.mkdir(parents=True, exist_ok=True)
        
#         tokenizer = self.model.tokenizer

#         print("starting the iteration", flush = True)
#         print("index list:")

#         for feat_idx in range(d_latent):
#             # Skip features not in the index_list if index_list is provided
#             if index_list is not None and feat_idx not in index_list:
#                 continue
#             print(f"Generating prompt for feature: {feat_idx}", flush=True)

#             sequences = top_sequences[feat_idx]
            
#             # Always sample exactly 50% of top_k: keep first 10% + randomly sample to reach 50% total
#             target_size = top_k // 2  # Exactly 50% of top_k
#             if len(sequences) >= target_size:
#                 top_10_percent_size = max(1, target_size // 5)  # 10% of target (which is 5% of top_k)
                
#                 # Keep top 10% of target
#                 sampled_sequences = sequences[:top_10_percent_size]
                
#                 # Randomly sample from remaining to reach exactly 50% of top_k
#                 remaining_needed = target_size - top_10_percent_size
#                 if remaining_needed > 0 and len(sequences) > top_10_percent_size:
#                     remaining_sequences = sequences[top_10_percent_size:]
#                     remaining_indices = sorted(random.sample(range(len(remaining_sequences)), 
#                                                            min(remaining_needed, len(remaining_sequences))))
#                     sampled_sequences.extend([remaining_sequences[i] for i in remaining_indices])
#             else:
#                 sampled_sequences = sequences

#             top_texts = []
#             for seq_data in sampled_sequences:
#                 tokens = seq_data['tokens'][1:]  # Remove BOS
#                 activations = seq_data['activations'][1:]  # Remove BOS
#                 highlighted_text = highlight_activations(tokens, activations, tokenizer)
#                 top_texts.append(highlighted_text)
            
#             if "CausalNLP" in self.cfg.model_name: 
#                 prompt = generate_prompt_multilingual(top_texts, layer, feat_idx)
#             else: 
#                 prompt = generate_prompt(top_texts, layer, feat_idx)

#             explanation_path = prompt_dir / f"explanation_{feat_idx}.txt"
#             with open(explanation_path, "w") as f:
#                 f.write(prompt)

#         self._stored_top_sequences = top_sequences
#         logger.info(f"Generated {d_latent} prompts for layer {layer}")

#     def generate_explanations_from_prompts(self, layer: int, index_list: Optional[List[int]] = None):
#         if self.cfg.latent_cache_path is None: 
#             raise ValueError("latent_cache_path must not be none here.")
        
#         PROMPT_DIR = Path(self.cfg.latent_cache_path) / PROMPTS_FOLDERNAME / f"layer{layer}"
#         OUT_DIR = Path(self.cfg.latent_cache_path) / EXPLANATIONS_FOLDERNAME / f"layer{layer}"
#         OUT_DIR.mkdir(parents=True, exist_ok=True)

#         # Filter prompts based on index_list if provided
#         all_prompts = sorted(PROMPT_DIR.glob("*.txt"))
#         if index_list is not None:
#             filtered_prompts = [
#                 prompt for prompt in all_prompts
#                 if int(prompt.stem.split("_")[-1]) in index_list
#             ]
#         else:
#             filtered_prompts = all_prompts

#         logger.info(f"Found {len(filtered_prompts)} prompts in {PROMPT_DIR} matching index list")

#         print(filtered_prompts, flush=True)

#         run_client(
#             prompts=filtered_prompts, 
#             out_dir=OUT_DIR, 
#             vllm_model=self.cfg.vllm_model, 
#             vllm_max_tokens=self.cfg.vllm_max_tokens
#         )

#     def generate_feature_dictionaries(self, layer: int,  index_list: Optional[List[int]] = None):
#         if self.cfg.latent_cache_path is None: 
#             raise ValueError("latent_cache_path must not be none here.")
        
#         explanations_dir = Path(self.cfg.latent_cache_path) / EXPLANATIONS_FOLDERNAME / f"layer{layer}"
#         out_dir = Path(self.cfg.latent_cache_path) / DICT_FOLDERNAME / f"layer{layer}"
#         out_dir.mkdir(parents=True, exist_ok=True)

#         if hasattr(self, '_stored_top_sequences'):
#             top_sequences = self._stored_top_sequences
#             feature_averages = self._stored_feature_averages
#             general_language_distributions = getattr(self, '_stored_general_language_distributions', None)
#         else:
#             logger.warning("No stored top sequences found, loading again...")
#             compute_lang_dist = "CausalNLP" in self.cfg.model_name ### TODO: remove this 
#             result = self.get_top_activating_sequences(layer, top_k=TOP_K_DEFAULT, index_list=index_list, compute_language_distributions=compute_lang_dist)
            
#             if isinstance(result, tuple):
#                 top_sequences = result[0]
#                 feature_averages = result[1] if len(result) > 1 else None
#                 general_language_distributions = result[2] if len(result) > 2 else None
#             else:
#                 top_sequences = result
#                 feature_averages = None
#                 general_language_distributions = None

#         d_latent = len(top_sequences)
#         tokenizer = self.model.tokenizer
#         comprehensive_explanations = {}
        
#         for feat_idx in range(d_latent):
#             # Skip features not in the index_list if index_list is provided
#             if index_list is not None and feat_idx not in index_list:
#                 continue

#             explanation_file = explanations_dir / f"explanation_{feat_idx}.txt"
#             if explanation_file.exists():
#                 raw_explanation = explanation_file.read_text().strip()
#                 description, explanation = self._parse_explanation(raw_explanation)
#             else:
#                 description, explanation = "Unknown", "No explanation generated"
#                 raw_explanation = ""

#             sequences = top_sequences[feat_idx]
            
#             top_examples = []
#             for seq_data in sequences:
#                 tokens = seq_data['tokens'][1:]  # Remove BOS
#                 activations = seq_data['activations'][1:]  # Remove BOS
#                 highlighted_text = highlight_activations(
#                     tokens, 
#                     activations, 
#                     tokenizer,
#                     threshold_ratio=0.6
#                 )
#                 top_examples.append(highlighted_text)

#             if sequences: 
#                 # Get top activating tokens from the top sequences only
#                 top_activating_tokens = self._get_top_activating_tokens_from_sequences(
#                     sequences, tokenizer, top_k=N_TOP_ACTIVATING_TOKENS_SHOWN
#                 )
#             else:
#                 # Dead feature - no sequences
#                 top_activating_tokens = []
            
#             # Convert tensors to lists for JSON serialization  
#             sequences_serializable = []
#             for seq_data in sequences:
#                 sequences_serializable.append({
#                     'activations': seq_data['activations'].tolist(),
#                     'tokens': seq_data['tokens'].tolist(),
#                     'max_val': seq_data['max_val']
#                 })
            
#             feature_dict = {
#                 "layer": layer,
#                 "feature_index": feat_idx,
#                 "description": description,
#                 "explanation": explanation,
#                 "top_examples": top_examples,
#                 "top_examples_tks": sequences_serializable,
#                 "average_activation": float(feature_averages[feat_idx]) if feature_averages else 0.0,
#                 "top_activating_tokens": top_activating_tokens,
#                 "raw_explanation": raw_explanation
#             }
                        
#             if ("CausalNLP" in self.cfg.model_name) and sequences:
#                 # Top sequences language distribution (existing behavior)
#                 lang_counts: dict[str, int] = {}
#                 for i, seq_data in enumerate(sequences):
#                     print(seq_data, flush=True)
#                     lang = seq_data['language']
#                     lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
#                 total_count = sum(lang_counts.values())
#                 lang_dist = {lang: count / total_count for lang, count in lang_counts.items()}
#                 feature_dict["language_distribution"] = lang_dist
                
#                 # General language distribution across all chunks (new behavior)
#                 if general_language_distributions and feat_idx in general_language_distributions:
#                     feature_dict["general_language_distribution"] = general_language_distributions[feat_idx]
#                 else:
#                     feature_dict["general_language_distribution"] = {}
            
#             comprehensive_explanations[feat_idx] = feature_dict
            
#             feature_file = out_dir / f"feature_{feat_idx}_complete.json"
#             with open(feature_file, 'w') as f:
#                 json.dump(feature_dict, f, indent=2)
                
#         # Delete cache files
#         if index_list:
#             last_idx = max(index_list)
#             cache_dir = Path(self.cfg.latent_cache_path) / "stored_cache"
#             seq_file = cache_dir / f"sequences_layer_{layer}_feat_{last_idx}.pkl"
#             avg_file = cache_dir / f"averages_layer_{layer}_feat_{last_idx}.pkl"
            
#             if seq_file.exists():
#                 seq_file.unlink()
#             if avg_file.exists():
#                 avg_file.unlink()

#     def scoring(self, layer: int, index_list: Optional[List[int]] = None): 
#         if self.cfg.latent_cache_path is None: 
#             raise ValueError("latent_cache_path must not be none here.")
        
#         out_dir = Path(self.cfg.latent_cache_path) / DICT_FOLDERNAME / f"layer{layer}"
#         out_dir.mkdir(parents=True, exist_ok=True)

#         for feat_idx in range(self.clt.d_latent):

#             if index_list is not None and feat_idx not in index_list: 
#                 continue

#             # Load the dictionary 
#             dict_file = out_dir / f"feature_{feat_idx}_complete.json"
#             if dict_file.exists():
#                 with open(dict_file, 'r') as f:
#                     feature_dict = json.load(f)

#             # Get the top activating sequences 
#             top_sequences = feature_dict.get("top_examples")

#             # Sample half for prompts, and half for testing
#             len(top_sequences) // 2  # Calculate target size but don't store in unused variable


#     def _parse_explanation(self, raw_text: str) -> tuple[str, str]:
#         # For simplified format, the entire raw_text is both description and explanation
#         explanation = raw_text.strip()
#         description = explanation  # Use the whole explanation as description too
        
#         return description, explanation

#     def _get_top_activating_tokens_from_sequences(
#         self, 
#         sequences: list, 
#         tokenizer,
#         top_k: int = 3
#     ) -> list[dict]:
#         all_tokens = []
#         all_activations = []
        
#         for seq_data in sequences:
#             tokens = seq_data['tokens'][1:]  # Remove BOS
#             activations = seq_data['activations'][1:]  # Remove BOS
#             all_tokens.append(tokens)
#             all_activations.append(activations)
        
#         # Combine all tokens and activations
#         combined_tokens = torch.cat(all_tokens)
#         combined_activations = torch.cat(all_activations)
        
#         threshold = combined_activations.max() * 0.6 
#         high_activation_mask = combined_activations > threshold
        
#         activating_tokens = combined_tokens[high_activation_mask]
#         activating_values = combined_activations[high_activation_mask]
        
#         token_stats = {}
#         for token_id, activation in zip(activating_tokens, activating_values):
#             token_id = token_id.item()
#             activation = activation.item()
            
#             if token_id not in token_stats:
#                 token_stats[token_id] = {"count": 0, "total_activation": 0.0}
            
#             token_stats[token_id]["count"] += 1
#             token_stats[token_id]["total_activation"] += activation
        
#         token_ranking = []
#         for token_id, stats in token_stats.items():
#             avg_activation = stats["total_activation"] / stats["count"]
#             token_text = tokenizer.decode([token_id])
            
#             token_ranking.append({
#                 "token": token_text,
#                 "token_id": token_id,
#                 "frequency": stats["count"],
#                 "average_activation": avg_activation
#             })
        
#         token_ranking.sort(key=lambda x: x["frequency"], reverse=True)
#         return token_ranking[:top_k]

# def highlight_activations(tokens: torch.Tensor, activations: torch.Tensor, tokenizer, threshold_ratio: float = 0.6) -> str:
#     assert len(tokens) == len(activations), "Token and activation lengths must match"

#     max_act = activations.max().item()
#     threshold = max_act * threshold_ratio
#     str_tokens = tokenizer.convert_ids_to_tokens(tokens)

#     # Find highlight boundaries and extend to word boundaries
#     highlight_mask = activations > threshold
    
#     # Helper function to detect if text contains Chinese characters
#     def contains_chinese(text):
#         return any('\u4e00' <= char <= '\u9fff' for char in text)
    
#     # Check if we're dealing with Chinese text
#     full_text_sample = tokenizer.convert_tokens_to_string(str_tokens[:min(10, len(str_tokens))])
#     is_chinese_text = contains_chinese(full_text_sample)
    
#     # Extend highlighting to complete words, but handle Chinese differently
#     extended_mask = highlight_mask.clone()
    
#     if is_chinese_text:
#         # For Chinese text, don't extend highlighting - use exact token boundaries
#         # This prevents over-highlighting entire sentences
#         pass  # extended_mask remains the same as highlight_mask
#     else:
#         # For non-Chinese text, extend to complete words as before
#         for i in range(len(str_tokens)):
#             if highlight_mask[i]:
#                 # Extend backwards to start of word
#                 j = i - 1
#                 while j >= 0 and not str_tokens[j].startswith(('▁', ' ', 'Ġ')) and str_tokens[j] not in ['<|endoftext|>', '</s>', '<s>', '[CLS]', '[SEP]']:
#                     extended_mask[j] = True
#                     j -= 1
#                 # Extend forwards to end of word  
#                 j = i + 1
#                 while j < len(str_tokens) and not str_tokens[j].startswith(('▁', ' ', 'Ġ')) and str_tokens[j] not in ['<|endoftext|>', '</s>', '<s>', '[CLS]', '[SEP]']:
#                     extended_mask[j] = True
#                     j += 1

#     marked_tokens = []
#     in_highlight = False

#     for i, (tok, is_high) in enumerate(zip(str_tokens, extended_mask)):
#         if is_high and not in_highlight:
#             marked_tokens.append("<<")
#             in_highlight = True
#         elif not is_high and in_highlight:
#             marked_tokens.append(">>")
#             in_highlight = False

#         marked_tokens.append(tok)

#     if in_highlight:
#         marked_tokens.append(">>")

#     segments = []
#     buffer: list[str] = []
#     for tok in marked_tokens:
#         if tok in {"<<", ">>"}:
#             if buffer:
#                 segments.append(tokenizer.convert_tokens_to_string(buffer))
#                 buffer = []
#             segments.append(tok)
#         else:
#             buffer.append(tok)
#     if buffer:
#         segments.append(tokenizer.convert_tokens_to_string(buffer))

#     result = ""
#     i = 0
#     while i < len(segments):
#         seg = segments[i]
#         if seg == "<<":
#             if result and not result.endswith(" "):
#                 result += " "
#             result += "<<"
#             i += 1
#             if i < len(segments):
#                 result += segments[i].lstrip()
#                 i += 1
#             if i < len(segments) and segments[i] == ">>":
#                 result += ">>"
#                 i += 1
#         else:
#             if result and not result.endswith(" "):
#                 result += " "
#             result += seg
#             i += 1

#     return result
