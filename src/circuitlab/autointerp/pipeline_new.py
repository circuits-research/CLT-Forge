"""
AutoInterp pipeline — streaming, feature-parallel, single-pass.
Storage layout:
    save_dir/
      features/
        layer_{l}/
          job_{j}.json   ← {str(feat_id): {feature_dict}, ...}
      prompts/           ← written only if generate_explanations=True
      explanations/      ← written only if generate_explanations=True
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from sae_lens.load_model import load_model

from circuitlab.config import AutoInterpConfig
from circuitlab.clt import CLT
from circuitlab.training.activations_store import ActivationsStore
from circuitlab.training.optim import JumpReLU
from circuitlab.transformer_lens.multilingual_patching import (
    patch_official_model_names,
    patch_convert_hf_model_config,
)
from circuitlab.utils import DTYPE_MAP
from circuitlab import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOP_K_DEFAULT = 100
N_TOP_ACTIVATING_TOKENS_SHOWN = 4

# DDL shared by _save_to_sqlite and merge_job_databases
_FEATURES_TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS features (
        layer                 INTEGER NOT NULL,
        feature_id            INTEGER NOT NULL,
        average_activation    REAL    NOT NULL,
        top_examples          TEXT    NOT NULL,
        top_examples_tks      TEXT    NOT NULL,
        top_activating_tokens TEXT    NOT NULL,
        description           TEXT    NOT NULL,
        explanation           TEXT    NOT NULL,
        raw_explanation       TEXT    NOT NULL,
        PRIMARY KEY (layer, feature_id)
    )
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AutoInterp:
    """
    Single-pass streaming AutoInterp.

    Usage:
        cfg = AutoInterpConfig(...)
        interp = AutoInterp(cfg)
        interp.run(job_id=0, total_jobs=32, top_k=100, save_dir=Path(...))
    """

    def __init__(self, cfg: AutoInterpConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.ctx = cfg.context_size

        patch_official_model_names()
        patch_convert_hf_model_config()

        self.model = load_model(
            cfg.model_class_name,
            cfg.model_name,
            device=self.device,
            model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
        )

        # CLT stays on CPU; only the feature subset is moved to GPU per job.
        self.clt = CLT.load_from_pretrained(cfg.clt_path, "cpu")

        self.activations_store = ActivationsStore(
            self.model,
            cfg,
            estimated_norm_scaling_factor_in=self.clt.estimated_norm_scaling_factor_in,
            estimated_norm_scaling_factor_out=self.clt.estimated_norm_scaling_factor_out,
        )
        # ActivationsStore defaults to return_tokens=False (yields 2-tuples).
        # We need the token ids alongside activations, so enable it here.
        self.activations_store.return_tokens = True

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        job_id: int,
        total_jobs: int,
        top_k: int = TOP_K_DEFAULT,
        save_dir: Optional[Path] = None,
        generate_explanations: bool = False,
    ) -> None:
        """
        Run the full pipeline for one job (= one feature slice).

        Args:
            job_id:               Index of this job in [0, total_jobs).
            total_jobs:           Total number of parallel jobs.
            top_k:                Number of top-activating sequences per feature.
            save_dir:             Root output directory.
            generate_explanations: If True, run vLLM to generate explanations.
        """
        if save_dir is None:
            if self.cfg.latent_cache_path is None:
                raise ValueError(
                    "Either pass save_dir or set cfg.latent_cache_path."
                )
            save_dir = Path(self.cfg.latent_cache_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_list = self._compute_index_list(job_id, total_jobs)
        logger.info(
            f"[Job {job_id}/{total_jobs}] features {index_list[0]}–{index_list[-1]}"
            f" ({len(index_list)} total)"
        )

        run_dtype = DTYPE_MAP[self.cfg.dtype]
        self._prepare_encoder_subset(index_list=index_list, dtype=run_dtype)

        state = self._stream_and_build_topk(top_k=top_k, run_dtype=run_dtype)

        logger.info(f"[Job {job_id}] Building feature dictionaries…")
        feature_dicts_by_layer = self._state_to_feature_dicts(
            state=state, index_list=index_list, top_k=top_k
        )

        self._save_to_sqlite(
            feature_dicts_by_layer=feature_dicts_by_layer,
            job_id=job_id,
            save_dir=save_dir,
        )

        if generate_explanations:
            logger.info(f"[Job {job_id}] Generating LLM explanations…")
            self._generate_and_add_explanations(
                feature_dicts_by_layer=feature_dicts_by_layer,
                job_id=job_id,
                save_dir=save_dir,
            )

        logger.info(f"[Job {job_id}] Done.")

    # ------------------------------------------------------------------
    # Feature index computation
    # ------------------------------------------------------------------

    def _compute_index_list(self, job_id: int, total_jobs: int) -> List[int]:
        d_latent = self.clt.d_latent
        per_job = d_latent // total_jobs
        start = job_id * per_job
        end = start + per_job if job_id < total_jobs - 1 else d_latent
        return list(range(start, end))

    # ------------------------------------------------------------------
    # Encoder subset — slice CTL params for index_list, load onto GPU
    # ------------------------------------------------------------------

    def _prepare_encoder_subset(
        self, *, index_list: List[int], dtype: torch.dtype
    ) -> None:
        idx = torch.as_tensor(index_list, dtype=torch.long)
        self._index_list = list(index_list)
        self._W_enc_sub = self.clt.W_enc[:, :, idx].to(
            self.device, dtype=dtype, non_blocking=True
        )  # [L, d_in, F]
        self._b_enc_sub = self.clt.b_enc[:, idx].to(
            self.device, dtype=dtype, non_blocking=True
        )  # [L, F]
        self._threshold_sub = torch.exp(self.clt.log_threshold[:, idx]).to(
            self.device, dtype=dtype, non_blocking=True
        )  # [L, F]
        self._bandwidth = self.clt.bandwidth

    @torch.no_grad()
    def _encode_subset(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_in] → [B, L, F]  (all on GPU)"""
        hidden_pre = (
            torch.einsum("bld,ldf->blf", x, self._W_enc_sub) + self._b_enc_sub
        )
        return JumpReLU.apply(hidden_pre, self._threshold_sub, self._bandwidth)

    # ------------------------------------------------------------------
    # Single-pass streaming + top-K accumulation
    # ------------------------------------------------------------------

    def _stream_and_build_topk(
        self, *, top_k: int, run_dtype: torch.dtype
    ) -> Dict[str, Any]:
        """One pass over the data: encode and accumulate per-feature top-K."""
        state: Optional[Dict[str, Any]] = None
        n_tokens = 0
        iterator = iter(self.activations_store)
        log_every = max(1, self.cfg.total_autointerp_tokens // 20)

        while n_tokens < self.cfg.total_autointerp_tokens:
            tokens_cpu, acts_in, _ = next(iterator)
            x = acts_in.to(self.device, dtype=run_dtype)

            with torch.no_grad():
                feat_act = self._encode_subset(x)  # [B, L, F]

            tokens_seq, acts_seq = self._to_sequences(tokens_cpu, feat_act)
            n_tokens += int(feat_act.shape[0])

            if tokens_seq.shape[0] > 0:
                if state is None:
                    state = self._init_topk_state(
                        n_layers=int(acts_seq.shape[2]),
                        F=len(self._index_list),
                        top_k=top_k,
                        ctx=self.ctx,
                        dtype=run_dtype,
                        device=self.device,
                    )
                self._update_state(
                    state=state,
                    tokens_seq=tokens_seq,
                    acts_seq=acts_seq,
                    top_k=top_k,
                )

            del x, feat_act, acts_seq
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if n_tokens % log_every < self.cfg.train_batch_size_tokens:
                logger.info(
                    f"  {n_tokens:,} / {self.cfg.total_autointerp_tokens:,} tokens"
                )

        if state is None:
            raise RuntimeError("No data was processed — check ActivationsStore setup.")
        return state

    def _to_sequences(
        self, tokens_cpu: torch.Tensor, feat_act_gpu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape flat token/activation arrays into context-sized sequences.

        tokens_cpu:   [B]          (CPU)
        feat_act_gpu: [B, L, F]   (GPU)

        returns:
            tokens_seq: [B_seq, ctx]        (CPU)
            acts_seq:   [B_seq, ctx, L, F]  (GPU)
        """
        B = int(feat_act_gpu.shape[0])
        excess = B % self.ctx
        if excess:
            feat_act_gpu = feat_act_gpu[:-excess]
            tokens_cpu = tokens_cpu[:-excess]
            B -= excess

        if B == 0:
            L, F = feat_act_gpu.shape[1], feat_act_gpu.shape[2]
            return (
                tokens_cpu.new_zeros((0, self.ctx)),
                feat_act_gpu.new_zeros((0, self.ctx, L, F)),
            )

        B_seq = B // self.ctx
        tokens_seq = tokens_cpu.view(B_seq, self.ctx)
        acts_seq = feat_act_gpu.view(
            B_seq, self.ctx, feat_act_gpu.shape[1], feat_act_gpu.shape[2]
        )
        return tokens_seq, acts_seq

    @staticmethod
    def _init_topk_state(
        *,
        n_layers: int,
        F: int,
        top_k: int,
        ctx: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, Any]:
        neg_inf = torch.full((top_k, F), float("-inf"), device=device, dtype=dtype)
        tok_zero = torch.zeros((top_k, F, ctx), device=device, dtype=torch.int32)
        act_zero = torch.zeros((top_k, F, ctx), device=device, dtype=dtype)
        return {
            "top_vals":   [neg_inf.clone() for _ in range(n_layers)],
            "top_tokens": [tok_zero.clone() for _ in range(n_layers)],
            "top_acts":   [act_zero.clone() for _ in range(n_layers)],
            "sum_pos":    [torch.zeros(F, device=device, dtype=dtype) for _ in range(n_layers)],
            "count_pos":  [torch.zeros(F, device=device, dtype=torch.long) for _ in range(n_layers)],
        }

    @torch.no_grad()
    def _update_state(
        self,
        *,
        state: Dict[str, Any],
        tokens_seq: torch.Tensor,  # [B_seq, ctx]      CPU
        acts_seq: torch.Tensor,    # [B_seq, ctx, L, F] GPU
        top_k: int,
    ) -> None:
        B_seq = int(tokens_seq.shape[0])
        if B_seq == 0:
            return

        tokens_gpu = tokens_seq.to(self.device, dtype=torch.int32)
        L = int(acts_seq.shape[2])
        F = int(acts_seq.shape[3])
        ctx = int(acts_seq.shape[1])

        # Broadcast token sequence across all F features: [B_seq, F, ctx]
        batch_tokens = tokens_gpu[:, None, :].expand(B_seq, F, ctx).contiguous()

        for layer in range(L):
            # [B_seq, F, ctx]
            batch_acts = acts_seq[:, :, layer, :].permute(0, 2, 1).contiguous()
            # [B_seq, F] — max activation per sequence per feature
            batch_vals = batch_acts.max(dim=2).values

            # Running stats (for average activation computation)
            pos = batch_vals > 0
            state["sum_pos"][layer].add_((batch_vals * pos).sum(dim=0))
            state["count_pos"][layer].add_(pos.sum(dim=0))

            # Merge batch with current top-K, keep top-K
            merged_vals   = torch.cat([state["top_vals"][layer],   batch_vals  ], dim=0)  # [K+B, F]
            merged_tokens = torch.cat([state["top_tokens"][layer], batch_tokens], dim=0)  # [K+B, F, ctx]
            merged_acts   = torch.cat([state["top_acts"][layer],   batch_acts  ], dim=0)  # [K+B, F, ctx]

            new_vals, sel = torch.topk(merged_vals, k=top_k, dim=0)   # [K, F]
            gather_idx = sel[:, :, None].expand(top_k, F, ctx)        # [K, F, ctx]

            state["top_vals"][layer]   = new_vals
            state["top_tokens"][layer] = torch.gather(merged_tokens, 0, gather_idx)
            state["top_acts"][layer]   = torch.gather(merged_acts,   0, gather_idx)

        del tokens_gpu, batch_tokens

    # ------------------------------------------------------------------
    # Build feature dictionaries from accumulated state
    # ------------------------------------------------------------------

    def _state_to_feature_dicts(
        self,
        *,
        state: Dict[str, Any],
        index_list: List[int],
        top_k: int,
    ) -> List[Dict[str, Dict[str, Any]]]:
        """
        Returns a list of length n_layers.
        Each element is {str(feat_id): feature_dict} for that layer.
        String keys are used for JSON compatibility.
        """
        tokenizer = self.model.tokenizer
        n_layers = len(state["top_vals"])
        result: List[Dict[str, Dict[str, Any]]] = [{} for _ in range(n_layers)]

        for layer in range(n_layers):
            top_vals   = state["top_vals"][layer].cpu()    # [K, F]
            top_tokens = state["top_tokens"][layer].cpu()  # [K, F, ctx]
            top_acts   = state["top_acts"][layer].cpu()    # [K, F, ctx]
            sum_pos    = state["sum_pos"][layer].cpu()     # [F]
            count_pos  = state["count_pos"][layer].cpu()   # [F]

            for j, feat_id in enumerate(index_list):
                # Collect non-trivial top sequences (descending order from topk)
                sequences: List[Dict[str, Any]] = []
                for k in range(top_k):
                    v = float(top_vals[k, j])
                    if v == float("-inf") or v <= 0:
                        break  # remaining slots are unfilled or zero
                    sequences.append(
                        {
                            "tokens":      top_tokens[k, j],
                            "activations": top_acts[k, j],
                            "max_val":     v,
                        }
                    )

                top_examples: List[str] = []
                sequences_serializable: List[Dict[str, Any]] = []
                for s in sequences:
                    tks  = s["tokens"][1:]       # drop BOS token
                    acts = s["activations"][1:]  # drop BOS
                    top_examples.append(highlight_activations(tks, acts, tokenizer))
                    sequences_serializable.append(
                        {
                            "tokens":      s["tokens"].tolist(),
                            "activations": s["activations"].tolist(),
                            "max_val":     float(s["max_val"]),
                        }
                    )

                c = int(count_pos[j])
                avg_activation = float(sum_pos[j]) / c if c > 0 else 0.0

                result[layer][str(feat_id)] = {
                    "layer":                 int(layer),
                    "feature_index":         int(feat_id),
                    "average_activation":    avg_activation,
                    "top_examples":          top_examples,
                    "top_examples_tks":      sequences_serializable,
                    "top_activating_tokens": _get_top_activating_tokens(
                        sequences=sequences,
                        tokenizer=tokenizer,
                        top_k=N_TOP_ACTIVATING_TOKENS_SHOWN,
                    ),
                    "description":           "Unknown",
                    "explanation":           "No explanation generated",
                    "raw_explanation":       "",
                }

        return result

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_to_sqlite(
        self,
        *,
        feature_dicts_by_layer: List[Dict[str, Dict[str, Any]]],
        job_id: int,
        save_dir: Path,
    ) -> Path:
        """
        Persist all features for this job in a single SQLite database.

        Output: save_dir/job_{job_id}.db  — one file per job regardless of
        how many layers or features it contains.

        Query specific features later with load_features(db_path, layer, [ids]).
        Merge all jobs with merge_job_databases([...], output_db_path).
        """
        import sqlite3

        db_path = save_dir / f"job_{job_id}.db"
        con = sqlite3.connect(db_path)
        con.execute(_FEATURES_TABLE_DDL)

        rows = [
            (
                d["layer"],
                d["feature_index"],
                d["average_activation"],
                json.dumps(d["top_examples"]),
                json.dumps(d["top_examples_tks"]),
                json.dumps(d["top_activating_tokens"]),
                d["description"],
                d["explanation"],
                d["raw_explanation"],
            )
            for layer_dicts in feature_dicts_by_layer
            for d in layer_dicts.values()
        ]

        con.executemany(
            "INSERT OR REPLACE INTO features VALUES (?,?,?,?,?,?,?,?,?)", rows
        )
        con.commit()
        con.close()

        logger.info(f"[Job {job_id}] Saved {len(rows)} features → {db_path}")
        return db_path

    # ------------------------------------------------------------------
    # Optional LLM explanations
    # ------------------------------------------------------------------

    def _generate_and_add_explanations(
        self,
        *,
        feature_dicts_by_layer: List[Dict[str, Dict[str, Any]]],
        job_id: int,
        save_dir: Path,
    ) -> None:
        """
        Build prompts in-memory, run vLLM, parse responses, patch the
        in-memory dicts, and re-save to SQLite.  No files are written.
        """
        from circuitlab.autointerp.client import run_client
        from circuitlab.autointerp.prompt import generate_prompt

        prompt_texts: List[str] = []
        feat_layer_keys: List[Tuple[int, str]] = []

        for layer, layer_dicts in enumerate(feature_dicts_by_layer):
            for feat_key, feat_dict in layer_dicts.items():
                if not feat_dict["top_examples"]:
                    continue  # dead feature — skip
                prompt_texts.append(
                    generate_prompt(feat_dict["top_examples"], layer, int(feat_key))
                )
                feat_layer_keys.append((layer, feat_key))

        if not prompt_texts:
            logger.info(f"[Job {job_id}] No live features — skipping vLLM.")
            return

        explanations = run_client(
            prompts=prompt_texts,
            vllm_model=self.cfg.vllm_model,
            vllm_max_tokens=self.cfg.vllm_max_tokens,
        )

        for raw, (layer, feat_key) in zip(explanations, feat_layer_keys):
            desc, expl = _parse_explanation(raw)
            d = feature_dicts_by_layer[layer][feat_key]
            d["raw_explanation"] = raw
            d["description"]     = desc
            d["explanation"]     = expl

        # Re-save to SQLite with explanations filled in
        self._save_to_sqlite(
            feature_dicts_by_layer=feature_dicts_by_layer,
            job_id=job_id,
            save_dir=save_dir,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def highlight_activations(
    tokens: torch.Tensor,
    activations: torch.Tensor,
    tokenizer,
    threshold_ratio: float = 0.6,
) -> str:
    """
    Build a text string with <<highlighted>> spans around the most active tokens.
    Handles Chinese text (no word-boundary extension) vs. Latin scripts.
    """
    assert len(tokens) == len(activations), "Token/activation length mismatch."
    if activations.numel() == 0:
        return ""

    max_act = float(activations.max())
    if max_act <= 0:
        return tokenizer.decode(tokens.tolist())

    threshold = max_act * threshold_ratio
    str_tokens = tokenizer.convert_ids_to_tokens(tokens.tolist())
    highlight_mask = (activations > threshold).tolist()

    def _contains_chinese(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    sample = tokenizer.convert_tokens_to_string(str_tokens[: min(10, len(str_tokens))])
    is_chinese = _contains_chinese(sample)

    extended = list(highlight_mask)
    if not is_chinese:
        WORD_STARTS = {"▁", " ", "Ġ"}
        SPECIAL = {"<|endoftext|>", "</s>", "<s>", "[CLS]", "[SEP]"}
        for i in range(len(str_tokens)):
            if not highlight_mask[i]:
                continue
            # Extend backwards to word start
            j = i - 1
            while (
                j >= 0
                and str_tokens[j][:1] not in WORD_STARTS
                and str_tokens[j] not in SPECIAL
            ):
                extended[j] = True
                j -= 1
            # Extend forwards to word end
            j = i + 1
            while (
                j < len(str_tokens)
                and str_tokens[j][:1] not in WORD_STARTS
                and str_tokens[j] not in SPECIAL
            ):
                extended[j] = True
                j += 1

    # Build marked token list
    marked: List[str] = []
    in_hl = False
    for tok, hi in zip(str_tokens, extended):
        if hi and not in_hl:
            marked.append("<<")
            in_hl = True
        elif not hi and in_hl:
            marked.append(">>")
            in_hl = False
        marked.append(tok)
    if in_hl:
        marked.append(">>")

    # Merge token subwords back into strings
    segments: List[str] = []
    buf: List[str] = []
    for tok in marked:
        if tok in {"<<", ">>"}:
            if buf:
                segments.append(tokenizer.convert_tokens_to_string(buf))
                buf = []
            segments.append(tok)
        else:
            buf.append(tok)
    if buf:
        segments.append(tokenizer.convert_tokens_to_string(buf))

    # Assemble final string
    result = ""
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg == "<<":
            if result and not result.endswith(" "):
                result += " "
            result += "<<"
            i += 1
            if i < len(segments):
                result += segments[i].lstrip()
                i += 1
            if i < len(segments) and segments[i] == ">>":
                result += ">>"
                i += 1
        else:
            if result and not result.endswith(" "):
                result += " "
            result += seg
            i += 1

    return result


def _get_top_activating_tokens(
    *,
    sequences: List[Dict[str, Any]],
    tokenizer,
    top_k: int = N_TOP_ACTIVATING_TOKENS_SHOWN,
    threshold_ratio: float = 0.6,
) -> List[Dict[str, Any]]:
    """Return the most frequently highly-active tokens across top sequences."""
    if not sequences:
        return []

    all_tks  = torch.cat([s["tokens"][1:]      for s in sequences])
    all_acts = torch.cat([s["activations"][1:] for s in sequences])

    if all_acts.numel() == 0:
        return []

    thresh = float(all_acts.max()) * threshold_ratio
    mask = all_acts > thresh

    stats: Dict[int, Dict[str, float]] = {}
    for tid, act in zip(all_tks[mask].tolist(), all_acts[mask].tolist()):
        if tid not in stats:
            stats[tid] = {"count": 0.0, "total": 0.0}
        stats[tid]["count"] += 1.0
        stats[tid]["total"] += float(act)

    ranking = [
        {
            "token":              tokenizer.decode([int(tid)]),
            "token_id":           int(tid),
            "frequency":          int(v["count"]),
            "average_activation": v["total"] / v["count"],
        }
        for tid, v in stats.items()
    ]
    ranking.sort(key=lambda x: x["frequency"], reverse=True)
    return ranking[:top_k]


def _parse_explanation(raw: str) -> Tuple[str, str]:
    """Parse the [DESCRIPTION]: / [EXPLANATION]: response format."""
    description = "Unknown"
    explanation = raw
    for line in raw.splitlines():
        if line.startswith("[DESCRIPTION]:"):
            description = line[len("[DESCRIPTION]:"):].strip()
        elif line.startswith("[EXPLANATION]:"):
            explanation = line[len("[EXPLANATION]:"):].strip()
    return description, explanation


# ---------------------------------------------------------------------------
# Public utilities for loading and merging results
# ---------------------------------------------------------------------------

def load_features(
    db_path: Path,
    layer: int,
    feature_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Load feature dicts from a job SQLite database.

    Args:
        db_path:     Path to a job_{j}.db file (or merged features.db).
        layer:       Which CLT layer to query.
        feature_ids: Specific feature IDs to load.  None → all for this layer.

    Returns:
        List of feature dicts with JSON fields already deserialized.

    Example:
        features = load_features(Path("save_dir/job_0.db"), layer=3)
        feature  = load_features(Path("save_dir/features.db"), layer=3, feature_ids=[1234])
    """
    import sqlite3

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    if feature_ids is None:
        cursor = con.execute(
            "SELECT * FROM features WHERE layer = ?", (layer,)
        )
    else:
        placeholders = ",".join("?" * len(feature_ids))
        cursor = con.execute(
            f"SELECT * FROM features WHERE layer = ? AND feature_id IN ({placeholders})",
            (layer, *feature_ids),
        )

    rows = cursor.fetchall()
    con.close()

    result = []
    for row in rows:
        d = dict(row)
        d["top_examples"]          = json.loads(d["top_examples"])
        d["top_examples_tks"]      = json.loads(d["top_examples_tks"])
        d["top_activating_tokens"] = json.loads(d["top_activating_tokens"])
        result.append(d)
    return result


def merge_job_databases(job_db_paths: List[Path], output_db_path: Path) -> None:
    """
    Merge all per-job SQLite databases into one file.

    Typical use after all jobs have finished:
        merge_job_databases(
            sorted(save_dir.glob("job_*.db")),
            save_dir / "features.db",
        )

    The resulting features.db supports the same load_features() queries
    across the full feature space.
    """
    import sqlite3

    con = sqlite3.connect(output_db_path)
    con.execute(_FEATURES_TABLE_DDL)
    con.commit()

    for db_path in job_db_paths:
        con.execute("ATTACH DATABASE ? AS src", (str(db_path),))
        con.execute("INSERT OR REPLACE INTO features SELECT * FROM src.features")
        con.execute("DETACH DATABASE src")
        con.commit()

    con.close()
    logger.info(f"Merged {len(job_db_paths)} databases → {output_db_path}")
