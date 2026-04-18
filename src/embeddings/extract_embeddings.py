import sys
import json
import time
import logging
import argparse
import warnings
import torch
from datetime import datetime
from pathlib import Path
from typing import Iterator

import torch

# ── path setup: works from any CWD ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from constants import OUT_DIR, PROTEINS_WITH_SPLIT_TSV_PATH

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
#  CONSTANTS & DEFAULTS
# =============================================================================

DEFAULT_MODEL      = "facebook/esm2_t33_650M_UR50D"
DEFAULT_MAX_TOKENS = 4096    # max total tokens per forward pass
DEFAULT_SHARD_SIZE = 512     # proteins per shard file
DEFAULT_MAX_LEN    = 1022    # ESM-2 max sequence length (excl. special tokens)
SPLIT_ORDER        = ["train", "val", "test"]
AMINO_ACIDS        = set("ACDEFGHIKLMNPQRSTVWY")  # standard AA alphabet


# =============================================================================
#  ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-residue ESM-2 embeddings for a protein subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", type=Path,
        default=OUT_DIR / "subsets" / "proteins_subset_10pct.tsv",
        help="Path to the proteins TSV (must have accession, sequence, split).",
    )
    p.add_argument(
        "--outdir", type=Path,
        default=OUT_DIR / "embeddings" / "esm2_650m",
        help="Directory where shard .pt files will be written.",
    )
    p.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Hugging Face model identifier.",
    )
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens per batch (including special tokens). "
             "Reduce if you get OOM errors.",
    )
    p.add_argument(
        "--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
        help="Number of proteins per shard file.",
    )
    p.add_argument(
        "--max-len", type=int, default=DEFAULT_MAX_LEN,
        help="Proteins longer than this are truncated (ESM-2 limit is 1022).",
    )
    p.add_argument(
        "--splits", nargs="+", default=SPLIT_ORDER,
        choices=["train", "val", "test"],
        help="Which splits to process.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip splits whose shards are already complete.",
    )
    p.add_argument(
        "--fp16", action="store_true",
        help="Force FP16 mixed-precision on GPU (auto-enabled for CUDA).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Optional Hugging Face model cache directory.",
    )
    return p.parse_args()


# =============================================================================
#  INPUT LOADING & VALIDATION
# =============================================================================

def load_input(path: Path) -> pd.DataFrame:
    """Read the protein TSV and normalise column names."""
    log.info(f"Reading input: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    log.info(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def validate_input(df: pd.DataFrame) -> None:
    """Check required columns exist and report basic stats."""
    required = {"accession", "sequence", "split"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Input TSV is missing required columns: {missing}")

    df["split"] = df["split"].str.strip().str.lower()

    n_total = len(df)
    for sp in SPLIT_ORDER:
        n = (df["split"] == sp).sum()
        log.info(f"  {sp:5s}: {n:>8,} proteins")

    dup = df["accession"].duplicated().sum()
    if dup:
        log.warning(f"  {dup:,} duplicate accessions found — keeping first occurrence")


# =============================================================================
#  SEQUENCE CLEANING
# =============================================================================

def clean_sequences(
    df: pd.DataFrame,
    max_len: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and clean sequences.

    Rules:
    - Drop rows with null / empty sequences.
    - Drop rows containing non-standard characters (after uppercasing).
      Non-standard characters confuse the tokenizer and produce misleading
      embeddings; it is safer to skip than to silently embed garbage.
    - Truncate sequences longer than max_len with a warning (preserves the
      protein but logs the truncation so the user is aware).

    Returns (clean_df, skipped_df).
    """
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

    # --- null / empty
    mask_empty = df["sequence"].isin(["", "NAN", "NONE"])
    skipped_empty = df[mask_empty].copy()
    skipped_empty["skip_reason"] = "empty_sequence"
    df = df[~mask_empty]

    # --- non-standard amino acids
    # vectorised: check if every character is in AMINO_ACIDS
    def _is_valid(seq: str) -> bool:
        return bool(seq) and all(c in AMINO_ACIDS for c in seq)

    valid_mask = df["sequence"].map(_is_valid)
    skipped_invalid = df[~valid_mask].copy()
    skipped_invalid["skip_reason"] = "invalid_characters"
    df = df[valid_mask]

    # --- truncate (warn only, keep the protein)
    long_mask = df["sequence"].str.len() > max_len
    n_long = long_mask.sum()
    if n_long:
        log.warning(
            f"  {n_long:,} sequences exceed max_len={max_len} and will be truncated."
        )
        df.loc[long_mask, "sequence"] = df.loc[long_mask, "sequence"].str[:max_len]

    # collect all skipped
    skipped = pd.concat([skipped_empty, skipped_invalid], ignore_index=True)

    log.info(f"  After cleaning: {len(df):,} valid proteins, "
             f"{len(skipped):,} skipped")
    return df.reset_index(drop=True), skipped


# =============================================================================
#  TOKEN-COUNT-AWARE BATCHING
# =============================================================================

def build_batches(
    sequences: list[str],
    indices: list[int],
    max_tokens: int,
) -> list[list[int]]:
    """
    Group sequence indices into batches where the total token count
    (including 2 special tokens per sequence) does not exceed max_tokens.

    Proteins are sorted by length (longest first within each batch) to
    minimise padding waste.  This is the key strategy that prevents OOM on
    sequences of highly variable length.

    Why token-count batching instead of fixed batch size?
    A batch of 32 proteins averaging 50 aa is fine; 32 proteins averaging
    900 aa would require >29k tokens × D floats per layer and OOM on most
    GPUs.  By capping total tokens we get stable memory usage.
    """
    # sort by length descending so the first sequence in each batch determines
    # the padding length, and we know immediately if it fits
    order = sorted(range(len(sequences)),
                   key=lambda i: len(sequences[i]), reverse=True)

    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_tokens = 0

    for pos in order:
        # +2 for [CLS] and [EOS] special tokens
        seq_tokens = len(sequences[pos]) + 2
        # a single sequence that exceeds the limit is processed alone
        if seq_tokens > max_tokens and not current_batch:
            batches.append([indices[pos]])
            continue
        # flush current batch if adding this sequence would overflow
        if current_tokens + seq_tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(indices[pos])
        current_tokens += seq_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


# =============================================================================
#  MODEL LOADING
# =============================================================================

def load_model_and_tokenizer(
    model_name: str,
    device: "torch.device",
    fp16: bool,
    cache_dir: Path | None,
) -> tuple:
    """
    Load the ESM-2 model and tokenizer from Hugging Face.

    The model is set to eval() mode immediately; dropout layers are disabled,
    which is required for deterministic, reproducible embeddings.
    """
    log.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    log.info(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    # fp16 reduces VRAM ~2× with negligible embedding quality loss
    if fp16 and device.type == "cuda":
        model = model.half()
        log.info("  FP16 (half precision) enabled")

    model = model.to(device)
    model.eval()

    # get hidden dimension from config
    hidden_dim = model.config.hidden_size
    log.info(f"  Hidden dimension : {hidden_dim}")
    log.info(f"  Device           : {device}")

    return model, tokenizer, hidden_dim


# =============================================================================
#  EMBEDDING EXTRACTION
# =============================================================================

def trim_special_tokens(
    hidden_states: "torch.Tensor",
    attention_mask: "torch.Tensor",
) -> list["torch.Tensor"]:
    """
    Remove the [CLS] (position 0) and [EOS] (last real token) special tokens
    from each sequence in a batch.

    ESM tokenizer adds:
        [CLS]  aa_1  aa_2  ...  aa_L  [EOS]  [PAD] ...

    attention_mask is 1 for real tokens (incl. special) and 0 for padding.
    The real length (incl. special tokens) = attention_mask.sum(dim=1).
    Residue embeddings span positions [1 : real_len - 1].

    Returns a list of [L, D] tensors (one per sequence, on CPU).
    """
    trimmed: list["torch.Tensor"] = []
    real_lengths = attention_mask.sum(dim=1)  # includes CLS + EOS

    for i, real_len in enumerate(real_lengths):
        rl = real_len.item()
        # positions [1 : rl-1] are the actual residue embeddings
        residue_emb = hidden_states[i, 1 : rl - 1, :]   # [L, D]
        trimmed.append(residue_emb.cpu().to(torch.float16))        # always save as fp16

    return trimmed


def extract_batch(
    model,
    tokenizer,
    sequences: list[str],
    device: "torch.device",
    fp16: bool,
) -> tuple[list["torch.Tensor"], list["torch.Tensor"]]:
    """
    Run one forward pass for a batch of sequences.

    Returns
    -------
    per_residue  : list of [L, D] tensors (special tokens removed)
    mean_pooled  : list of [D] tensors
    """
    # no_grad is applied here rather than as a decorator so that torch
    # can be imported lazily (deferred until main() runs).
    with torch.no_grad():
        return _extract_batch_inner(model, tokenizer, sequences, device, fp16)


def _extract_batch_inner(model, tokenizer, sequences, device, fp16):
    # tokenize with padding to the longest sequence in this batch
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=DEFAULT_MAX_LEN + 2,  # +2 for special tokens
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # mixed precision context — on CPU this is a no-op
    ctx = (torch.cuda.amp.autocast() if fp16 and device.type == "cuda"
           else torch.inference_mode())

    with ctx:
        outputs = model(**inputs, output_hidden_states=False)
        # last_hidden_state shape: [batch, seq_len_padded, D]
        hidden = outputs.last_hidden_state

    # --- per-residue embeddings (special tokens stripped) --------------------
    per_residue = trim_special_tokens(hidden, inputs["attention_mask"])

    # --- mean-pooled embeddings (over residues only, not special tokens) -----
    # We reuse the trimmed tensors rather than re-indexing hidden to avoid
    # redundancy.  Mean over the L dimension → [D].
    mean_pooled = [emb.mean(dim=0) for emb in per_residue]

    return per_residue, mean_pooled


# =============================================================================
#  SHARD I/O
# =============================================================================

def save_shard(
    shard_data: dict,
    path: Path,
) -> None:
    """
    Save a shard dict to disk.

    Tensors are already on CPU (enforced in extract_batch / trim_special_tokens).
    torch.save uses pickle internally; loading is:
        data = torch.load("train_shard_000.pt", weights_only=False)
    """
    torch.save(shard_data, path)
    n = len(shard_data["accessions"])
    size_mb = path.stat().st_size / 1024 / 1024
    log.info(f"    Saved shard: {path.name}  "
             f"({n} proteins, {size_mb:.1f} MB)")


def _empty_shard() -> dict:
    return {
        "accessions":      [],
        "splits":          [],
        "lengths":         [],
        "embeddings":      [],
        "mean_embeddings": [],
    }


# =============================================================================
#  SHARD RESUME CHECK
# =============================================================================

def count_existing_shards(outdir: Path, split: str) -> int:
    """Return how many shards for this split already exist on disk."""
    return len(list(outdir.glob(f"{split}_shard_*.pt")))


# =============================================================================
#  PER-SPLIT PROCESSING
# =============================================================================

def process_split(
    split: str,
    df_split: pd.DataFrame,
    model,
    tokenizer,
    device: "torch.device",
    args: argparse.Namespace,
    outdir: Path,
) -> dict:
    """
    Process all proteins in one split and write shard files incrementally.

    Returns a summary dict for the final report.
    """
    log.info(f"\n{'─'*60}")
    log.info(f"  Processing split: {split.upper()}  ({len(df_split):,} proteins)")
    log.info(f"{'─'*60}")

    sequences = df_split["sequence"].tolist()
    accessions = df_split["accession"].tolist()
    splits_col = df_split["split"].tolist()
    lengths    = [len(s) for s in sequences]

    # build token-count-aware batches over position indices
    all_indices = list(range(len(sequences)))
    batches = build_batches(sequences, all_indices, args.max_tokens)

    log.info(f"  Batches: {len(batches):,}  "
             f"(max_tokens={args.max_tokens})")

    shard_idx  = 0
    shard      = _empty_shard()
    n_written  = 0   # proteins written to disk
    n_errors   = 0
    shard_map: list[str] = []  # shard file → list of accessions (for metadata)

    # tqdm progress bar over batches
    pbar = tqdm_auto(
        batches,
        desc=f"{split:5s}",
        unit="batch",
        dynamic_ncols=True,
    )

    for batch_positions in pbar:
        batch_seqs = [sequences[i] for i in batch_positions]
        avg_len    = sum(len(s) for s in batch_seqs) / len(batch_seqs)
        pbar.set_postfix(avg_len=f"{avg_len:.0f}", shard=shard_idx)

        try:
            per_res, mean_emb = extract_batch(
                model, tokenizer, batch_seqs, device, args.fp16
            )
        except RuntimeError as exc:
            # OOM or other GPU error — log and skip batch
            log.error(f"  Batch error (shard {shard_idx}): {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            n_errors += len(batch_positions)
            continue

        # append to current shard
        for local_i, global_i in enumerate(batch_positions):
            emb  = per_res[local_i]
            mean = mean_emb[local_i]

            # safety: verify embedding length matches sequence length
            seq_len = len(sequences[global_i])
            if emb.shape[0] != seq_len:
                log.warning(
                    f"  Length mismatch for {accessions[global_i]}: "
                    f"seq={seq_len}, emb={emb.shape[0]} — skipping"
                )
                n_errors += 1
                continue

            shard["accessions"].append(accessions[global_i])
            shard["splits"].append(splits_col[global_i])
            shard["lengths"].append(lengths[global_i])
            shard["embeddings"].append(emb)
            shard["mean_embeddings"].append(mean)

        # flush shard when full
        if len(shard["accessions"]) >= args.shard_size:
            shard_path = outdir / f"{split}_shard_{shard_idx:03d}.pt"
            save_shard(shard, shard_path)
            shard_map.append(shard_path.name)
            n_written += len(shard["accessions"])
            shard_idx += 1
            shard = _empty_shard()

            # periodically flush GPU cache to avoid fragmentation
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # flush remaining proteins into the last (possibly partial) shard
    if shard["accessions"]:
        shard_path = outdir / f"{split}_shard_{shard_idx:03d}.pt"
        save_shard(shard, shard_path)
        shard_map.append(shard_path.name)
        n_written += len(shard["accessions"])
        shard_idx += 1

    return {
        "split":          split,
        "n_input":        len(df_split),
        "n_written":      n_written,
        "n_errors":       n_errors,
        "n_shards":       shard_idx,
        "shards":         shard_map,
    }


# =============================================================================
#  METADATA
# =============================================================================

def save_metadata(
    outdir: Path,
    args: argparse.Namespace,
    hidden_dim: int,
    device: "torch.device",
    results: list[dict],
) -> None:
    """Write run_metadata.json to the output directory."""
    meta = {
        "model":          args.model,
        "run_timestamp":  datetime.now().isoformat(),
        "device":         str(device),
        "fp16":           args.fp16 and device.type == "cuda",
        "max_tokens":     args.max_tokens,
        "shard_size":     args.shard_size,
        "max_len":        args.max_len,
        "hidden_dim":     hidden_dim,
        "seed":           args.seed,
        "input_file":     str(args.input),
        "splits_processed": [r["split"] for r in results],
        "per_split": {
            r["split"]: {
                "n_input":   r["n_input"],
                "n_written": r["n_written"],
                "n_errors":  r["n_errors"],
                "n_shards":  r["n_shards"],
            }
            for r in results
        },
        "shard_format": {
            "keys": ["accessions", "splits", "lengths",
                     "embeddings", "mean_embeddings"],
            "embeddings_shape":      "[L, D]  per-residue, fp16, CPU",
            "mean_embeddings_shape": "[D]     mean-pooled, fp16, CPU",
        },
    }
    path = outdir / "run_metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Metadata saved: {path.name}")


# =============================================================================
#  KEY FINDINGS
# =============================================================================

def print_summary(
    results: list[dict],
    skipped: pd.DataFrame,
    t_start: float,
) -> None:
    elapsed = time.time() - t_start
    bar = "═" * 58

    print(f"\n{bar}")
    print("  SUMMARY — ESM-2 Embedding Extraction")
    print(bar)
    print(f"  {'split':<8}  {'input':>8}  {'written':>8}  "
          f"{'errors':>7}  {'shards':>7}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}")

    total_in = total_out = total_err = total_sh = 0
    for r in results:
        print(f"  {r['split']:<8}  {r['n_input']:>8,}  "
              f"{r['n_written']:>8,}  {r['n_errors']:>7,}  "
              f"{r['n_shards']:>7,}")
        total_in  += r["n_input"]
        total_out += r["n_written"]
        total_err += r["n_errors"]
        total_sh  += r["n_shards"]

    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}")
    print(f"  {'TOTAL':<8}  {total_in:>8,}  {total_out:>8,}  "
          f"{total_err:>7,}  {total_sh:>7,}")
    print()
    print(f"  Skipped at cleaning stage : {len(skipped):,}")
    print(f"  Total runtime             : {elapsed/60:.1f} min  "
          f"({elapsed:.0f} s)")
    print()

    if total_err == 0 and len(skipped) == 0:
        print("  ✅ All proteins processed successfully.")
    else:
        if total_err:
            print(f"  ⚠  {total_err:,} proteins failed during inference — "
                  "see logs above.")
        if len(skipped):
            print(f"  ⚠  {len(skipped):,} proteins skipped at cleaning — "
                  "see skipped_proteins.tsv.")

    print(bar)


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    global torch, AutoTokenizer, AutoModel, tqdm_auto

    args = parse_args()

    # ── deferred heavy imports ───────────────────────────────────────────────
    try:
        import torch as _torch
        from transformers import AutoTokenizer as _AT, AutoModel as _AM
        from tqdm.auto import tqdm as _tqdm
        torch = _torch
        AutoTokenizer = _AT
        AutoModel = _AM
        tqdm_auto = _tqdm
    except ImportError as e:
        sys.exit(
            f"\n[ERROR] Missing dependency: {e}\n"
            "Install with:  pip install torch transformers tqdm\n"
        )

    # ── reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto-enable fp16 on CUDA unless user explicitly set it
    if device.type == "cuda" and not args.fp16:
        args.fp16 = True
        log.info("Auto-enabling FP16 on CUDA device")

    # ── output directory ─────────────────────────────────────────────────────
    args.outdir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {args.outdir}")

    t_start = time.time()

    # ── load & validate input ────────────────────────────────────────────────
    df = load_input(args.input)
    validate_input(df)

    # ── clean sequences ──────────────────────────────────────────────────────
    df, skipped = clean_sequences(df, args.max_len)

    # save skipped proteins for inspection
    if not skipped.empty:
        skip_path = args.outdir / "skipped_proteins.tsv"
        skipped.to_csv(skip_path, sep="\t", index=False)
        log.warning(f"  Skipped proteins saved: {skip_path.name}")

    # ── load model ───────────────────────────────────────────────────────────
    model, tokenizer, hidden_dim = load_model_and_tokenizer(
        args.model, device, args.fp16, args.cache_dir
    )

    # ── process each split ───────────────────────────────────────────────────
    results: list[dict] = []

    for split in args.splits:
        df_split = df[df["split"] == split].reset_index(drop=True)
        if df_split.empty:
            log.warning(f"  Split '{split}' is empty — skipping")
            continue

        # resume: skip if all shards already written
        if args.resume:
            n_existing = count_existing_shards(args.outdir, split)
            n_expected = (len(df_split) + args.shard_size - 1) // args.shard_size
            if n_existing >= n_expected:
                log.info(f"  Resuming: {split} already has {n_existing} shards — skipping")
                results.append({
                    "split": split, "n_input": len(df_split),
                    "n_written": len(df_split), "n_errors": 0,
                    "n_shards": n_existing, "shards": [],
                })
                continue

        result = process_split(
            split, df_split, model, tokenizer, device, args, args.outdir
        )
        results.append(result)

    # ── metadata ─────────────────────────────────────────────────────────────
    save_metadata(args.outdir, args, hidden_dim, device, results)

    # ── summary ──────────────────────────────────────────────────────────────
    print_summary(results, skipped, t_start)


if __name__ == "__main__":
    main()