"""
Extract per-residue attention evidence from ESM-2 for proteins where
top SAE features showed strong alignment with biological annotations.

What we extract
---------------
For each protein, we run ESM-2 forward with output_attentions=True and
compute, for each residue position i:

  attention_to[i]   = mean attention received by residue i from all other
                       positions, aggregated across all heads and all layers
                       (or a selected subset).  Shape: [L] per protein.

  attention_from[i] = mean attention sent from residue i to all others.

  head_max[i]       = max attention received across all (layer, head) pairs —
                       which head attends most strongly to this residue?

We then run the same alignment pipeline as feature_alignment.py:
  - For each (attention_head_or_agg, annotation_type) pair:
    compute AUPRC, enrichment, odds ratio
  - Compare convergence with SAE feature activations:
    do the heads that attend to annotated regions match the SAE features
    that activate there?
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import average_precision_score

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from constants import OUT_DIR, ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH

torch = None
AutoTokenizer = None
AutoModel = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ESM-2 650M architecture constants
ESM2_N_LAYERS = 33
ESM2_N_HEADS  = 20
DEFAULT_MODEL  = "facebook/esm2_t33_650M_UR50D"
DEFAULT_MAX_LEN = 1022


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract ESM-2 attention and align with annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--shard-dir",   type=Path,
                   default=OUT_DIR / "embeddings" / "esm2_650m")
    p.add_argument("--alignment",   type=Path,
                   default=OUT_DIR / "feature_alignment" / "per_annot_summary.tsv",
                   help="per_annot_summary.tsv — used to select proteins of interest.")
    p.add_argument("--sae-scores",  type=Path,
                   default=OUT_DIR / "feature_alignment" / "alignment_scores.parquet",
                   help="Full SAE alignment scores for convergence analysis.")
    p.add_argument("--annotations", type=Path,
                   default=ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH)
    p.add_argument("--model",       default=DEFAULT_MODEL)
    p.add_argument("--cache-dir",   type=Path, default=None)
    p.add_argument("--split",       default="test")
    p.add_argument("--top-pairs",   type=int, default=8,
                   help="Analyse top-N annotation types from alignment results.")
    p.add_argument("--n-circular-perms", type=int, default=50)
    p.add_argument("--max-proteins-per-type", type=int, default=200,
                   help="Max proteins per annotation type to process (speed).")
    p.add_argument("--outdir",      type=Path,
                   default=OUT_DIR / "attention")
    p.add_argument("--device",      default="auto")
    return p.parse_args()


# =============================================================================
#  ATTENTION AGGREGATION
# =============================================================================

def aggregate_attentions(
    attentions: tuple,          # tuple of [1, n_heads, L_tok, L_tok] per layer
    seq_len:    int,            # real sequence length (without special tokens)
) -> np.ndarray:
    """
    Aggregate raw attention tensors into per-residue attention profiles.

    ESM-2 attentions include special tokens ([CLS] at 0, [EOS] at end).
    We strip them and aggregate across layers and heads.

    Returns array of shape [seq_len, n_agg] where n_agg columns are:
      0: mean_to        mean attention received per residue (all layers, all heads)
      1: mean_from      mean attention sent per residue
      2: max_to         max attention received (over layers × heads)
      3: last_layer_to  mean attention received in the last layer only
      plus one column per layer: mean_to for that layer (33 cols)
    Each column is normalised to [0, 1] within the protein.
    """
    # attentions: tuple of n_layers tensors, each [1, n_heads, L_tok, L_tok]
    # Filter out None entries — some ESM-2 implementations (e.g. with flash
    # attention or certain transformers versions) return None for some layers.
    valid = [a for a in attentions if a is not None]
    if not valid:
        raise RuntimeError(
            "All attention tensors are None. The model may be using an "
            "attention implementation that does not support output_attentions=True. "
            "Try loading the model with attn_implementation='eager'."
        )
    n_layers = len(valid)
    L_tok    = valid[0].shape[-1]   # includes [CLS] and [EOS]
    # real residue indices: 1 .. seq_len (0 is CLS, seq_len+1 is EOS)
    r_start, r_end = 1, 1 + seq_len

    # stack valid layers: [n_layers, n_heads, L_tok, L_tok]
    attn_stack = np.stack([
        a[0].cpu().float().numpy()   # [n_heads, L_tok, L_tok]
        for a in valid
    ], axis=0)  # [n_layers, n_heads, L_tok, L_tok]

    # slice to residue-only rows and columns
    attn_res = attn_stack[:, :, r_start:r_end, r_start:r_end]
    # shape: [n_layers, n_heads, seq_len, seq_len]

    # "to" = column sum (attention flowing INTO residue i from all others)
    # "from" = row sum (attention flowing FROM residue i to all others)
    to_all   = attn_res.sum(axis=3)   # [n_layers, n_heads, seq_len]
    from_all = attn_res.sum(axis=2)   # [n_layers, n_heads, seq_len]

    cols = []

    # col 0: mean_to across all layers and heads
    mean_to = to_all.mean(axis=(0, 1))          # [seq_len]
    cols.append(_norm(mean_to))

    # col 1: mean_from
    mean_from = from_all.mean(axis=(0, 1))      # [seq_len]
    cols.append(_norm(mean_from))

    # col 2: max_to across layers × heads
    max_to = to_all.max(axis=(0, 1))            # [seq_len]
    cols.append(_norm(max_to))

    # col 3: last layer mean_to
    last_to = to_all[-1].mean(axis=0)           # [seq_len]
    cols.append(_norm(last_to))

    # col 4 .. 4+n_layers-1: per-layer mean_to
    for layer_i in range(n_layers):
        layer_to = to_all[layer_i].mean(axis=0) # [seq_len]
        cols.append(_norm(layer_to))

    return np.stack(cols, axis=1).astype(np.float32)
    # shape: [seq_len, 4 + n_layers]


def _norm(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    rng = x.max() - x.min()
    return (x - x.min()) / (rng + 1e-8)


def agg_col_names(n_layers: int) -> list[str]:
    """Column names matching aggregate_attentions output."""
    return (
        ["mean_to", "mean_from", "max_to", "last_layer_to"] +
        [f"layer_{i:02d}_to" for i in range(n_layers)]
    )


# =============================================================================
#  ESM-2 FORWARD PASS WITH ATTENTION
# =============================================================================

def extract_attention_for_protein(
    seq:       str,
    model,
    tokenizer,
    device,
    max_len:   int = DEFAULT_MAX_LEN,
) -> np.ndarray:
    """
    Run ESM-2 forward pass for one protein and return aggregated attention.
    Returns [L, n_agg] float32 numpy array.
    """
    seq = seq[:max_len]
    L   = len(seq)

    with torch.no_grad():
        inputs = tokenizer(
            seq, return_tensors="pt", truncation=True,
            max_length=max_len + 2,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=False,
        )

    # outputs.attentions: tuple of n_layers tensors [1, n_heads, L_tok, L_tok]
    # Guard: some transformer versions return None or empty tuple for attentions
    # even with output_attentions=True — check before proceeding.
    if outputs.attentions is None or len(outputs.attentions) == 0:
        raise RuntimeError(
            "Model returned no attention tensors. "
            "Ensure the model supports output_attentions=True "
            "(ESM-2 from transformers>=4.24 does). "
            f"Got: {type(outputs.attentions)}"
        )
    # Also guard: ESM-2 sometimes wraps attentions in an extra tuple layer
    attentions = outputs.attentions
    if isinstance(attentions[0], (tuple, list)):
        attentions = attentions[0]
    if len(attentions) == 0:
        raise RuntimeError("Empty attentions after unwrapping.")

    attn_agg = aggregate_attentions(attentions, L)
    return attn_agg   # [L, n_agg]


# =============================================================================
#  ALIGNMENT (same metrics as feature_alignment.py)
# =============================================================================

def _circ_enr(act_binary, ann_mask, tp, n_perms):
    L = len(act_binary)
    if n_perms <= 0 or act_binary.sum() == 0:
        return float("nan")
    shifts  = np.random.randint(1, L, size=n_perms)
    idx     = (np.arange(L)[None, :] - shifts[:, None]) % L
    perms   = act_binary[idx]
    expected = (perms * ann_mask[None, :]).sum(axis=1).mean()
    return float(tp / expected) if expected > 1e-8 else float("nan")


def compute_alignment(act_scores, ann_mask, threshold, n_perms):
    if ann_mask.sum() == 0:
        return None
    binary = (act_scores > threshold).astype(np.float32)
    if binary.sum() == 0:
        return None
    try:
        auprc = float(average_precision_score(ann_mask, act_scores))
    except Exception:
        auprc = float("nan")
    tp = float((binary * ann_mask).sum())
    fp = float((binary * (1 - ann_mask)).sum())
    fn = float(((1 - binary) * ann_mask).sum())
    tn = float(((1 - binary) * (1 - ann_mask)).sum())
    if tp > 0:
        try:
            or_, pv = sp_stats.fisher_exact(
                np.array([[tp, fp], [fn, tn]]), alternative="greater")
        except Exception:
            or_, pv = float("nan"), float("nan")
    else:
        or_, pv = 0.0, 1.0
    enr = _circ_enr(binary, ann_mask, tp, n_perms)
    return {"auprc": auprc, "enrichment": enr, "odds_ratio": or_,
            "fisher_p": pv, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def aggregate_metrics(records):
    if not records:
        return {}
    auprcs = [r["auprc"] for r in records if not np.isnan(r["auprc"])]
    enrs   = [r["enrichment"] for r in records
              if not np.isnan(r["enrichment"]) and r["enrichment"] > 0]
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    tn = sum(r["tn"] for r in records)
    try:
        or_, pv = sp_stats.fisher_exact(np.array([[tp, fp], [fn, tn]]),
                                         alternative="greater")
    except Exception:
        or_, pv = float("nan"), float("nan")
    return {
        "auprc_mean":       float(np.mean(auprcs))   if auprcs else float("nan"),
        "auprc_median":     float(np.median(auprcs)) if auprcs else float("nan"),
        "enrichment_gmean": float(np.exp(np.mean(np.log(enrs)))) if enrs else float("nan"),
        "odds_ratio":       float(or_),
        "fisher_p":         float(pv),
        "n_proteins":       len(records),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# =============================================================================
#  HELPERS
# =============================================================================

def build_mask(ann_sub, L):
    mask = np.zeros(L, dtype=np.float32)
    for _, row in ann_sub.iterrows():
        s = max(0, int(row["start"]) - 1)
        e = min(L, int(row["end"]))
        mask[s:e] = 1.0
    return mask


def load_protein_sequences(shard_dir, split, accs_needed):
    """
    Load sequences for a specific set of accessions from shards.
    Returns dict: accession → sequence string.
    Sequences are recovered from the tokenizer — here we just return
    embeddings length since sequences come from the proteins TSV.
    """
    # We don't store sequences in shards; caller should pass seq_by_acc
    pass


# =============================================================================
#  MAIN
# =============================================================================

def main():
    global torch, AutoTokenizer, AutoModel

    args = parse_args()

    try:
        import torch as _t
        from transformers import AutoTokenizer as _AT, AutoModel as _AM
        torch = _t; AutoTokenizer = _AT; AutoModel = _AM
    except ImportError as e:
        sys.exit(f"\n[ERROR] {e}\npip install torch transformers\n")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    log.info(f"Device: {device}")

    # ── load model ────────────────────────────────────────────────────────────
    log.info(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    model = AutoModel.from_pretrained(
        args.model,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        attn_implementation="eager",   # required for output_attentions=True
    ).to(device).eval()
    log.info("  Model loaded")

    # ── load annotations ──────────────────────────────────────────────────────
    log.info("Loading annotations ...")
    ann_df = pd.read_csv(args.annotations, sep="\t", low_memory=False)
    ann_df.columns = ann_df.columns.str.strip().str.lower()
    ann_df = ann_df[ann_df["split"] == args.split].copy()
    ann_df["start"] = ann_df["start"].astype(int)
    ann_df["end"]   = ann_df["end"].astype(int)

    # use annot_subtype as feature_type when available
    if ("annot_subtype" in ann_df.columns
            and ann_df["annot_subtype"].notna().any()
            and ann_df["annot_subtype"].nunique() > ann_df["feature_type"].nunique()):
        ann_df["feature_type"] = ann_df["annot_subtype"]
        log.info(f"  Using annot_subtype as feature_type "
                 f"({ann_df['feature_type'].nunique()} subtypes)")

    ann_by_acc = {acc: grp for acc, grp in ann_df.groupby("accession")}

    # ── load alignment summary to pick annotation types ───────────────────────
    log.info("Loading alignment summary ...")
    align_df = pd.read_csv(args.alignment, sep="\t")
    align_df = align_df.sort_values("auprc_mean", ascending=False)
    target_types = align_df["annot_type"].tolist()[:args.top_pairs]
    log.info(f"  Target annotation types: {target_types}")

    # ── load SAE alignment scores for convergence ─────────────────────────────
    sae_scores_df = None
    if args.sae_scores.exists():
        sae_scores_df = pd.read_parquet(args.sae_scores)

    # ── collect proteins of interest ──────────────────────────────────────────
    ann_filtered = ann_df[ann_df["feature_type"].isin(target_types)]

    # Cap proteins per annotation type to keep runtime tractable.
    # 100 proteins per type is sufficient for stable AUPRC estimates.
    capped_accs = set()
    for at in target_types:
        type_accs = ann_filtered[ann_filtered["feature_type"] == at]["accession"].unique()
        capped_accs.update(type_accs[:args.max_proteins_per_type])
    target_accs = capped_accs
    log.info(f"  Capped to {len(target_accs):,} unique proteins "
             f"({args.max_proteins_per_type} max per annotation type)")

    # load sequences directly from proteins TSV (streaming — no shards needed)
    proteins_path = args.annotations.parent / "proteins_with_split.tsv"
    if not proteins_path.exists():
        proteins_path = args.annotations.parent.parent / "data" / "proteins_with_split.tsv"
    if not proteins_path.exists():
        sys.exit(f"[ERROR] proteins_with_split.tsv not found near {args.annotations}")

    log.info(f"Loading sequences from {proteins_path.name} ...")
    prot_df = pd.read_csv(proteins_path, sep="\t", low_memory=False)
    prot_df = prot_df[prot_df["split"] == args.split]
    seq_map = dict(zip(prot_df["accession"], prot_df["sequence"]))

    acc_to_seq = {a: seq_map[a] for a in target_accs if a in seq_map}
    log.info(f"  {len(acc_to_seq):,} proteins with sequences found")

    # ── column names for aggregated attention ─────────────────────────────────
    col_names = agg_col_names(ESM2_N_LAYERS)   # 4 + 33 = 37 columns
    n_agg     = len(col_names)

    # ── pass 1: extract attention for all target proteins ─────────────────────
    log.info(f"\nExtracting attention for {len(acc_to_seq):,} proteins ...")
    attn_cache: dict[str, np.ndarray] = {}   # acc → [L, n_agg]

    accs_list = list(acc_to_seq.keys())
    pbar = (_tqdm(accs_list, desc="Extract attn", unit="prot",
                  dynamic_ncols=True)
            if _tqdm else accs_list)

    for acc in pbar:
        seq = acc_to_seq[acc]
        if not seq:
            continue
        try:
            attn_agg = extract_attention_for_protein(
                seq, model, tokenizer, device)
            attn_cache[acc] = attn_agg   # [L, n_agg]
        except Exception as e:
            log.warning(f"  Failed {acc}: {e}")
            continue
        finally:
            # flush GPU cache after each protein to prevent fragmentation
            # (eager attention materialises large [L,L] tensors per layer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log.info(f"  Extracted attention for {len(attn_cache):,} proteins")

    # ── pass 2: align each aggregation column with each annotation type ───────
    log.info(f"\nAligning {n_agg} attention aggregations × "
             f"{len(target_types)} annotation types ...")

    # threshold for binarising attention: top-10% of residues
    ATTN_THRESHOLD_QUANTILE = 0.90

    records: dict = defaultdict(lambda: defaultdict(list))

    pbar2 = (_tqdm(attn_cache.items(), desc="Align attn", unit="prot",
                   dynamic_ncols=True)
             if _tqdm else attn_cache.items())

    for acc, attn_agg in pbar2:
        L = attn_agg.shape[0]
        if acc not in ann_by_acc:
            continue

        for annot_type in target_types:
            ann_sub = ann_by_acc[acc][ann_by_acc[acc]["feature_type"] == annot_type]
            if ann_sub.empty:
                continue
            mask = build_mask(ann_sub, L)
            if mask.sum() == 0:
                continue

            for col_i, col_name in enumerate(col_names):
                scores = attn_agg[:, col_i]
                # binarise at protein-level 90th percentile
                threshold = float(np.quantile(scores, ATTN_THRESHOLD_QUANTILE))
                m = compute_alignment(scores, mask, threshold,
                                      args.n_circular_perms)
                if m is not None:
                    records[col_name][annot_type].append(m)

    # ── aggregate and build results ───────────────────────────────────────────
    log.info("\nAggregating ...")
    rows = []
    for col_name in col_names:
        for annot_type in target_types:
            recs = records[col_name].get(annot_type, [])
            if not recs:
                continue
            agg = aggregate_metrics(recs)
            if not agg:
                continue
            rows.append({"attn_agg": col_name, "annot_type": annot_type, **agg})

    results_df = pd.DataFrame(rows)
    results_df.to_parquet(args.outdir / "alignment_scores.parquet", index=False)

    # best aggregation per annotation type
    per_annot = (results_df
                 .sort_values("auprc_mean", ascending=False)
                 .groupby("annot_type")
                 .first()
                 .reset_index()
                 .sort_values("auprc_mean", ascending=False))
    per_annot.to_csv(args.outdir / "per_annot_summary.tsv", sep="\t", index=False)

    # ── convergence: SAE feature vs best attention head ───────────────────────
    if sae_scores_df is not None:
        _save_convergence(per_annot, align_df, sae_scores_df, args.outdir)

    # ── console summary ───────────────────────────────────────────────────────
    log.info("\n" + "═"*60)
    log.info("  ATTENTION ALIGNMENT RESULTS")
    log.info("═"*60)
    print(per_annot[["annot_type","attn_agg","auprc_mean",
                      "enrichment_gmean","odds_ratio","n_proteins"]].to_string(index=False))

    # ── metadata ──────────────────────────────────────────────────────────────
    meta = {
        "model":           args.model,
        "split":           args.split,
        "n_proteins":      len(attn_cache),
        "n_annot_types":   len(target_types),
        "n_attn_agg_cols": n_agg,
        "attn_threshold_quantile": ATTN_THRESHOLD_QUANTILE,
        "n_circular_perms": args.n_circular_perms,
        "col_names":       col_names,
    }
    with open(args.outdir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"\n✅ Done. Outputs in {args.outdir}")


# =============================================================================
#  CONVERGENCE ANALYSIS
# =============================================================================

def _save_convergence(
    attn_per_annot:  pd.DataFrame,   # best attention agg per annot_type
    sae_per_annot:   pd.DataFrame,   # best SAE feature per annot_type
    sae_scores_df:   pd.DataFrame,   # full SAE alignment scores
    outdir:          Path,
) -> None:
    """
    Build a side-by-side table:
      annot_type | best_attn_agg | attn_auprc | best_sae_feature | sae_auprc
                 | convergence_score

    convergence_score = harmonic mean of attn_auprc and sae_auprc.
    High convergence = both attention and SAE independently identify the
    same biological region → stronger multi-evidence claim.
    """
    rows = []
    for _, attn_row in attn_per_annot.iterrows():
        at = attn_row["annot_type"]
        sae_row = sae_per_annot[sae_per_annot["annot_type"] == at]
        if sae_row.empty:
            continue
        sae_row = sae_row.iloc[0]

        attn_auprc = float(attn_row["auprc_mean"])
        sae_auprc  = float(sae_row["auprc_mean"])
        # harmonic mean of the two AUPRC scores
        h_mean = (2 * attn_auprc * sae_auprc / (attn_auprc + sae_auprc)
                  if (attn_auprc + sae_auprc) > 0 else 0.0)

        rows.append({
            "annot_type":       at,
            "best_attn_agg":    attn_row["attn_agg"],
            "attn_auprc":       attn_auprc,
            "attn_or":          float(attn_row["odds_ratio"]),
            "best_sae_feature": int(sae_row["feature_id"]),
            "sae_auprc":        sae_auprc,
            "sae_or":           float(sae_row["odds_ratio"]),
            "convergence_score": h_mean,
        })

    conv_df = pd.DataFrame(rows).sort_values("convergence_score", ascending=False)
    conv_df.to_csv(outdir / "convergence_with_sae.tsv", sep="\t", index=False)

    log.info("\n  SAE × Attention convergence:")
    print(conv_df[["annot_type","best_attn_agg","attn_auprc",
                   "best_sae_feature","sae_auprc","convergence_score"]].to_string(index=False))


if __name__ == "__main__":
    main()