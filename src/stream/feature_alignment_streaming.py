"""
feature_alignment_streaming.py — PIPLV2 / sae
===============================================
Streaming SAE feature alignment — fully GPU-batched, pass-2 checkpointed every 10%.

Performance design
------------------
  - DynamicBatchPrefetcher: buckets proteins by length → ESM-2 batch → SAE batch
    in a background thread. buffer_size = align_batch_size x 3 so the GPU
    never idles waiting for the prefetcher.
  - compute_alignment_multiprotein: accumulates align_batch_size proteins then
    does one GPU pass for all features x all annotation types — no per-protein
    kernel launches.
  - No checkpoint I/O during the run (saves ~1 min/checkpoint at 30 min total).
    Results written once at the end.
"""

import sys
import json
import logging
import argparse
import warnings
import queue
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from constants import OUT_DIR, ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH

torch = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Streaming SAE feature alignment — fully GPU-batched.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--esm2-model",   default="esm2_t33_650M_UR50D")
    p.add_argument("--checkpoint",   type=Path, required=True)
    p.add_argument("--annotations",  type=Path,
                   default=ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH)
    p.add_argument("--fasta",        type=Path, required=True)
    p.add_argument("--proteins",     type=Path,
                   default=OUT_DIR.parent / "data" / "proteins_with_split.tsv")
    p.add_argument("--split",        default="test")
    # feature selection
    p.add_argument("--all-features", action="store_true",
                   help="Evaluate all K features.")
    p.add_argument("--top-k-features", type=int, default=200)
    p.add_argument("--select-by",    default="variance",
                   choices=["mean", "variance", "max", "sparsity"])
    # filters
    p.add_argument("--min-proteins",         type=int,   default=30)
    p.add_argument("--min-annot-residues",   type=int,   default=500)
    p.add_argument("--activation-threshold", type=float, default=0.1)
    # novel triage
    p.add_argument("--novel-auprc-threshold", type=float, default=0.25)
    p.add_argument("--fast-triage",           action="store_true")
    p.add_argument("--triage-max-proteins",   type=int,   default=5000)
    # batching
    p.add_argument("--esm-batch-size",   type=int, default=16,
                   help="Proteins per ESM-2 forward pass. "
                        "16 = sweet spot for A5500 + Swiss-Prot (~350 aa avg). "
                        "Reduce to 8 if OOM.")
    p.add_argument("--align-batch-size", type=int, default=128,
                   help="Proteins accumulated before one GPU alignment call. "
                        "Higher = better utilization. Prefetcher buffer = 3×.")
    p.add_argument("--bucket-width",     type=int, default=64,
                   help="Sequence length bucket width for ESM-2 batching.")
    # output
    p.add_argument("--outdir",      type=Path,
                   default=OUT_DIR / "feature_alignment_streaming")
    p.add_argument("--device",      default="auto")
    p.add_argument("--max-seq-len", type=int, default=1022)
    # sharding (parallel pass-2 across multiple GPUs)
    p.add_argument("--num-shards",   type=int, default=1,
                   help="Split pass-2 proteins across N parallel shards. "
                        "Pass-1 must already be cached (pass1_stats.npz).")
    p.add_argument("--shard-idx",    type=int, default=0,
                   help="Index of this shard (0-based).")
    p.add_argument("--merge-shards", action="store_true",
                   help="Merge pass2_shard_*.pkl files and produce final outputs.")
    return p.parse_args()


# =============================================================================
#  ESM-2
# =============================================================================

def load_esm2(model_name, device):
    log.info(f"Loading ESM-2 ({model_name}) on {device} ...")
    try:
        import esm as _esm
        model, alphabet = _esm.pretrained.load_model_and_alphabet(model_name)
        bc = alphabet.get_batch_converter()
        model.eval().to(device)
        log.info(f"  fair-esm — device: {next(model.parameters()).device}")
        return model, bc, "esm"
    except Exception:
        log.info("  fair-esm not found, trying HuggingFace ...")
        from transformers import EsmModel, EsmTokenizer
        tok   = EsmTokenizer.from_pretrained(f"facebook/{model_name}")
        model = EsmModel.from_pretrained(f"facebook/{model_name}")
        model.eval().to(device)
        log.info(f"  HuggingFace — device: {next(model.parameters()).device}")
        return model, tok, "hf"


def get_embedding_esm_batch(batch_data, model, converter, backend,
                             device, layer=33, max_len=1022):
    """
    batch_data : list of (acc, seq)
    Returns    : list of (acc, emb [L, D] float32)
    """
    results = []
    if backend == "esm":
        data = [(acc, seq[:max_len]) for acc, seq in batch_data]
        _, _, tokens = converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[layer], return_contacts=False)
        reps = out["representations"][layer]           # [B, Lmax+2, D]
        for i, (acc, seq) in enumerate(batch_data):
            L   = min(len(seq), max_len)
            emb = reps[i, 1:L+1, :].cpu().numpy().astype(np.float32)
            results.append((acc, emb))
    else:
        seqs   = [seq[:max_len] for _, seq in batch_data]
        inputs = converter(seqs, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len + 2)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        reps = out.last_hidden_state                   # [B, Lmax+2, D]
        for i, (acc, seq) in enumerate(batch_data):
            L   = min(len(seq), max_len)
            emb = reps[i, 1:L+1, :].cpu().numpy().astype(np.float32)
            results.append((acc, emb))
    return results


# =============================================================================
#  SAE
# =============================================================================

def load_sae_encoder(checkpoint_path, device):
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd    = ckpt["model_state_dict"]
    W_enc = sd["encoder.weight"].float().to(device)
    b_enc = sd["encoder.bias"].float().to(device)
    K, D  = W_enc.shape
    cfg   = ckpt.get("config", {})
    topk_k = cfg.get("topk_k", None) if cfg.get("architecture") == "topk" else None
    log.info(f"  SAE: D={D}, K={K}, topk_k={topk_k} | config: {cfg}")
    return W_enc, b_enc, K, D, topk_k


def encode_residues_batch(emb_list, W_enc, b_enc, topk_k=None):
    """
    emb_list : list of [Li, D] float32 numpy
    topk_k   : if set, apply TopK sparsity gate (keep only top-k per token)
    Returns  : list of [Li, K] float32 numpy
    One matmul for the whole list.
    """
    sizes = [e.shape[0] for e in emb_list]
    x_cat = np.concatenate(emb_list, axis=0)
    with torch.no_grad():
        x_t = torch.from_numpy(x_cat).float().to(W_enc.device)
        z_t = torch.relu(x_t @ W_enc.T + b_enc)
        if topk_k is not None:
            topk_vals, topk_idx = z_t.topk(topk_k, dim=-1)
            z_t = torch.zeros_like(z_t)
            z_t.scatter_(-1, topk_idx, topk_vals)
    z_cat = z_t.cpu().numpy().astype(np.float32)
    out, off = [], 0
    for s in sizes:
        out.append(z_cat[off:off+s])
        off += s
    return out


# =============================================================================
#  DYNAMIC BATCH PREFETCHER
# =============================================================================

class DynamicBatchPrefetcher:
    """
    Background thread: bucket proteins by length → ESM-2 batch → SAE batch.
    buffer_size should be set to align_batch_size × 3 so the GPU never idles.
    Yields (acc, emb [L,D], z [L,K], err).
    """

    def __init__(self, accs, seqs, esm_model, converter, backend,
                 device, W_enc, b_enc, layer=33, max_len=1022,
                 batch_size=16, bucket_width=64, buffer_size=384, topk_k=None):
        self.q         = queue.Queue(maxsize=buffer_size)
        self._sentinel = object()
        threading.Thread(
            target=self._worker,
            args=(accs, seqs, esm_model, converter, backend,
                  device, W_enc, b_enc, layer, max_len, batch_size, bucket_width,
                  topk_k),
            daemon=True,
        ).start()

    @staticmethod
    def _make_batches(accs, seqs, max_len, batch_size, bucket_width):
        buckets = defaultdict(list)
        for acc in accs:
            seq = seqs.get(acc, "")
            L   = min(len(seq), max_len)
            if L == 0:
                continue
            buckets[(L // bucket_width) * bucket_width].append((acc, seq))
        batches = []
        for bk in sorted(buckets):
            items = buckets[bk]
            for i in range(0, len(items), batch_size):
                batches.append(items[i:i+batch_size])
        return batches

    def _worker(self, accs, seqs, esm_model, converter, backend,
                device, W_enc, b_enc, layer, max_len, batch_size, bucket_width,
                topk_k=None):
        batches  = self._make_batches(accs, seqs, max_len, batch_size, bucket_width)
        n_failed = 0
        for batch_data in batches:
            try:
                emb_results = get_embedding_esm_batch(
                    batch_data, esm_model, converter, backend, device, layer, max_len)
                z_list = encode_residues_batch(
                    [e for _, e in emb_results], W_enc, b_enc, topk_k=topk_k)
                for (acc, emb), z in zip(emb_results, z_list):
                    self.q.put((acc, emb, z, None))
            except Exception as e:
                n_failed += len(batch_data)
                if n_failed <= 3:
                    log.warning(f"  Batch failed: {e}")
                for acc, _ in batch_data:
                    self.q.put((acc, None, None, str(e)))
        self.q.put(self._sentinel)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is self._sentinel:
                break
            yield item


# =============================================================================
#  ANNOTATION MASK BUILDER
# =============================================================================

def build_masks_for_protein(ann_acc, L):
    """Returns {feat_type: np.ndarray [L] float32}. Vectorised, no iterrows."""
    masks = {}
    for feat_type, grp in ann_acc.groupby("feature_type"):
        mask   = np.zeros(L, dtype=np.float32)
        starts = np.clip(grp["start"].values.astype(int) - 1, 0, L)
        ends   = np.clip(grp["end"].values.astype(int),       0, L)
        for s, e in zip(starts, ends):
            if e > s:
                mask[s:e] = 1.0
        if mask.sum() > 0:
            masks[feat_type] = mask
    return masks


# =============================================================================
#  ALIGNMENT — MULTI-PROTEIN BATCH
# =============================================================================

def _gpu_auprc_batch(scores_t, mask_t):
    """
    scores_t : [L, K] float32 tensor
    mask_t   : [L]    float32 tensor
    Returns  : [K]    float32 tensor
    """
    L, K  = scores_t.shape
    n_pos = mask_t.sum()
    if n_pos == 0:
        return torch.zeros(K, device=scores_t.device)
    idx      = torch.argsort(scores_t, dim=0, descending=True)
    s_mask   = mask_t[idx]
    cum_tp   = s_mask.cumsum(0)
    cum_n    = torch.arange(1, L+1, device=scores_t.device,
                             dtype=torch.float32).unsqueeze(1)
    prec     = cum_tp / cum_n
    rec      = cum_tp / n_pos
    prec_pad = torch.cat([torch.ones(1, K,  device=scores_t.device), prec], 0)
    rec_pad  = torch.cat([torch.zeros(1, K, device=scores_t.device), rec],  0)
    is_tp    = torch.cat([s_mask[:1], s_mask], 0)
    delta_r  = (rec_pad[1:] - rec_pad[:-1]) * is_tp[1:]
    return (prec_pad[1:] * delta_r).sum(0)


def compute_alignment_multiprotein(protein_buffer, eval_feature_ids,
                                   threshold, or_min, device,
                                   acc_to_cluster=None):
    """
    One GPU pass for a buffer of N proteins — fully vectorised 3D ops.

    Para cada annotation type: empilha masks de todas as proteínas em [N, L_max]
    e activations em [N, L_max, K], faz TP/FP/OR em um único kernel 3D.
    Zero loop Python sobre proteínas — um kernel por annotation type.
    """
    # índice: annot_type → [(pidx, mask_np)]
    annot_to_proteins = defaultdict(list)
    for pidx, pdata in enumerate(protein_buffer):
        for feat_type, mask in pdata["masks"].items():
            annot_to_proteins[feat_type].append((pidx, mask))

    # move activations para GPU uma vez só (só eval features)
    z_tensors = [
        torch.from_numpy(pdata["z"][:, eval_feature_ids]).to(device)
        for pdata in protein_buffer
    ]
    lengths = [z.shape[0] for z in z_tensors]
    K_eval  = len(eval_feature_ids)
    results = defaultdict(lambda: defaultdict(list))

    for annot_type, prot_mask_pairs in annot_to_proteins.items():
        if not prot_mask_pairs:
            continue

        pidxs     = [p for p, _ in prot_mask_pairs]
        mask_list = [m for _, m in prot_mask_pairs]
        N         = len(pidxs)
        L_max     = max(lengths[p] for p in pidxs)

        # empilha masks → [N, L_max] e activations → [N, L_max, K_eval]
        masks_t = torch.zeros(N, L_max, device=device)
        acts_t  = torch.zeros(N, L_max, K_eval, device=device)
        for i, (pidx, mask_np) in enumerate(zip(pidxs, mask_list)):
            L = lengths[pidx]
            masks_t[i, :L] = torch.from_numpy(mask_np).to(device)
            acts_t[i, :L]  = z_tensors[pidx]

        # [N, L_max, K_eval] → binary activations
        act_bin = (acts_t > threshold).float()           # [N, L_max, K_eval]
        not_m   = 1.0 - masks_t                          # [N, L_max]

        # TP/FP/FN/TN vectorizados — [N, K_eval]
        tp_t = (act_bin * masks_t.unsqueeze(2)).sum(1)
        fp_t = (act_bin * not_m.unsqueeze(2)).sum(1)
        n_pos = masks_t.sum(1, keepdim=True)              # [N, 1]
        n_neg = L_max - n_pos
        fn_t  = n_pos - tp_t
        tn_t  = n_neg - fp_t

        # OR vectorizado — [N, K_eval]
        denom = fp_t * fn_t
        or_t  = torch.where(
            denom > 0,
            tp_t * tn_t / denom,
            torch.where(tp_t > 0,
                        torch.full_like(tp_t, float("inf")),
                        torch.zeros_like(tp_t))
        )

        # por proteína: só computa AUPRC onde OR >= or_min
        tp_c  = tp_t.cpu().numpy()
        fp_c  = fp_t.cpu().numpy()
        fn_c  = fn_t.cpu().numpy()
        tn_c  = tn_t.cpu().numpy()
        or_c  = or_t.cpu().numpy()

        for i, pidx in enumerate(pidxs):
            cand_local = np.where(or_c[i] >= or_min)[0]
            if len(cand_local) == 0:
                continue

            # AUPRC só pros candidatos desta proteína
            z_cand  = acts_t[i, :lengths[pidx], :][:, cand_local]  # [Li, n_cand]
            mask_1d = masks_t[i, :lengths[pidx]]                    # [Li]
            auprc_t = _gpu_auprc_batch(z_cand, mask_1d)             # [n_cand]
            auprc_c = auprc_t.cpu().numpy()

            acc = protein_buffer[pidx]["acc"]
            cluster_id = (acc_to_cluster.get(acc, acc)
                          if acc_to_cluster is not None else acc)
            for j, local_idx in enumerate(cand_local):
                auprc = float(auprc_c[j])
                if auprc < 0.05:
                    continue
                global_fid = int(eval_feature_ids[local_idx])
                results[global_fid][annot_type].append({
                    "auprc":      auprc,
                    "cluster_id": cluster_id,
                    "enrichment": 1.0,
                    "odds_ratio": (float("inf") if not np.isfinite(or_c[i, local_idx])
                                   else float(or_c[i, local_idx])),
                    "fisher_p":   float("nan"),
                    "tp": float(tp_c[i, local_idx]),
                    "fp": float(fp_c[i, local_idx]),
                    "fn": float(fn_c[i, local_idx]),
                    "tn": float(tn_c[i, local_idx]),
                })

    return results


# =============================================================================
#  METRIC AGGREGATION
# =============================================================================

def aggregate_metrics(records):
    """Aggregate per-protein records into per-(feature, annot_type) metrics.

    To avoid inflating AUPRC with many highly similar proteins from the same
    cluster, we first average per-cluster (AUPRC, TP/FP/FN/TN, enrichment),
    then compute mean/median across clusters.  Fisher's exact test is run on
    the cluster-pooled contingency table.
    """
    if not records:
        return {}

    # ── group by cluster ──────────────────────────────────────────────────────
    from collections import defaultdict as _dd
    by_cluster = _dd(list)
    for r in records:
        by_cluster[r.get("cluster_id", "__singleton__")].append(r)

    cluster_auprcs = []
    cluster_enrs   = []
    tp = fp = fn = tn = 0.0
    n_proteins = len(records)

    for cid, crecs in by_cluster.items():
        c_auprcs = [r["auprc"]      for r in crecs if not np.isnan(r["auprc"])]
        c_enrs   = [r["enrichment"] for r in crecs
                    if not np.isnan(r["enrichment"]) and r["enrichment"] > 0]
        if c_auprcs:
            cluster_auprcs.append(float(np.mean(c_auprcs)))
        if c_enrs:
            cluster_enrs.append(float(np.mean(c_enrs)))
        # pool contingency counts within the cluster, then accumulate
        tp += sum(r["tp"] for r in crecs) / len(crecs)
        fp += sum(r["fp"] for r in crecs) / len(crecs)
        fn += sum(r["fn"] for r in crecs) / len(crecs)
        tn += sum(r["tn"] for r in crecs) / len(crecs)

    table = np.array([[tp, fp], [fn, tn]], dtype=np.float64)
    try:
        odds_ratio, fisher_p = sp_stats.fisher_exact(table, alternative="greater")
    except Exception:
        odds_ratio, fisher_p = float("nan"), float("nan")

    enrs = cluster_enrs
    return {
        "auprc_mean":       float(np.mean(cluster_auprcs))   if cluster_auprcs else float("nan"),
        "auprc_median":     float(np.median(cluster_auprcs)) if cluster_auprcs else float("nan"),
        "enrichment_gmean": float(np.exp(np.mean(np.log(enrs)))) if enrs else float("nan"),
        "odds_ratio":       float(odds_ratio),
        "fisher_p":         float(fisher_p),
        "n_proteins":       n_proteins,
        "n_clusters":       len(by_cluster),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# =============================================================================
#  FASTA
# =============================================================================

def read_fasta(fasta_path):
    seqs = {}
    acc, buf = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if acc:
                    seqs[acc] = "".join(buf)
                header = line[1:].split()[0]
                acc    = header.split("|")[1] if "|" in header else header
                buf    = []
            else:
                buf.append(line)
    if acc:
        seqs[acc] = "".join(buf)
    log.info(f"  Read {len(seqs):,} sequences from {fasta_path.name}")
    return seqs


# =============================================================================
#  ANNOTATIONS
# =============================================================================

def load_annotations(args):
    log.info("Loading annotations ...")
    ann_df = pd.read_csv(args.annotations, sep="\t", low_memory=False)
    ann_df.columns = ann_df.columns.str.strip().str.lower()
    ann_df = ann_df[ann_df["split"] == args.split].copy()
    ann_df["start"] = ann_df["start"].astype(int)
    ann_df["end"]   = ann_df["end"].astype(int)

    skip = {"Chain", "Initiator methionine", "Peptide", "Alternative sequence",
            "Sequence conflict", "Non-adjacent residues", "Non-terminal residue"}
    ann_df = ann_df[~ann_df["feature_type"].isin(skip)]

    if ("annot_subtype" in ann_df.columns
            and ann_df["annot_subtype"].notna().any()
            and ann_df["annot_subtype"].nunique() > ann_df["feature_type"].nunique()):
        ann_df["feature_type"] = ann_df["annot_subtype"]
        log.info(f"  Using annot_subtype ({ann_df['feature_type'].nunique()} subtypes)")
    else:
        log.info(f"  Using feature_type ({ann_df['feature_type'].nunique()} types)")

    res_counts  = {ft: int((g["end"] - g["start"] + 1).sum())
                   for ft, g in ann_df.groupby("feature_type")}
    prot_counts = {ft: g["accession"].nunique()
                   for ft, g in ann_df.groupby("feature_type")}
    valid = {t for t, c in res_counts.items()
             if c >= args.min_annot_residues
             and prot_counts.get(t, 0) >= args.min_proteins}
    ann_df = ann_df[ann_df["feature_type"].isin(valid)]
    log.info(f"  {len(ann_df):,} rows | {len(valid)} subtypes | "
             f"{ann_df['accession'].nunique():,} proteins")
    return ann_df, valid


# =============================================================================
#  SHARD MERGE
# =============================================================================

def _aggregate_drain(drain_path):
    """Load one drain, aggregate immediately, discard raw records. Returns a
    list of row dicts {feature_id, annot_type, ...agg_metrics}."""
    import pickle
    with open(drain_path, "rb") as f:
        d = pickle.load(f)
    rows = []
    for fid, annot_dict in d["records"].items():
        for annot_type, recs in annot_dict.items():
            agg = aggregate_metrics(recs)
            if agg:
                rows.append({"feature_id": fid, "annot_type": annot_type, **agg})
    return rows


def _merge_shards(args):
    """Aggregate each drain independently, then combine aggregated rows and save.

    Strategy: aggregate_metrics on the full concatenated raw records is the
    most accurate but requires all records in RAM. Instead we aggregate per
    drain (small RAM), collect the resulting rows DataFrames, and combine them
    by re-aggregating the cluster-level statistics across drains.
    This is exact because aggregate_metrics is associative over the record lists
    — concatenating records from two drains gives the same result as aggregating
    each drain separately and then re-merging the per-cluster sub-lists.
    We achieve this by keeping track of per-(fid, annot_type) record lists but
    loading only one drain at a time.
    """
    import pickle

    sentinel_files = sorted(args.outdir.glob("pass2_shard_*.json"))
    if not sentinel_files:
        sentinel_files = sorted(args.outdir.glob("pass2_shard*.json"))
    if not sentinel_files:
        raise FileNotFoundError(
            f"No shard sentinel files (pass2_shard*.json) found in {args.outdir}")

    log.info(f"Merging {len(sentinel_files)} shard(s): {[f.name for f in sentinel_files]}")

    # collect all drain files across all shards
    all_drains       = []
    n_total          = 0
    n_failed         = 0
    eval_feature_ids = None

    for sf in sentinel_files:
        meta         = json.load(open(sf))
        shard_suffix = f"_shard{meta['shard_idx']:02d}of{meta['num_shards']}"
        drains       = sorted(args.outdir.glob(f"pass2_drain{shard_suffix}_*.pkl"))
        if not drains:
            raise FileNotFoundError(f"No drain files for {shard_suffix}")
        all_drains.extend(drains)
        n_total  += meta["n_processed"]
        n_failed += meta["n_failed"]
        if eval_feature_ids is None:
            eval_feature_ids = np.array(meta["eval_feature_ids"])
        log.info(f"  {sf.name}: {meta['n_processed']:,} proteins, {len(drains)} drains")

    log.info(f"  Total: {n_total:,} proteins across {len(all_drains)} drains")
    log.info("  Aggregating one drain at a time (compact cluster-keyed mode) ...")

    # Instead of storing raw records (expensive: ~9 Python floats per protein×feature×annot),
    # pre-group by cluster_id as we load each drain. This stores only one list of AUPRCs
    # per (fid, annot_type, cluster_id) plus pooled tp/fp/fn/tn — far smaller than raw records.
    # compact[(fid, annot_type)][cluster_id] = [
    #   auprc_sum, auprc_n, log_enr_sum, log_enr_n, tp, fp, fn, tn
    # ]
    # Only 8 scalars per cluster — O(features × annots × clusters), constant per drain.
    compact = {}   # (fid, annot_type) → {cluster_id → [8 floats]}

    for i, dp in enumerate(all_drains):
        log.info(f"  [{i+1}/{len(all_drains)}] {dp.name} ...")
        with open(dp, "rb") as f:
            d = pickle.load(f)
        for fid, annot_dict in d["records"].items():
            for annot_type, recs in annot_dict.items():
                key = (fid, annot_type)
                if key not in compact:
                    compact[key] = {}
                cmap = compact[key]
                for r in recs:
                    cid = r.get("cluster_id", "__singleton__")
                    if cid not in cmap:
                        cmap[cid] = [0., 0, 0., 0, 0., 0., 0., 0.]
                    s = cmap[cid]
                    if not np.isnan(r["auprc"]):
                        s[0] += r["auprc"]; s[1] += 1
                    enr = r.get("enrichment", float("nan"))
                    if not np.isnan(enr) and enr > 0:
                        s[2] += np.log(enr); s[3] += 1
                    s[4] += r["tp"]; s[5] += r["fp"]
                    s[6] += r["fn"]; s[7] += r["tn"]
        del d

    log.info(f"  Compact index: {len(compact):,} (feature, annot) pairs — aggregating ...")
    rows = []
    for (fid, annot_type), cmap in compact.items():
        # per-cluster mean auprc and log-enr
        cluster_auprcs = [s[0]/s[1] for s in cmap.values() if s[1] > 0]
        if not cluster_auprcs:
            continue
        n_prot = sum(int(s[1]) for s in cmap.values())
        log_enrs = [s[2]/s[3] for s in cmap.values() if s[3] > 0]
        tp = sum(s[4]/max(s[1],1) for s in cmap.values())
        fp = sum(s[5]/max(s[1],1) for s in cmap.values())
        fn = sum(s[6]/max(s[1],1) for s in cmap.values())
        tn = sum(s[7]/max(s[1],1) for s in cmap.values())
        table = np.array([[tp, fp], [fn, tn]], dtype=np.float64)
        try:
            odds_ratio, fisher_p = sp_stats.fisher_exact(table, alternative="greater")
        except Exception:
            odds_ratio, fisher_p = float("nan"), float("nan")
        rows.append({
            "feature_id":       fid,
            "annot_type":       annot_type,
            "auprc_mean":       float(np.mean(cluster_auprcs)),
            "auprc_median":     float(np.median(cluster_auprcs)),
            "enrichment_gmean": float(np.exp(np.mean(log_enrs))) if log_enrs else float("nan"),
            "odds_ratio":       float(odds_ratio),
            "fisher_p":         float(fisher_p),
            "n_proteins":       n_prot,
            "n_clusters":       len(cmap),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    del compact

    if not rows:
        log.error("No results after aggregation.")
        return

    # reload pass-1 stats for feat_mean / feat_var (needed for novel triage)
    pass1_path = args.outdir / "pass1_stats.npz"
    if not pass1_path.exists():
        raise FileNotFoundError(f"pass1_stats.npz not found in {args.outdir}")
    d1        = np.load(pass1_path)
    n_p       = max(int(d1["n_seen"][0]), 1)
    feat_mean = d1["feat_sum"] / n_p
    feat_var  = np.maximum(d1["feat_sum2"] / n_p - feat_mean ** 2, 0.0)

    log.info(f"  {len(rows):,} (feature, annot) pairs")
    results_df = pd.DataFrame(rows)
    results_df.to_parquet(args.outdir / "alignment_scores.parquet", index=False)

    per_annot = (results_df.sort_values("auprc_mean", ascending=False)
                 .groupby("annot_type").first().reset_index()
                 .sort_values("auprc_mean", ascending=False))
    per_annot.to_csv(args.outdir / "per_annot_summary.tsv", sep="\t", index=False)

    robust = per_annot[
        (per_annot["n_proteins"] >= 100) & (per_annot["fisher_p"] < 0.01) &
        (per_annot["n_clusters"] >= 30)
    ].sort_values("auprc_mean", ascending=False)
    robust.to_csv(args.outdir / "per_annot_summary_robust.tsv", sep="\t", index=False)
    log.info(f"  Robust pairs (n≥100, p<0.01): {len(robust)}")

    # novel triage
    feat_rows = {}
    for r in rows:
        fid  = r["feature_id"]
        auprc = r["auprc_mean"]
        if np.isnan(auprc):
            continue
        if fid not in feat_rows or auprc > feat_rows[fid]["best_auprc"]:
            feat_rows[fid] = {"feature_id": fid, "best_auprc": auprc,
                               "best_annot": r["annot_type"]}

    feat_summary_rows = []
    for fid, best in feat_rows.items():
        all_auprcs = [r["auprc_mean"] for r in rows
                      if r["feature_id"] == fid and not np.isnan(r["auprc_mean"])]
        feat_summary_rows.append({
            **best,
            "mean_auprc_all":    float(np.mean(all_auprcs)),
            "n_annots_above_03": sum(1 for a in all_auprcs if a > 0.30),
            "n_annots_above_05": sum(1 for a in all_auprcs if a > 0.50),
            "feat_mean_act":     float(feat_mean[fid]),
            "feat_variance":     float(feat_var[fid]),
            "mystery_score":     float(1.0 - best["best_auprc"]),
        })

    feat_summary = pd.DataFrame(feat_summary_rows).sort_values("best_auprc")
    feat_summary.to_csv(args.outdir / "feature_triage_summary.tsv", sep="\t", index=False)

    novel = feat_summary[
        feat_summary["best_auprc"] < args.novel_auprc_threshold
    ].sort_values("mystery_score", ascending=False)
    novel.to_csv(args.outdir / "novel_candidates.tsv", sep="\t", index=False)

    log.info(f"  Features evaluated: {len(feat_summary):,}")
    log.info(f"  Novel candidates: {len(novel):,}")

    json.dump({
        "timestamp":          datetime.now().isoformat(),
        "n_proteins":         n_total,
        "n_features_eval":    len(eval_feature_ids),
        "n_annot_types":      results_df["annot_type"].nunique(),
        "n_robust_pairs":     len(robust),
        "n_novel_candidates": len(novel),
        "topk_k":             topk_k,
    }, open(args.outdir / "run_metadata.json", "w"), indent=2)

    log.info(f"\n✅ Merge complete → {args.outdir}")
    log.info(f"   alignment_scores.parquet     — {len(results_df):,} pairs")
    log.info(f"   per_annot_summary_robust.tsv — {len(robust)} pairs")
    log.info(f"   novel_candidates.tsv         — {len(novel)} candidates")


def _load_drains(outdir, shard_suffix):
    """Load and merge all drain pickle files into a single records dict."""
    import pickle
    drains = sorted(outdir.glob(f"pass2_drain{shard_suffix}_*.pkl"))
    if not drains:
        raise FileNotFoundError(
            f"No drain files (pass2_drain{shard_suffix}_*.pkl) in {outdir}")
    records = defaultdict(lambda: defaultdict(list))
    for dp in drains:
        log.info(f"  Loading {dp.name} ...")
        with open(dp, "rb") as f:
            d = pickle.load(f)
        for fid, annot_dict in d["records"].items():
            for annot_type, recs in annot_dict.items():
                records[fid][annot_type].extend(recs)
    log.info(f"  Loaded {len(drains)} drain(s)")
    return records


def _aggregate_and_save(records, eval_feature_ids, feat_mean, feat_var,
                        args, n_processed, topk_k=None):
    """Aggregate per-protein records and write all output files."""
    log.info("\nAggregating and saving results ...")

    rows = []
    for fid, annot_dict in records.items():
        for annot_type, recs in annot_dict.items():
            agg = aggregate_metrics(recs)
            if agg:
                rows.append({"feature_id": fid, "annot_type": annot_type, **agg})

    if not rows:
        log.error("No results — check annotations and sequences.")
        return

    results_df = pd.DataFrame(rows)
    log.info(f"  {len(results_df):,} (feature, annot) pairs")

    results_df.to_parquet(args.outdir / "alignment_scores.parquet", index=False)

    per_annot = (results_df.sort_values("auprc_mean", ascending=False)
                 .groupby("annot_type").first().reset_index()
                 .sort_values("auprc_mean", ascending=False))
    per_annot.to_csv(args.outdir / "per_annot_summary.tsv", sep="\t", index=False)

    robust = per_annot[
        (per_annot["n_proteins"] >= 100) & (per_annot["fisher_p"] < 0.01) &
        (per_annot["n_clusters"] >= 30)
    ].sort_values("auprc_mean", ascending=False)
    robust.to_csv(args.outdir / "per_annot_summary_robust.tsv", sep="\t", index=False)
    log.info(f"  Robust pairs (n≥100, p<0.01): {len(robust)}")

    # ── novel triage ──────────────────────────────────────────────────────────
    log.info("\nNovel concept triage ...")
    feat_rows = []
    for fid, annot_dict in records.items():
        all_auprcs = []
        for annot_type, recs in annot_dict.items():
            agg = aggregate_metrics(recs)
            if agg and not np.isnan(agg.get("auprc_mean", float("nan"))):
                all_auprcs.append((annot_type, agg["auprc_mean"]))
        if not all_auprcs:
            continue
        all_auprcs.sort(key=lambda x: -x[1])
        best_annot, best_auprc = all_auprcs[0]
        feat_rows.append({
            "feature_id":        fid,
            "best_auprc":        best_auprc,
            "best_annot":        best_annot,
            "mean_auprc_all":    float(np.mean([a for _, a in all_auprcs])),
            "n_annots_above_03": sum(1 for _, a in all_auprcs if a > 0.30),
            "n_annots_above_05": sum(1 for _, a in all_auprcs if a > 0.50),
            "feat_mean_act":     float(feat_mean[fid]),
            "feat_variance":     float(feat_var[fid]),
            "mystery_score":     float(1.0 - best_auprc),
        })

    feat_summary = pd.DataFrame(feat_rows).sort_values("best_auprc")
    feat_summary.to_csv(args.outdir / "feature_triage_summary.tsv",
                        sep="\t", index=False)

    novel = feat_summary[
        feat_summary["best_auprc"] < args.novel_auprc_threshold
    ].sort_values("mystery_score", ascending=False)
    novel.to_csv(args.outdir / "novel_candidates.tsv", sep="\t", index=False)

    log.info(f"  Features evaluated: {len(feat_summary):,}")
    log.info(f"  best_auprc distribution:")
    for thr in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]:
        n = (feat_summary["best_auprc"] < thr).sum()
        log.info(f"    < {thr:.2f}: {n:>6,} ({100*n/max(len(feat_summary),1):.1f}%)")
    log.info(f"  Novel candidates: {len(novel):,}")

    if len(novel) > 0:
        log.info("\n  TOP 15 NOVEL CANDIDATES:")
        print(novel.head(15)[["feature_id", "best_auprc", "best_annot",
                               "feat_mean_act", "feat_variance",
                               "n_annots_above_03"]].to_string(index=False))

    json.dump({
        "timestamp":          datetime.now().isoformat(),
        "n_proteins":         n_processed,
        "n_features_eval":    len(eval_feature_ids),
        "n_annot_types":      len(set(results_df["annot_type"])),
        "n_robust_pairs":     len(robust),
        "n_novel_candidates": len(novel),
        "topk_k":             topk_k,
    }, open(args.outdir / "run_metadata.json", "w"), indent=2)

    log.info(f"   alignment_scores.parquet     — {len(results_df):,} pairs")
    log.info(f"   per_annot_summary_robust.tsv — {len(robust)} pairs")
    log.info(f"   novel_candidates.tsv         — {len(novel)} candidates")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    global torch
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.merge_shards:
        _merge_shards(args)
        return

    import torch as _torch
    torch = _torch

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu"))
    log.info(f"Device: {device}")

    # prefetcher buffer: keep 3 full align-batches ready so GPU never idles
    prefetch_buffer = args.align_batch_size * 3

    # ── load models ───────────────────────────────────────────────────────────
    log.info("Loading SAE ...")
    W_enc, b_enc, K, D, topk_k = load_sae_encoder(args.checkpoint, device)
    esm_model, converter, backend = load_esm2(args.esm2_model, device)

    # ── load data ─────────────────────────────────────────────────────────────
    ann_df, valid_types = load_annotations(args)
    annot_types    = sorted(valid_types)
    annotated_accs = set(ann_df["accession"])
    ann_by_acc     = {acc: grp for acc, grp in ann_df.groupby("accession")}

    prot_df = pd.read_csv(args.proteins, sep="\t", low_memory=False)
    prot_df.columns = prot_df.columns.str.strip().str.lower()
    split_accs = set(prot_df[prot_df["split"] == args.split]["accession"])
    log.info(f"  {len(split_accs):,} proteins in split='{args.split}'")

    # cluster map — used to deweight large homologous clusters in metrics
    if "cluster_id" in prot_df.columns:
        acc_to_cluster = dict(zip(prot_df["accession"], prot_df["cluster_id"]))
        n_clusters = prot_df[prot_df["split"] == args.split]["cluster_id"].nunique()
        log.info(f"  {n_clusters:,} clusters in split='{args.split}'")
    else:
        acc_to_cluster = None
        log.warning("  'cluster_id' column not found — metrics will NOT be cluster-weighted")

    seqs = read_fasta(args.fasta)
    target_accs = sorted(split_accs & annotated_accs & set(seqs.keys()))
    log.info(f"  {len(target_accs):,} target proteins")

    if args.fast_triage:
        import random as _random
        _random.seed(42)
        if len(target_accs) > args.triage_max_proteins:
            target_accs = _random.sample(target_accs, args.triage_max_proteins)
            log.info(f"  Fast-triage: subsampled to {len(target_accs):,}")

    if args.num_shards > 1:
        if args.shard_idx >= args.num_shards:
            raise ValueError(f"--shard-idx {args.shard_idx} >= --num-shards {args.num_shards}")
        pass1_path_check = args.outdir / "pass1_stats.npz"
        if not pass1_path_check.exists():
            raise RuntimeError(
                "pass1_stats.npz not found. Run without --num-shards first "
                "to cache pass-1, then re-run with sharding.")
        log.info(f"Sharding: shard {args.shard_idx + 1}/{args.num_shards}")

    # ── feature selection ─────────────────────────────────────────────────────
    if args.all_features:
        log.info(f"\nMode: ALL {K} features")
        K_eval = K
    else:
        log.info(f"\nMode: top-{args.top_k_features} features by {args.select_by}")
        K_eval = args.top_k_features

    # ══════════════════════════════════════════════════════════════════════════
    #  PASS 1 — feature statistics
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\nPass 1: accumulating feature statistics ...")
    pass1_path = args.outdir / "pass1_stats.npz"

    if pass1_path.exists():
        d         = np.load(pass1_path)
        feat_sum  = d["feat_sum"]
        feat_sum2 = d["feat_sum2"]
        feat_max  = d["feat_max"]
        feat_nz   = d["feat_nz"]
        n_seen    = int(d["n_seen"][0])
        log.info(f"  Existing stats ({n_seen:,} proteins) — skipping")
    else:
        feat_sum  = np.zeros(K, dtype=np.float64)
        feat_sum2 = np.zeros(K, dtype=np.float64)
        feat_max  = np.zeros(K, dtype=np.float64)
        feat_nz   = np.zeros(K, dtype=np.float64)
        n_seen    = 0

        pbar1 = (_tqdm(total=len(target_accs), desc="Pass 1",
                       unit="prot", dynamic_ncols=True) if _tqdm else None)
        p1 = DynamicBatchPrefetcher(
            target_accs, seqs, esm_model, converter, backend, device,
            W_enc=W_enc, b_enc=b_enc,
            layer=33, max_len=args.max_seq_len,
            batch_size=args.esm_batch_size,
            bucket_width=args.bucket_width,
            buffer_size=prefetch_buffer,
            topk_k=topk_k,
        )
        for acc, emb, z, err in p1:
            if pbar1: pbar1.update(1)
            if err:   continue
            z64        = z.astype(np.float64)
            feat_sum  += z64.mean(0)
            feat_sum2 += (z64 ** 2).mean(0)
            feat_max   = np.maximum(feat_max, z64.max(0))
            feat_nz   += (z64 > 0).mean(0)
            n_seen    += 1
        if pbar1: pbar1.close()

        np.savez(pass1_path, feat_sum=feat_sum, feat_sum2=feat_sum2,
                 feat_max=feat_max, feat_nz=feat_nz, n_seen=np.array([n_seen]))
        log.info(f"  Done: {n_seen:,} proteins → {pass1_path}")

    n_p       = max(n_seen, 1)
    feat_mean = feat_sum / n_p
    feat_var  = np.maximum(feat_sum2 / n_p - feat_mean ** 2, 0.0)
    feat_spar = feat_max / (feat_nz / n_p + 1e-6)

    crit = {"mean": feat_mean, "variance": feat_var,
            "max":  feat_max,  "sparsity": feat_spar}[args.select_by]
    eval_feature_ids = (np.arange(K) if args.all_features
                        else np.argsort(crit)[::-1][:K_eval])
    log.info(f"  Evaluating {len(eval_feature_ids):,} features")

    # ══════════════════════════════════════════════════════════════════════════
    #  PASS 2 — alignment (drain-checkpointed every 10%)
    #  Each drain saves records to disk and clears RAM, bounding peak memory
    #  to ~1/10 of the total regardless of corpus size.
    # ══════════════════════════════════════════════════════════════════════════
    import pickle

    log.info(f"\nPass 2: alignment "
             f"({len(eval_feature_ids)} features × {len(annot_types)} subtypes)")
    log.info(f"  ESM batch: {args.esm_batch_size} | "
             f"Align batch: {args.align_batch_size} | "
             f"Prefetch buffer: {prefetch_buffer}")

    # sequential slice: shard i covers [i*n_per : (i+1)*n_per]
    if args.num_shards > 1:
        n_per = (len(target_accs) + args.num_shards - 1) // args.num_shards
        start = args.shard_idx * n_per
        end   = min(start + n_per, len(target_accs))
        target_accs_shard = target_accs[start:end]
    else:
        target_accs_shard = target_accs

    shard_suffix   = (f"_shard{args.shard_idx:02d}of{args.num_shards}"
                      if args.num_shards > 1 else "")
    meta_path      = args.outdir / f"pass2_meta{shard_suffix}.json"
    drain_interval = max(1, len(target_accs_shard) // 10)

    records      = defaultdict(lambda: defaultdict(list))
    n_processed  = 0
    n_failed     = 0
    resume_from  = 0
    drain_count  = [0]   # list so the nested closure can mutate it

    if meta_path.exists():
        meta           = json.load(open(meta_path))
        resume_from    = meta["resume_from"]
        n_processed    = meta["n_processed"]
        n_failed       = meta["n_failed"]
        drain_count[0] = meta["drain_count"]
        log.info(f"  Resumed: {n_processed:,} proteins, "
                 f"{drain_count[0]} drain(s) on disk, "
                 f"continuing from {resume_from:,}/{len(target_accs_shard):,}")

    target_accs_p2 = target_accs_shard[resume_from:]

    p2 = DynamicBatchPrefetcher(
        target_accs_p2, seqs, esm_model, converter, backend, device,
        W_enc=W_enc, b_enc=b_enc,
        layer=33, max_len=args.max_seq_len,
        batch_size=args.esm_batch_size,
        bucket_width=args.bucket_width,
        buffer_size=prefetch_buffer,
        topk_k=topk_k,
    )

    pbar2 = (_tqdm(total=len(target_accs_shard), desc="Pass 2", initial=resume_from,
                   unit="prot", dynamic_ncols=True) if _tqdm else None)

    n_iterated     = 0
    protein_buffer = []

    def flush_buffer(buf):
        if not buf:
            return
        br = compute_alignment_multiprotein(
            buf, eval_feature_ids,
            threshold=args.activation_threshold,
            or_min=1.5,
            device=device,
            acc_to_cluster=acc_to_cluster,
        )
        for fid, annot_dict in br.items():
            for annot_type, recs in annot_dict.items():
                records[fid][annot_type].extend(recs)

    def save_drain(global_idx):
        """Flush records to a numbered drain file, clear RAM, update meta."""
        idx  = drain_count[0]
        path = args.outdir / f"pass2_drain{shard_suffix}_{idx:03d}.pkl"
        tmp  = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(
                {"records": {fid: dict(ad) for fid, ad in records.items()}},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp.rename(path)
        json.dump(
            {"resume_from": global_idx, "n_processed": n_processed,
             "n_failed": n_failed, "drain_count": idx + 1},
            open(meta_path, "w"),
        )
        drain_count[0] += 1
        log.info(f"  [drain {idx}] {global_idx:,}/{len(target_accs_shard):,} proteins "
                 f"({n_processed:,} with annotations) → {path.name}  (RAM freed)")
        records.clear()   # ← the whole point: free the CPU RAM
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()   # return cached GPU allocations to OS

    for acc, emb, z, err in p2:
        if pbar2: pbar2.update(1)
        n_iterated += 1

        if err:
            n_failed += 1
            if n_failed <= 5:
                log.warning(f"  Failed {acc}: {err}")
        else:
            ann_acc = ann_by_acc.get(acc)
            if ann_acc is not None:
                L     = emb.shape[0]
                masks = build_masks_for_protein(ann_acc, L)
                if masks:
                    protein_buffer.append({"acc": acc, "z": z, "masks": masks})
                    if len(protein_buffer) >= args.align_batch_size:
                        flush_buffer(protein_buffer)
                        protein_buffer = []
                    n_processed += 1

        if (resume_from + n_iterated) % drain_interval == 0:
            flush_buffer(protein_buffer)
            protein_buffer = []
            save_drain(resume_from + n_iterated)

    flush_buffer(protein_buffer)   # remainder
    if pbar2: pbar2.close()
    if n_failed:
        log.warning(f"  {n_failed} proteins failed")
    log.info(f"  Pass 2 done: {n_processed:,} proteins")

    # save final drain (records since last drain boundary)
    save_drain(resume_from + n_iterated)
    records = defaultdict(lambda: defaultdict(list))   # free before aggregation

    # clean up meta — signals a complete run
    if meta_path.exists():
        meta_path.unlink()

    # ── sharded run: write sentinel and exit ──────────────────────────────────
    if args.num_shards > 1:
        sentinel = args.outdir / f"pass2_shard{shard_suffix}.json"
        json.dump({"n_processed": n_processed, "n_failed": n_failed,
                   "drain_count": drain_count[0],
                   "eval_feature_ids": eval_feature_ids.tolist(),
                   "shard_idx": args.shard_idx, "num_shards": args.num_shards},
                  open(sentinel, "w"))
        log.info(f"\n✅ Shard {args.shard_idx}/{args.num_shards} done "
                 f"({drain_count[0]} drains) → {args.outdir}")
        log.info("   Run with --merge-shards when all shards are complete.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    #  AGGREGATE AND SAVE  (merge drains first, then aggregate)
    # ══════════════════════════════════════════════════════════════════════════
    records = _load_drains(args.outdir, shard_suffix)
    _aggregate_and_save(records, eval_feature_ids, feat_mean, feat_var,
                        args, n_processed, topk_k=topk_k)
    log.info(f"\n✅ Done → {args.outdir}")


if __name__ == "__main__":
    main()