"""
Generate all publication-quality figures for the multi-evidence
PLM interpretability paper.

Figures produced
----------------
Fig 1 — Overview heatmap
    SAE feature x annotation alignment matrix (AUPRC) for top features.
    Rows = annotation types, columns = top SAE features.

Fig 2 — Multi-evidence summary table
    For each top (feature, annotation) pair: bar chart of all four
    evidence layers (alignment AUPRC, physicochemical Cohen's d,
    ablation contrast, attention AUPRC) side by side.

Fig 3 — Convergence: SAE vs Attention
    Scatter plot — x=SAE AUPRC, y=attention AUPRC, coloured by
    annotation type, with convergence_score as bubble size.
    Diagonal = perfect agreement. Off-diagonal = complementary signals.

Fig 4 — Structural features: Helix and Beta strand
    Run the same alignment pipeline on UniProt Helix and Beta strand
    annotations with physicochemical focus on helix/sheet propensity.
    Shows the SAE captures 3D structural propensity from sequence alone.

Fig 5 — Layer-wise attention profile
    For the top annotation types, show AUPRC as a function of ESM-2
    layer (1-33), revealing at which layer biological information
    becomes accessible in the attention heads.

Fig 6a/b/c — Protein case studies (one page per protein)
    Three proteins chosen for maximum interpretability diversity:
      P1: a zinc finger protein (f1310, structural motif)
      P2: a secreted protein (f3087, signal peptide, N-terminal)
      P3: an enzyme with annotated active site (f1696, catalytic)
    Each case study shows:
      - Full sequence coloured by SAE activation
      - Annotation track (UniProt regions)
      - Attention head track (best head)
      - Physicochemical track (most discriminative property)
      - Per-amino-acid table for the annotated region
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

from constants import OUT_DIR, ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH

torch = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
})

PALETTE = {
    "sae":      "#1D9E75",
    "attn":     "#3266AD",
    "physico":  "#BA7517",
    "ablation": "#7F77DD",
    "neutral":  "#888780",
    "annot":    "#E8593C",
}

# physicochemical tables (copied from multi_evidence.py)
HYDROPHOBICITY = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
                  'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
                  'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGE         = {'A':0,'R':1,'N':0,'D':-1,'C':0,'Q':0,'E':-1,'G':0,'H':0,'I':0,
                  'L':0,'K':1,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
HELIX_PROP     = {'A':1.0,'R':0.79,'N':0.23,'D':0.69,'C':0.77,'Q':0.98,'E':1.0,
                  'G':0.0,'H':0.69,'I':1.0,'L':1.0,'K':0.74,'M':1.0,'F':0.92,
                  'P':0.0,'S':0.0,'T':0.64,'W':0.92,'Y':0.72,'V':0.91}
SHEET_PROP     = {'A':0.37,'R':0.90,'N':0.37,'D':0.40,'C':1.0,'Q':0.87,'E':0.26,
                  'G':0.75,'H':0.87,'I':1.0,'L':0.87,'K':0.37,'M':0.75,'F':1.0,
                  'P':0.0,'S':0.75,'T':1.0,'W':1.0,'Y':1.0,'V':1.0}
DISORDER_PROP  = {'A':0.06,'R':0.18,'N':0.21,'D':0.23,'C':0.02,'Q':0.20,'E':0.22,
                  'G':0.13,'H':0.08,'I':0.01,'L':0.02,'K':0.20,'M':0.04,'F':0.02,
                  'P':0.27,'S':0.18,'T':0.14,'W':0.02,'Y':0.05,'V':0.01}
POLARITY       = {'A':0.0,'R':0.65,'N':0.68,'D':0.68,'C':0.28,'Q':0.68,'E':0.68,
                  'G':0.0,'H':0.68,'I':0.13,'L':0.13,'K':0.65,'M':0.28,'F':0.28,
                  'P':0.28,'S':0.62,'T':0.62,'W':0.56,'Y':0.56,'V':0.13}
VOLUME         = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,
                  'E':138.4,'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,
                  'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,
                  'Y':193.6,'V':140.0}
AROMATICITY    = {aa:1 if aa in 'FYWH' else 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}

PROP_TABLES = {
    "hydrophobicity": HYDROPHOBICITY, "charge": CHARGE,
    "helix_prop": HELIX_PROP,         "sheet_prop": SHEET_PROP,
    "disorder_prop": DISORDER_PROP,   "polarity": POLARITY,
    "volume": VOLUME,                 "aromaticity": AROMATICITY,
}

AA_FULL = {
    'A':'Ala','C':'Cys','D':'Asp','E':'Glu','F':'Phe','G':'Gly','H':'His',
    'I':'Ile','K':'Lys','L':'Leu','M':'Met','N':'Asn','P':'Pro','Q':'Gln',
    'R':'Arg','S':'Ser','T':'Thr','V':'Val','W':'Trp','Y':'Tyr',
}


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--alignment-dir",  type=Path,
                   default=OUT_DIR / "feature_alignment")
    p.add_argument("--multi-ev-dir",   type=Path,
                   default=OUT_DIR / "multi_evidence")
    p.add_argument("--attention-dir",  type=Path,
                   default=OUT_DIR / "attention")
    p.add_argument("--esm2-model", default="esm2_t33_650M_UR50D",
                   help="ESM-2 model name for on-the-fly embedding.")
    p.add_argument("--esm2-layer", type=int, default=33,
                   help="ESM-2 transformer layer to extract embeddings from.")
    p.add_argument("--max-seq-len", type=int, default=1022,
                   help="Truncate sequences longer than this.")
    p.add_argument("--checkpoint",     type=Path, required=True)
    p.add_argument("--annotations",    type=Path,
                   default=ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH)
    p.add_argument("--proteins",       type=Path,
                   default=OUT_DIR.parent / "data" / "proteins_with_split.tsv")
    p.add_argument("--split",          default="test")
    p.add_argument("--outdir",         type=Path,
                   default=OUT_DIR / "paper_figures")
    p.add_argument("--device",         default="auto")
    p.add_argument("--esm-model",    default=None,
               help="ESM-2 model name for logit-delta ablation (optional). "
                    "E.g. facebook/esm2_t33_650M_UR50D")
    p.add_argument("--esm-cache-dir", type=Path, default=None)
    p.add_argument("--case-proteins",  type=str, default=None,
                   help="Comma-sep ACC:fid:annot_type:label")
    return p.parse_args()


# =============================================================================
#  SAE HELPERS
# =============================================================================

def load_sae(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model_state_dict"]
    W_enc = sd["encoder.weight"].float().to(device)
    b_enc = sd["encoder.bias"].float().to(device)
    W_dec = sd["decoder.weight"].float().to(device)
    b_dec = sd["decoder.bias"].float().to(device)
    return W_enc, b_enc, W_dec, b_dec


def get_activations(seq_emb, W_enc, b_enc):
    """seq_emb: [L,D] numpy → z: [L,K] numpy float32"""
    with torch.no_grad():
        x = torch.tensor(seq_emb).float().to(W_enc.device)
        z = torch.relu(x @ W_enc.T + b_enc)
    return z.cpu().numpy().astype(np.float32)


def ablate_delta(seq_emb, feature_id, W_enc, b_enc, W_dec, b_dec):
    """Returns per-residue L2 ablation delta [L]"""
    with torch.no_grad():
        x     = torch.tensor(seq_emb).float().to(W_enc.device)
        z     = torch.relu(x @ W_enc.T + b_enc)
        z_abl = z.clone(); z_abl[:, feature_id] = 0.0
        d     = (z @ W_dec.T + b_dec) - (z_abl @ W_dec.T + b_dec)
    return d.norm(dim=1).cpu().numpy().astype(np.float32)


def annot_proj_delta(seq_emb, feature_id, W_dec, annot_mask):
    """
    Returns (emb_proj [L], structural_alignment float).

    emb_proj[i] = (emb[i] - mu_non_annotated) · v_annot
        where v_annot = normalize(mean_annotated_emb - mean_non_annotated_emb).
        Completely SAE-independent — measures annotation-likeness of each
        residue's embedding using the annotation-discriminative direction.

    structural_alignment = cos(W_dec[:,feature_id], v_annot)
        A single scalar: does the feature's decoder direction geometrically
        point toward the annotation subspace?
        > 0.1  → geometrically consistent (decoder pushes toward annotation)
        ≈ 0    → orthogonal (feature may be a positional false positive)
        < -0.1 → conflicting geometry (decoder pushes away from annotation)
    """
    mask = annot_mask.astype(bool)
    if mask.sum() == 0 or (~mask).sum() == 0:
        return np.zeros(seq_emb.shape[0], dtype=np.float32), 0.0

    x = seq_emb.astype(np.float32)
    mu_in  = x[mask].mean(0)    # [D]
    mu_out = x[~mask].mean(0)   # [D]
    v = mu_in - mu_out
    v = v / (np.linalg.norm(v) + 1e-8)

    # per-residue embedding projection (SAE-independent)
    emb_proj = ((x - mu_out) @ v).astype(np.float32)   # [L]

    # geometric alignment of the feature's decoder direction
    # W_dec shape is [D, K]; column k = decoder direction for feature k
    w_feat = W_dec[:, feature_id].cpu().float().numpy()  # [D]
    structural_alignment = float(
        np.dot(w_feat / (np.linalg.norm(w_feat) + 1e-8), v)
    )
    return emb_proj, structural_alignment


def load_esm_for_logits(model_name, device, cache_dir=None):
    """Load ESM-2 LM head. Stored embeddings are last hidden states → apply directly."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    log.info(f"  Loading ESM-2 LM head from {model_name} ...")
    kw = {"cache_dir": str(cache_dir)} if cache_dir else {}
    model = AutoModelForMaskedLM.from_pretrained(model_name, **kw).float().to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
    return model.lm_head, tokenizer


def logit_delta(seq_emb, seq, feature_id, W_enc, b_enc, W_dec, b_dec,
                lm_head, tokenizer):
    """
    Per-residue change in log-probability of the correct amino acid when
    feature_id's contribution is removed from the embedding.
    Downstream causal test — independent of SAE reconstruction.
    delta[i] > 0 → feature was helping prediction at residue i.
    Returns delta_lp [L] float32.
    """
    import torch.nn.functional as F_func
    with torch.no_grad():
        x = torch.tensor(seq_emb).float().to(W_enc.device)
        z = torch.relu(x @ W_enc.T + b_enc)                     # [L, K]
        feat_dir    = W_dec[:, feature_id]                       # [D]
        feat_acts   = z[:, feature_id:feature_id+1]              # [L, 1]
        feat_contrib = feat_acts * feat_dir                       # [L, D]
        x_cf = x - feat_contrib                                  # [L, D]

        logits_orig = lm_head(x.unsqueeze(0)).squeeze(0)         # [L, vocab]
        logits_cf   = lm_head(x_cf.unsqueeze(0)).squeeze(0)

        enc_out   = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
        token_ids = enc_out["input_ids"][0, :len(seq_emb)].to(W_enc.device)

        lp_orig = F_func.log_softmax(logits_orig, dim=-1)
        lp_cf   = F_func.log_softmax(logits_cf,   dim=-1)
        idx      = torch.arange(len(token_ids), device=W_enc.device)
        delta_lp = lp_orig[idx, token_ids] - lp_cf[idx, token_ids]

    return delta_lp.cpu().numpy().astype(np.float32)


# =============================================================================
#  DATA LOADING
# =============================================================================

from esm_utils import load_esm2
from esm_utils import get_embedding as _get_embedding_live


def build_mask(ann_sub, L):
    mask = np.zeros(L, dtype=np.float32)
    for _, row in ann_sub.iterrows():
        s = max(0, int(row["start"]) - 1)
        e = min(L, int(row["end"]))
        mask[s:e] = 1.0
    return mask


def seq_props(seq):
    """Return dict of per-residue property arrays."""
    return {name: np.array([tbl.get(aa, 0.0) for aa in seq], dtype=np.float32)
            for name, tbl in PROP_TABLES.items()}


# =============================================================================
#  ALIGNMENT (for structural features)
# =============================================================================

def run_alignment_for_type(
    annot_type, feature_ids, accs, ann_by_acc, get_emb,
    W_enc, b_enc, threshold=0.1, n_perms=50,
):
    """Compute AUPRC for each feature_id × annot_type pair."""
    records = defaultdict(list)
    for acc in accs:
        if acc not in ann_by_acc: continue
        ann_sub = ann_by_acc[acc][ann_by_acc[acc]["feature_type"] == annot_type]
        if ann_sub.empty: continue
        emb  = get_emb(acc)
        L    = emb.shape[0]
        mask = build_mask(ann_sub, L)
        if mask.sum() == 0: continue
        z    = get_activations(emb, W_enc, b_enc)
        for fid in feature_ids:
            scores = z[:, fid]
            binary = (scores > threshold).astype(np.float32)
            if binary.sum() == 0: continue
            try:
                auprc = float(average_precision_score(mask, scores))
            except Exception:
                continue
            tp = float((binary * mask).sum())
            fp = float((binary * (1-mask)).sum())
            fn = float(((1-binary) * mask).sum())
            tn = float(((1-binary) * (1-mask)).sum())
            records[fid].append({"auprc": auprc, "tp": tp, "fp": fp,
                                  "fn": fn, "tn": tn})
    results = []
    for fid, recs in records.items():
        auprcs = [r["auprc"] for r in recs]
        results.append({"feature_id": fid, "annot_type": annot_type,
                        "auprc_mean": np.mean(auprcs),
                        "n_proteins": len(recs)})
    return pd.DataFrame(results).sort_values("auprc_mean", ascending=False)


# =============================================================================
#  FIGURE 1 — OVERVIEW HEATMAP
# =============================================================================

def fig1_heatmap(align_df, outdir):
    """
    AUPRC heatmap: rows=annotation types, cols=top features.
 
    Improvements over v1:
    - Filters trivially positional annotations (precision < 0.05)
    - Separates site features from domain features via annotation name
    - Clusters rows by annotation category for visual organisation
    - Shows feature ids only for features with max AUPRC > 0.5
    - Adds black border on cells with AUPRC > 0.9 (highlight top pairs)
    - Uses diverging colormap anchored at baseline (random AUPRC ~ 0.05)
    """
    log.info("  Fig 1: heatmap ...")
 
    # filter trivially positional (precision < 0.05)
    df = align_df.copy()
    if "tp" in df.columns and "fp" in df.columns:
        df["precision"] = df["tp"] / (df["tp"] + df["fp"] + 1e-9)
        df = df[df["precision"] >= 0.05]

    # ── remove redundant annotation types ─────────────────────────────────
    # 1. For annotation groups sharing a prefix (e.g. "Active site: X"),
    #    keep only the single best subtype to avoid visual redundancy.
    type_max = df.groupby("annot_type")["auprc_mean"].max()
    type_n_high = df[df["auprc_mean"] > 0.95].groupby("annot_type")["feature_id"].nunique()

    # keep only annotations with max AUPRC > 0.5 but not trivially captured by many features
    type_candidates = type_max[type_max > 0.5].index
    type_candidates = [t for t in type_candidates 
                    if type_n_high.get(t, 0) <= 5]  # max 5 features with AUPRC>0.95

    # 2. Deduplicate by prefix: for annotations sharing the same category
    #    prefix (before ": "), keep only the one with the highest AUPRC.
    _prefix_best = {}
    for t in type_candidates:
        prefix = t.split(":")[0].strip()
        score = float(type_max.get(t, 0))
        if prefix not in _prefix_best or score > _prefix_best[prefix][1]:
            _prefix_best[prefix] = (t, score)
    type_candidates_dedup = [v[0] for v in _prefix_best.values()]

    # 3. Also drop annotations whose best feature already is the best
    #    feature for another annotation with higher AUPRC (feature overlap).
    _feat_owner = {}
    _candidate_scored = sorted(type_candidates_dedup,
                               key=lambda t: float(type_max.get(t, 0)),
                               reverse=True)
    type_candidates_unique = []
    for t in _candidate_scored:
        best_fid = int(df[df["annot_type"] == t]
                       .sort_values("auprc_mean", ascending=False)
                       .iloc[0]["feature_id"])
        if best_fid not in _feat_owner:
            _feat_owner[best_fid] = t
            type_candidates_unique.append(t)
        else:
            # allow if this annotation has a different 2nd-best feature
            sub = df[df["annot_type"] == t].sort_values("auprc_mean", ascending=False)
            if len(sub) > 1:
                second_fid = int(sub.iloc[1]["feature_id"])
                if second_fid not in _feat_owner:
                    _feat_owner[second_fid] = t
                    type_candidates_unique.append(t)

    top_types = (df[df["annot_type"].isin(type_candidates_unique)]
                .groupby("annot_type")["auprc_mean"]
                .max().nlargest(16).index.tolist())

    # select top-25 features that are most specific
    # prefer features that are top-1 for at least one annotation
    top1_feats = df.loc[df.groupby("annot_type")["auprc_mean"].idxmax(), "feature_id"].unique()
    other_feats = (df.groupby("feature_id")["auprc_mean"]
                .max().nlargest(50).index.tolist())
    top_feats = list(dict.fromkeys(list(top1_feats) + other_feats))[:25]
 
    # categorise annotations for row ordering
    def _cat(at):
        at_lower = at.lower()
        if any(k in at_lower for k in ["binding site", "active site", "zinc finger",
                                        "region: g", "lipidation", "modified"]):
            return "0_focal"
        if any(k in at_lower for k in ["signal", "transit", "propeptide"]):
            return "1_targeting"
        if any(k in at_lower for k in ["domain", "dna binding"]):
            return "2_domain"
        return "3_other"
 
    type_order = sorted(top_types, key=lambda t: (_cat(t), t))
 
    sub = df[df["feature_id"].isin(top_feats) &
             df["annot_type"].isin(top_types)]
    pivot = sub.pivot_table(index="annot_type", columns="feature_id",
                            values="auprc_mean", fill_value=0.0)
    pivot = pivot.reindex(index=[t for t in type_order if t in pivot.index])
 
    # reorder columns: sort by argmax row, then by max value
    col_argmax  = pivot.values.argmax(axis=0)
    col_maxval  = pivot.max(axis=0).values
    col_order   = pivot.columns[
        np.lexsort((col_maxval[::-1], col_argmax))]
    pivot = pivot[col_order]
 
    n_rows, n_cols = pivot.shape
    fig_w = max(12, n_cols * 0.55)
    fig_h = max(5,  n_rows * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
 
    cmap = LinearSegmentedColormap.from_list(
        "sae", ["#FFFBF5", "#FAC775", "#D85A30", "#7F77DD", "#26215C"])
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                   vmin=0.0, vmax=1.0, interpolation="nearest")
 
    # highlight cells with AUPRC > 0.9
    for r in range(n_rows):
        for c in range(n_cols):
            v = pivot.values[r, c]
            if v >= 0.90:
                rect = plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    linewidth=1.2, edgecolor="#26215C",
                    facecolor="none", zorder=4)
                ax.add_patch(rect)
            if v >= 0.85:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if v > 0.7 else "#3d3d3a",
                        fontweight="bold")
 
    # x-axis: show label only for features with max AUPRC > 0.5
    xlabels = []
    for fid in pivot.columns:
        mx = pivot[fid].max()
        xlabels.append(f"f{fid}" if mx >= 0.5 else "")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=8.5)
 
    # y-axis: add category separator lines
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=10.5)
 
    cats = [_cat(t) for t in pivot.index]
    for i in range(1, n_rows):
        if cats[i] != cats[i-1]:
            ax.axhline(i - 0.5, color="white", lw=2.0, zorder=5)
 
    cb = plt.colorbar(im, ax=ax, label="AUPRC (cluster-averaged)",
                      fraction=0.012, pad=0.01)
    cb.ax.tick_params(labelsize=10)
    ax.set_title("SAE feature × UniProt annotation alignment",
                 fontsize=13, pad=10)
    ax.set_xlabel("SAE feature", fontsize=11)
 
    fig.tight_layout()
    path = outdir / "fig1_alignment_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"    → {path.name}")


# =============================================================================
#  FIGURE 2 — MULTI-EVIDENCE PANEL
# =============================================================================

def fig2_multi_evidence(ev_matrix, attn_conv, outdir,
                        ann_df=None, seq_by_acc=None, get_emb=None,
                        get_attn_scores=None):
    """
    Grouped bar chart: for each top annotation type, show all evidence
    layers side by side (normalised to [0,1] for visual comparison).
    
    If attention AUPRC is not available from convergence data, computes it
    on-the-fly using get_attn_scores when provided.
    """
    log.info("  Fig 2: multi-evidence panel ...")

    rows = []
    for _, er in ev_matrix.iterrows():
        at = er["annot_type"]
        
        # Try convergence data first
        attn_auprc = 0.0
        attn_row = attn_conv[attn_conv["annot_type"] == at] if not attn_conv.empty else pd.DataFrame()
        if not attn_row.empty:
            attn_auprc = float(attn_row["attn_auprc"].values[0])
        elif get_attn_scores is not None and ann_df is not None and seq_by_acc is not None:
            # Compute attention AUPRC on-the-fly for this annotation type
            auprc_vals = []
            ann_sub_all = ann_df[ann_df["feature_type"] == at]
            for acc in ann_sub_all["accession"].unique()[:50]:
                if acc not in seq_by_acc:
                    continue
                ann_acc = ann_sub_all[ann_sub_all["accession"] == acc]
                try:
                    attn_sc = get_attn_scores(acc)  # [L]
                except Exception:
                    continue
                L = len(attn_sc)
                mask = build_mask(ann_acc, L)
                if mask.sum() == 0 or mask.sum() == L:
                    continue
                try:
                    auprc_vals.append(float(average_precision_score(mask, attn_sc)))
                except Exception:
                    continue
            if auprc_vals:
                attn_auprc = float(np.mean(auprc_vals))
                log.info(f"    Attention AUPRC for {at}: {attn_auprc:.3f} (from {len(auprc_vals)} proteins)")

        rows.append({
            "annot_type":   at,
            "SAE align":    float(er["auprc_mean"]),
            "Physico |d|":  min(abs(float(er.get("pc_top_cohen_d", 0))), 2) / 2,
            "Struct. align.":  (float(er.get("abl_structural_align", 0)) + 1) / 2,
            "Attention":    attn_auprc,
            "Logit Δ (frac)":  float(er.get("logit_frac_positive", 0)),
        })

    df = pd.DataFrame(rows).sort_values("SAE align", ascending=False).head(8)
    all_metrics = ["SAE align", "Physico |d|", "Struct. align.", "Attention", "Logit Δ (frac)"]
    all_colors  = [PALETTE["sae"], PALETTE["physico"],
                   PALETTE["ablation"], PALETTE["attn"], "#BA7517"]
    # drop metrics that are all zero/nan — avoids invisible ghost bars
    metrics, colors = [], []
    for m, c in zip(all_metrics, all_colors):
        vals = df[m].values
        if np.nanmax(np.abs(vals)) > 1e-6:
            metrics.append(m)
            colors.append(c)

    n_groups = len(df)
    n_bars   = len(metrics)
    x = np.arange(n_groups)
    w = 0.15

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - n_bars/2 + 0.5) * w
        vals = df[metric].values
        bars = ax.bar(x + offset, vals, w, label=metric,
                      color=color, alpha=0.88, edgecolor="none")

    # Shorten long annotation labels
    short_labels = []
    for at in df["annot_type"]:
        lab = at
        lab = lab.replace("Transmembrane: Helical; Signal-anchor for type II membrane protein",
                          "TM: Signal-anchor (type II)")
        lab = lab.replace("Binding site: ", "BS: ")
        lab = lab.replace("Active site: ", "AS: ")
        lab = lab.replace("Domain: ", "Dom: ")
        lab = lab.replace("DNA binding: ", "DNA: ")
        short_labels.append(lab)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=10.5)
    ax.set_ylabel("Score (normalised)", fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.axhline(0, color="#cccccc", lw=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=10, ncol=2)
    ax.set_title("Multi-evidence alignment scores per annotation type",
                 fontsize=13, pad=10)

    # Add value labels on top of tallest bar per group
    for i, (_, row) in enumerate(df.iterrows()):
        best = max(row[metrics])
        ax.text(i, best + 0.02, f"{best:.2f}",
                ha="center", va="bottom", fontsize=9,
                color=PALETTE["neutral"])

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    path = outdir / "fig2_multi_evidence.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"    → {path.name}")


# =============================================================================
#  FIGURE 3 — SAE vs ATTENTION CONVERGENCE SCATTER
# =============================================================================

def fig3_convergence(attn_conv, outdir,
                     ev_matrix=None, ann_df=None, seq_by_acc=None,
                     get_attn_scores=None, sae_summary_df=None):
    """
    SAE × Attention convergence scatter.

    Plots points from convergence_with_sae.tsv, plus (optionally)
    multi-evidence annotation types with on-the-fly attention AUPRC.
    Filters out trivial types (n_clusters < 3) when sae_summary_df is given.
    Scales gracefully from a handful to hundreds of annotation types.
    """
    log.info("  Fig 3: SAE vs attention convergence ...")

    cat_colors = {
        "focal":            PALETTE["sae"],
        "signal/targeting": PALETTE["attn"],
        "structural":       PALETTE["physico"],
        "domain":           PALETTE["ablation"],
        "modification":     "#BA7517",
    }

    def _cat(at):
        at_l = at.lower()
        if any(k in at_l for k in ["binding site", "active site", "zinc finger",
                                    "region: g", "lipidation", "site:"]):
            return "focal"
        if any(k in at_l for k in ["signal", "transit", "propeptide"]):
            return "signal/targeting"
        if any(k in at_l for k in ["helix", "beta strand", "transmembrane",
                                    "coiled", "turn", "compositional"]):
            return "structural"
        if any(k in at_l for k in ["domain", "dna binding"]):
            return "domain"
        return "modification"

    # ── build unified data: convergence + multi-evidence on-the-fly ───────
    all_points = []
    existing_types = set()

    for _, row in attn_conv.iterrows():
        at = row["annot_type"]
        existing_types.add(at)
        all_points.append({
            "annot_type": at,
            "sae_auprc":  float(row["sae_auprc"]),
            "attn_auprc": float(row["attn_auprc"]),
            "convergence_score": float(row["convergence_score"]),
            "source": "attention_run",
        })

    # add multi-evidence types not already present
    if (ev_matrix is not None and get_attn_scores is not None
            and ann_df is not None and seq_by_acc is not None):
        for _, er in ev_matrix.iterrows():
            at = er["annot_type"]
            if at in existing_types:
                continue
            sae_auprc = float(er["auprc_mean"])

            # compute attention AUPRC on-the-fly
            ann_sub = ann_df[ann_df["feature_type"] == at]
            auprc_vals = []
            for acc in ann_sub["accession"].unique()[:50]:
                if acc not in seq_by_acc:
                    continue
                ann_acc = ann_sub[ann_sub["accession"] == acc]
                try:
                    attn_sc = get_attn_scores(acc)
                except Exception:
                    continue
                if attn_sc is None:
                    continue
                L = len(attn_sc)
                mask = build_mask(ann_acc, L)
                if mask.sum() == 0 or mask.sum() == L:
                    continue
                try:
                    auprc_vals.append(float(average_precision_score(mask, attn_sc)))
                except Exception:
                    continue

            if auprc_vals:
                attn_auprc = float(np.mean(auprc_vals))
                hmean = 2 * sae_auprc * attn_auprc / (sae_auprc + attn_auprc + 1e-9)
                log.info(f"    Fig3 on-the-fly: {at}  attn={attn_auprc:.3f}  sae={sae_auprc:.3f} ({len(auprc_vals)} proteins)")
                existing_types.add(at)
                all_points.append({
                    "annot_type": at,
                    "sae_auprc":  sae_auprc,
                    "attn_auprc": attn_auprc,
                    "convergence_score": hmean,
                    "source": "multi_evidence",
                })

    # Filter out trivial types (n_clusters < 3) to avoid AUPRC=1 artifacts
    if sae_summary_df is not None and "n_clusters" in sae_summary_df.columns:
        robust_types = set(sae_summary_df[sae_summary_df["n_clusters"] >= 3]["annot_type"])
        before = len(all_points)
        all_points = [p for p in all_points if p["annot_type"] in robust_types]
        log.info(f"    Filtered {before} → {len(all_points)} points (n_clusters >= 3)")

    n_pts = len(all_points)
    many = n_pts > 30   # adapt visual density

    fig, ax = plt.subplots(figsize=(7, 6.5))

    # quadrant shading
    thresh = 0.35
    ax.axvspan(thresh, 1.02, ymin=0, ymax=thresh/1.02,
               alpha=0.04, color=PALETTE["sae"], zorder=0)    # SAE dominant
    ax.axhspan(thresh, 1.02, xmin=0, xmax=thresh/1.02,
               alpha=0.04, color=PALETTE["attn"], zorder=0)   # Attn dominant
    ax.axvspan(thresh, 1.02, ymin=thresh/1.02, ymax=1.0,
               alpha=0.04, color="#888780", zorder=0)          # convergent

    # diagonal
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=0.8,
            linestyle="--", zorder=1)

    # ── plot points ───────────────────────────────────────────────────────
    pt_size_scale = 200 if many else 500
    pt_alpha = 0.55 if many else 0.78
    pt_base  = 15 if many else 30

    # batch scatter by category for cleaner legend
    for cat_name, cat_color in cat_colors.items():
        cat_pts = [p for p in all_points if _cat(p["annot_type"]) == cat_name]
        if not cat_pts:
            continue
        xs = [p["sae_auprc"] for p in cat_pts]
        ys = [p["attn_auprc"] for p in cat_pts]
        ss = [max(p["convergence_score"] * pt_size_scale + pt_base, pt_base)
              for p in cat_pts]
        ax.scatter(xs, ys, s=ss, color=cat_color, alpha=pt_alpha,
                   marker="o", edgecolors="white", linewidths=0.4,
                   zorder=3, label=cat_name)

    # ── legend ────────────────────────────────────────────────────────────
    # category patches
    used_cats = set(_cat(p["annot_type"]) for p in all_points)
    legend_elements = [mpatches.Patch(color=c, label=k, alpha=0.85)
                       for k, c in cat_colors.items() if k in used_cats]
    # size legend — fixed display sizes
    for conv_val, disp_s in [(0.3, 25), (0.6, 55), (0.9, 95)]:
        legend_elements.append(
            plt.scatter([], [], s=disp_s,
                        color="#888780", alpha=0.5, edgecolors="white",
                        label=f"conv={conv_val}"))
    # summary stats
    sae_wins = sum(1 for p in all_points if p["sae_auprc"] > p["attn_auprc"])
    attn_wins = n_pts - sae_wins
    # Pearson R²
    from scipy.stats import pearsonr
    _xs_all = [p["sae_auprc"] for p in all_points]
    _ys_all = [p["attn_auprc"] for p in all_points]
    _r, _r_p = pearsonr(_xs_all, _ys_all)
    _r2 = _r ** 2
    legend_elements.append(mpatches.Patch(color="none",
                           label=f"n={n_pts}  SAE>{sae_wins}  Attn>{attn_wins}  R²={_r2:.2f}"))

    ax.legend(handles=legend_elements, loc="upper left",
              frameon=False, fontsize=9, ncol=1,
              handletextpad=0.6, labelspacing=0.8)

    ax.set_xlabel("SAE feature AUPRC", fontsize=11)
    ax.set_ylabel("Attention aggregation AUPRC", fontsize=11)
    ax.set_title(f"SAE × Attention convergence  ({n_pts} annotation subtypes)\n"
                 f"(bubble size = convergence score; Pearson R² = {_r2:.2f})", fontsize=12)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(labelsize=10)

    # quadrant labels — high zorder + white bbox so they sit above points
    _qlabel_bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9)
    ax.text(0.97, 0.02, "SAE dominant", ha="right", va="bottom",
            fontsize=10, color=PALETTE["sae"], alpha=0.8, fontweight="bold",
            transform=ax.transAxes, zorder=10, bbox=_qlabel_bbox)
    ax.text(0.02, 0.60, "Attn dominant", ha="left", va="top",
            fontsize=10, color=PALETTE["attn"], alpha=0.8, fontweight="bold",
            transform=ax.transAxes, zorder=10, bbox=_qlabel_bbox)
    ax.text(0.97, 0.97, "Convergent", ha="right", va="top",
            fontsize=10, color="#888780", alpha=0.8, fontweight="bold",
            transform=ax.transAxes, zorder=10, bbox=_qlabel_bbox)

    fig.tight_layout()
    path = outdir / "fig3_convergence_scatter.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"    → {path.name}  ({n_pts} points)")


# =============================================================================
#  FIGURE 4 — STRUCTURAL FEATURES (Helix / Beta strand)
# =============================================================================

def fig4_structural(ann_df, get_emb, W_enc, b_enc,
                    align_df, outdir, n_proteins=100):
    """
    For Helix and Beta strand:
    1. Run alignment for top-50 features → show AUPRC distribution.
    2. For the best feature, show physicochemical profile
       (helix_prop/sheet_prop inside vs outside).
    3. Show that the best feature's top property matches the
       known Chou-Fasman propensity for that structure.
    """
    log.info("  Fig 4: structural features (Helix, Beta strand) ...")

    struct_types = ["Helix", "Beta strand"]

    # pick top-100 features by mean activation (already computed in align_df)
    top_features = (align_df.groupby("feature_id")["auprc_mean"]
                    .max().nlargest(100).index.tolist())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for row_i, stype in enumerate(struct_types):
        ann_sub = ann_df[ann_df["feature_type"] == stype]
        accs = ann_sub["accession"].unique()[:n_proteins]

        # run alignment
        aln = run_alignment_for_type(
            stype, top_features[:50], accs,
            {acc: ann_df[ann_df["accession"]==acc] for acc in accs},
            get_emb, W_enc, b_enc,
        )

        # ── left: AUPRC distribution across features ─────────────────────────
        ax = axes[row_i, 0]
        if not aln.empty:
            ax.hist(aln["auprc_mean"], bins=20,
                    color=PALETTE["sae"], alpha=0.8, edgecolor="none")
            best_auprc = aln.iloc[0]["auprc_mean"]
            ax.axvline(best_auprc, color=PALETTE["annot"],
                       lw=1.5, linestyle="--",
                       label=f"best: f{int(aln.iloc[0]['feature_id'])} "
                             f"({best_auprc:.3f})")
            ax.legend(frameon=False, fontsize=10)
        ax.set_xlabel("AUPRC", fontsize=11)
        ax.set_ylabel("# features", fontsize=11)
        ax.set_title(f"{stype} — AUPRC distribution\n"
                     f"across top-50 SAE features", fontsize=11)

        # ── middle: physicochemical inside vs outside ─────────────────────────
        ax2 = axes[row_i, 1]
        if not aln.empty:
            best_fid = int(aln.iloc[0]["feature_id"])
            prop_name = "helix_prop" if stype == "Helix" else "sheet_prop"
            prop_table = PROP_TABLES[prop_name]

            vals_in, vals_out = [], []
            for acc in accs[:50]:
                ann_acc = ann_df[(ann_df["accession"]==acc) &
                                 (ann_df["feature_type"]==stype)]
                if ann_acc.empty: continue
                emb  = get_emb(acc)
                L    = emb.shape[0]
                mask = build_mask(ann_acc, L)
                # We don't have sequence here — use helix/sheet prop
                # proxy from the embedding: project onto W_enc[best_fid]
                z    = get_activations(emb, W_enc, b_enc)
                acts = z[:, best_fid]
                vals_in.extend(acts[mask == 1].tolist())
                vals_out.extend(acts[mask == 0].tolist())

            if vals_in and vals_out:
                bins = np.linspace(0, max(max(vals_in), max(vals_out)) * 1.05, 40)
                ax2.hist(vals_out, bins=bins, density=True,
                         color=PALETTE["neutral"], alpha=0.5, label="outside")
                ax2.hist(vals_in,  bins=bins, density=True,
                         color=PALETTE["sae"],     alpha=0.7, label="inside")
                ax2.legend(frameon=False, fontsize=10)
        ax2.set_xlabel(f"f{best_fid if not aln.empty else '?'} activation", fontsize=11)
        ax2.set_ylabel("Density", fontsize=11)
        ax2.set_title(f"{stype} — activation inside vs outside\n"
                      f"(best feature)", fontsize=11)

        # ── right: Chou-Fasman propensity for best 20 AAs ───────────────────
        ax3 = axes[row_i, 2]
        prop_name = "helix_prop" if stype == "Helix" else "sheet_prop"
        prop_table = PROP_TABLES[prop_name]
        aa_sorted = sorted(prop_table.keys(),
                           key=lambda aa: prop_table[aa], reverse=True)
        vals = [prop_table[aa] for aa in aa_sorted]
        colors_bar = [PALETTE["sae"] if v > 0.7 else
                      PALETTE["physico"] if v > 0.4 else
                      PALETTE["neutral"] for v in vals]
        ax3.bar(aa_sorted, vals, color=colors_bar, edgecolor="none", alpha=0.85)
        ax3.axhline(0.7, color=PALETTE["annot"], lw=0.8, linestyle="--",
                    label="high propensity (>0.7)")
        ax3.set_xlabel("Amino acid", fontsize=11)
        ax3.set_ylabel("Chou-Fasman propensity (norm.)", fontsize=11)
        ax3.set_title(f"{stype} propensity scale\n"
                      f"(ground truth validation)", fontsize=11)
        ax3.legend(frameon=False, fontsize=9)

    fig.suptitle("Structural feature alignment: Helix and Beta strand",
                 fontsize=14, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = outdir / "fig4_structural_features.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"    → {path.name}")


# =============================================================================
#  FIGURE 5 — LAYER-WISE ATTENTION PROFILE
# =============================================================================

def fig5_layer_attention(attn_scores_df, outdir, sae_summary_df=None):
    """
    For each top annotation type, plot AUPRC as a function of ESM-2 layer.
    Shows at which layer biological information becomes accessible.
    Selects annotation types with meaningful variation across layers.
    Filters out trivial types (n_proteins < 5) if SAE summary is available.
    """
    log.info("  Fig 5: layer-wise attention profile ...")

    # filter to layer_XX_to columns only
    layer_rows = attn_scores_df[
        attn_scores_df["attn_agg"].str.match(r"layer_\d+_to")
    ].copy()
    layer_rows["layer"] = (layer_rows["attn_agg"]
                           .str.extract(r"layer_(\d+)_to")
                           .astype(int) + 1)

    # Filter out trivial types with very few proteins/clusters
    if sae_summary_df is not None and "n_clusters" in sae_summary_df.columns:
        robust_types = set(sae_summary_df[sae_summary_df["n_clusters"] >= 3]["annot_type"])
        layer_rows = layer_rows[layer_rows["annot_type"].isin(robust_types)].copy()
        log.info(f"    Filtered to {len(robust_types)} robust types (n_clusters >= 3)")
    elif sae_summary_df is not None and "n_proteins" in sae_summary_df.columns:
        robust_types = set(sae_summary_df[sae_summary_df["n_proteins"] >= 5]["annot_type"])
        layer_rows = layer_rows[layer_rows["annot_type"].isin(robust_types)].copy()

    # Select annotation types with meaningful variation across layers:
    # require range(max-min) > 0.1 AND max AUPRC > 0.15
    type_stats = (layer_rows.groupby("annot_type")["auprc_mean"]
                  .agg(["max", "min", "std"]))
    type_stats["range"] = type_stats["max"] - type_stats["min"]
    good_types = type_stats[(type_stats["range"] > 0.08) &
                            (type_stats["max"] > 0.1)].sort_values("range", ascending=False)
    top_types = good_types.head(8).index.tolist()

    if not top_types:
        # fallback: just use top by max AUPRC
        top_types = (attn_scores_df.groupby("annot_type")["auprc_mean"]
                     .max().nlargest(6).index.tolist())

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Identify 3 representative subtypes: early, mid, late peaking ---
    peak_layers = {}
    for at in top_types:
        sub = layer_rows[layer_rows["annot_type"] == at].sort_values("layer")
        if not sub.empty:
            peak_layers[at] = sub.loc[sub["auprc_mean"].idxmax(), "layer"]

    if peak_layers:
        sorted_by_peak = sorted(peak_layers.items(), key=lambda x: x[1])
        # Pick 3 maximally spread: earliest, one near middle, latest
        # but skip types whose peak is within 3 layers of already-selected
        highlight_types = [sorted_by_peak[0][0]]  # earliest
        for at, pk in sorted_by_peak[1:]:
            if all(abs(pk - peak_layers[h]) >= 3 for h in highlight_types):
                highlight_types.append(at)
            if len(highlight_types) == 3:
                break
        # Fallback if not enough spread
        if len(highlight_types) < 3:
            highlight_types = [sorted_by_peak[0][0],
                               sorted_by_peak[len(sorted_by_peak)//2][0],
                               sorted_by_peak[-1][0]]
            seen = set()
            highlight_types = [t for t in highlight_types if not (t in seen or seen.add(t))]
    else:
        highlight_types = top_types[:3]

    highlight_colors = ["#2166AC", "#D6604D", "#1B7837"]  # blue, red, green

    # --- Plot background types in grey first ---
    for at in top_types:
        if at in highlight_types:
            continue
        sub = (layer_rows[layer_rows["annot_type"] == at]
               .sort_values("layer"))
        if sub.empty:
            continue
        ax.plot(sub["layer"], sub["auprc_mean"],
                marker="", lw=1.0, color="#AAAAAA", alpha=0.35, zorder=1)

    # --- Plot highlighted types with bold lines ---
    for i, at in enumerate(highlight_types):
        sub = (layer_rows[layer_rows["annot_type"] == at]
               .sort_values("layer"))
        if sub.empty:
            continue
        color = highlight_colors[i % len(highlight_colors)]

        # Shorten label
        short = (at.replace("Transmembrane: ", "TM: ")
                   .replace("Binding site: ", "BS: ")
                   .replace("Topological domain: ", "TD: ")
                   .replace("Propeptide: ", "PP: ")
                   .replace("Transit peptide: ", "TP: "))

        ax.plot(sub["layer"], sub["auprc_mean"],
                marker="o", markersize=4, lw=2.5,
                color=color, label=short, alpha=1.0, zorder=4)
        # mark the best layer
        best = sub.loc[sub["auprc_mean"].idxmax()]
        ax.scatter(best["layer"], best["auprc_mean"],
                   s=80, color=color, zorder=5, edgecolors="white", lw=1.0)
        # annotate best layer number with background for readability
        ax.annotate(f"L{int(best['layer'])}",
                    (best["layer"], best["auprc_mean"]),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.9),
                    zorder=6)

    # Add grey label to legend
    ax.plot([], [], lw=1.0, color="#AAAAAA", alpha=0.5,
            label=f"other types (n={len(top_types) - len(highlight_types)})")

    # Add median + IQR band across all annotation types
    agg_by_layer = layer_rows.groupby("layer")["auprc_mean"].agg(
        ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    agg_by_layer.columns = ["median", "q25", "q75"]
    ax.fill_between(agg_by_layer.index, agg_by_layer["q25"], agg_by_layer["q75"],
                    color="#888780", alpha=0.12, zorder=0, label="IQR (all types)")
    ax.plot(agg_by_layer.index, agg_by_layer["median"],
            color="#888780", lw=2.0, linestyle="--", alpha=0.6,
            label="median (all types)", zorder=1)

    ax.set_xlabel("ESM-2 layer (1 = first, 33 = last)", fontsize=11)
    ax.set_ylabel("Attention AUPRC", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
              frameon=False, fontsize=9, ncol=3, columnspacing=1.0,
              handletextpad=0.5)
    ax.set_title("Layer-wise attention alignment with biological annotations",
                 fontsize=12, pad=55)
    ax.set_xlim(0.5, 33.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 33])

    # shade early/middle/late regions
    ax.axvspan(1, 11, alpha=0.03, color=PALETTE["attn"], zorder=0)
    ax.axvspan(11, 23, alpha=0.03, color=PALETTE["sae"], zorder=0)
    ax.axvspan(23, 34, alpha=0.03, color=PALETTE["ablation"], zorder=0)
    yhi = ax.get_ylim()[1]
    ax.text(6, yhi * 0.98, "early", ha="center", fontsize=9,
            color=PALETTE["attn"], alpha=0.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax.text(17, yhi * 0.98, "middle", ha="center", fontsize=9,
            color=PALETTE["sae"], alpha=0.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax.text(28.5, yhi * 0.98, "late", ha="center", fontsize=9,
            color=PALETTE["ablation"], alpha=0.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    fig.tight_layout()
    path = outdir / "fig5_layer_attention.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"    → {path.name}")


# =============================================================================
#  FIGURE 6 — PROTEIN CASE STUDIES
# =============================================================================

def _choose_case_study_proteins(ann_df, seq_by_acc,
                                 align_df, top_proteins_dir,
                                 forced_proteins=None):
    """
    Choose proteins for case studies.
    If forced_proteins is given, parse them.
    Otherwise, dynamically pick the best (feature, annotation) pairs from
    the multi-evidence directory, selecting the protein with highest
    ablation contrast for each pair.
    """
    if forced_proteins:
        selected = []
        for spec in forced_proteins:
            # Support both | and : as delimiter.
            # Prefer | because annotation names contain colons.
            if "|" in spec:
                parts = spec.split("|", 3)
            else:
                parts = spec.split(":", 3)
            if len(parts) != 4:
                log.warning(f"  Invalid spec: {spec!r}")
                continue
            acc, fid_str, at, label = [p.strip() for p in parts]
            if acc not in seq_by_acc:
                log.warning(f"  {acc} not available: sequence not in proteins TSV for split")
                continue
            # try to read n_proteins from evidence_summary.json
            n_proteins = None
            safe = at.lower().replace(" ", "_")
            ev_path = top_proteins_dir / f"f{fid_str}_{safe}" / "evidence_summary.json"
            if ev_path.exists():
                import json as _json
                try:
                    with open(ev_path) as _f:
                        _ev = _json.load(_f)
                    n_proteins = _ev.get("n_proteins")
                except Exception:
                    pass
            selected.append({"accession": acc, "feature_id": int(fid_str),
                              "annot_type": at, "label": label,
                              "length": len(seq_by_acc[acc]), "contrast": 0.0,
                              "n_proteins": n_proteins})
        return selected

    # ── Dynamic selection from multi-evidence directory ───────────────────
    ev_matrix_path = top_proteins_dir / "full_evidence_matrix.tsv"
    if ev_matrix_path.exists():
        ev_df = pd.read_csv(ev_matrix_path, sep="\t")
        selected = []
        for _, row in ev_df.iterrows():
            fid = int(row["feature_id"])
            at = row["annot_type"]
            safe = at.lower().replace(" ", "_")
            label = safe.replace(":", "").replace(";", "").replace(" ", "_")

            # Read n_proteins from evidence_summary.json
            n_proteins = None
            ev_path = top_proteins_dir / f"f{fid}_{safe}" / "evidence_summary.json"
            if ev_path.exists():
                import json as _json
                try:
                    with open(ev_path) as _f:
                        _ev = _json.load(_f)
                    n_proteins = _ev.get("n_proteins")
                except Exception:
                    pass

            # Read ablation.tsv to pick best protein
            abl_path = top_proteins_dir / f"f{fid}_{safe}" / "ablation.tsv"
            best_acc, best_contrast = None, -1.0
            if abl_path.exists():
                try:
                    abl_df = pd.read_csv(abl_path, sep="\t")
                    for _, ar in abl_df.iterrows():
                        acc = ar["accession"]
                        if acc in seq_by_acc:
                            contrast = float(ar.get("contrast", 0))
                            if contrast > best_contrast:
                                best_contrast = contrast
                                best_acc = acc
                except Exception:
                    pass

            if best_acc is None:
                # fallback: pick any protein with this annotation
                sub = ann_df[ann_df["feature_type"] == at]
                for acc in sub["accession"].unique():
                    if acc in seq_by_acc:
                        best_acc = acc
                        best_contrast = 0.0
                        break

            if best_acc:
                selected.append({
                    "accession": best_acc, "feature_id": fid,
                    "annot_type": at, "label": label,
                    "length": len(seq_by_acc[best_acc]),
                    "contrast": best_contrast,
                    "n_proteins": n_proteins,
                })
        return selected

    # ── Legacy fallback: hardcoded feature/annotation combos ──────────────
    cases = [
        {"feature_id": 1310, "annot_type": "Zinc finger",    "label": "zinc_finger"},
        {"feature_id": 3087, "annot_type": "Signal peptide", "label": "signal_peptide"},
        {"feature_id": 1696, "annot_type": "Active site",    "label": "active_site"},
    ]
    selected = []
    for c in cases:
        fid = c["feature_id"]
        at  = c["annot_type"]
        safe = at.lower().replace(" ", "_")
        tp_path = top_proteins_dir / f"f{fid}_{safe}" / "top_proteins.tsv"

        c["n_proteins"] = None
        ev_path = top_proteins_dir / f"f{fid}_{safe}" / "evidence_summary.json"
        if ev_path.exists():
            import json as _json
            try:
                with open(ev_path) as _f:
                    _ev = _json.load(_f)
                c["n_proteins"] = _ev.get("n_proteins")
            except Exception:
                pass

        if tp_path.exists():
            tp = pd.read_csv(tp_path, sep="\t")
            for _, row in tp.iterrows():
                acc = row["accession"]
                if acc in seq_by_acc:
                    c["accession"] = acc
                    c["length"]    = row["length"]
                    c["contrast"]  = row["contrast"]
                    break
        if "accession" not in c:
            sub = ann_df[ann_df["feature_type"] == at]
            for acc in sub["accession"].unique():
                if acc in seq_by_acc:
                    c["accession"] = acc
                    c["length"] = len(seq_by_acc[acc])
                    c["contrast"] = 0.0
                    break
        if "accession" in c:
            selected.append(c)
    return selected


def _choose_best_physchem_property(seq, mask, prop_tables):
    """Choose property with largest absolute annotated-vs-background effect."""
    best_name = "hydrophobicity"
    best_score = -1.0

    for name, table in prop_tables.items():
        arr = np.array([table.get(aa, 0.0) for aa in seq], dtype=np.float32)
        in_ann = arr[mask.astype(bool)]
        out_ann = arr[~mask.astype(bool)]

        if len(in_ann) < 2 or len(out_ann) < 2:
            continue

        diff = float(np.mean(in_ann) - np.mean(out_ann))
        pooled = float(np.std(arr) + 1e-8)
        effect = abs(diff) / pooled

        if effect > best_score:
            best_score = effect
            best_name = name

    return best_name


def _add_aa_strip(ax, seq, acts, start, end, cmap_name="YlGn"):
    """
    Compact amino-acid strip for annotated region.
    Shows residue letters with background color = SAE activation.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    subseq = seq[start:end]
    subacts = acts[start:end]
    xs = np.arange(start, end)

    if len(subacts) == 0:
        ax.axis("off")
        return

    vmax = max(float(np.max(subacts)), 1e-8)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    ax.set_xlim(start, end)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, (x, aa, act) in enumerate(zip(xs, subseq, subacts)):
        color = cmap(norm(act))
        rect = plt.Rectangle((x, 0), 1, 1, facecolor=color, edgecolor="white", lw=0.5)
        ax.add_patch(rect)

        text_color = "black" if norm(act) < 0.65 else "white"
        fontweight = "bold" if aa in {"C", "H"} else "normal"
        ax.text(x + 0.5, 0.5, aa, ha="center", va="center",
                fontsize=10, color=text_color, fontweight=fontweight)

    # sparse ticks
    tick_positions = xs[::max(1, len(xs)//8)]
    for x in tick_positions:
        ax.text(x + 0.5, -0.15, str(x + 1), ha="center", va="top", fontsize=9)


def fig6_case_study(case, ann_df, seq_by_acc, get_emb,
                    W_enc, b_enc, W_dec, b_dec,
                    attn_scores_df, outdir,
                    esm_lm_head=None, esm_tokenizer=None,
                    get_attn_scores=None,
                    attn_conv_df=None):
    fid = case["feature_id"]
    at = case["annot_type"]
    acc = case["accession"]
    seq = seq_by_acc[acc]
    label = case["label"]

    emb = get_emb(acc)
    L = emb.shape[0]
    seq = seq[:L]

    z = get_activations(emb, W_enc, b_enc)
    acts = z[:, fid].astype(np.float32)

    ann_acc = ann_df[(ann_df["accession"] == acc) & (ann_df["feature_type"] == at)]
    all_ann = ann_df[ann_df["accession"] == acc]
    mask = build_mask(ann_acc, L).astype(bool)

    # Option 1: annotation-direction embedding projection (SAE-independent)
    emb_proj, struct_align = annot_proj_delta(emb, fid, W_dec, mask)

    # Option 2: logit delta (ESM downstream causal test, only if ESM available)
    ldelta = None
    if esm_lm_head is not None:
        try:
            ldelta = logit_delta(emb, seq, fid, W_enc, b_enc, W_dec, b_dec,
                                  esm_lm_head, esm_tokenizer)
        except Exception as exc:
            log.warning(f"    logit_delta failed for {acc}: {exc}")

    # Determine best attention layer for this annotation type
    best_attn_layer = None
    if attn_conv_df is not None and not attn_conv_df.empty:
        row = attn_conv_df[attn_conv_df["annot_type"] == at]
        if not row.empty:
            agg = row.iloc[0]["best_attn_agg"]
            import re as _re
            m = _re.search(r"layer_(\d+)_to", str(agg))
            if m:
                best_attn_layer = int(m.group(1))

    # Attention at best layer (or fallback to smoothed activation)
    attn_signal = None
    attn_layer_used = best_attn_layer
    if get_attn_scores is not None:
        try:
            attn_signal = get_attn_scores(acc, layer_override=best_attn_layer)
            if attn_signal is not None:
                attn_signal = attn_signal[:L]
        except Exception as exc:
            log.warning(f"    attention extraction failed for {acc}: {exc}")

    # dynamic physicochemical choice
    prop_name = _choose_best_physchem_property(seq, mask, PROP_TABLES)
    prop_arr = np.array([PROP_TABLES[prop_name].get(aa, 0.0) for aa in seq], dtype=np.float32)

    xs = np.arange(L)

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(
        5, 1,
        height_ratios=[0.6, 1.3, 1.1, 1.1, 1.0],
        hspace=0.35,
    )

    ax0 = fig.add_subplot(gs[0])  # annotation
    ax1 = fig.add_subplot(gs[1])  # sae
    ax2 = fig.add_subplot(gs[2])  # ablation
    ax3 = fig.add_subplot(gs[3])  # attention
    ax4 = fig.add_subplot(gs[4])  # physchem

    # annotation track: only highlight main annotation + maybe 2 context tracks
    primary_color = "#E76F51"
    context_colors = ["#4F83C2", "#7F77DD", "#BA7517"]
    other_types = [t for t in all_ann["feature_type"].unique() if t != at][:3]

    y = 0.5
    for _, row in ann_acc.iterrows():
        s = max(0, int(row["start"]) - 1)
        e = min(L, int(row["end"]))
        ax0.barh(y, e - s, left=s, height=0.35, color=primary_color, alpha=0.9, label=at)

    for i, ot in enumerate(other_types):
        sub = all_ann[all_ann["feature_type"] == ot]
        for _, row in sub.iterrows():
            s = max(0, int(row["start"]) - 1)
            e = min(L, int(row["end"]))
            ax0.barh(y, e - s, left=s, height=0.22, color=context_colors[i], alpha=0.7, label=ot)

    ax0.set_xlim(0, L)
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.set_ylabel("Annotations", rotation=0, labelpad=40, va="center", fontsize=11)
    ax0.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # deduplicate legend — place outside to avoid overlap with annotation bars
    handles, labels = ax0.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax0.legend(uniq.values(), uniq.keys(),
               loc="upper left", bbox_to_anchor=(0.0, 1.55),
               frameon=False, fontsize=9.5, ncol=len(uniq))

    def shade_annotations(ax):
        """Shade target annotation regions with colored band + edge lines."""
        for _, row in ann_acc.iterrows():
            s = max(0, int(row["start"]) - 1)
            e = min(L, int(row["end"]))
            ax.axvspan(s, e, alpha=0.18, color=primary_color, zorder=0)
            ax.axvline(s, color=primary_color, lw=0.5, alpha=0.4, zorder=0)
            ax.axvline(e, color=primary_color, lw=0.5, alpha=0.4, zorder=0)

    # SAE
    ax1.fill_between(xs, acts, color="#2A9D8F", alpha=0.35)
    ax1.plot(xs, acts, color="#2A9D8F", lw=0.8)
    shade_annotations(ax1)
    ax1.set_xlim(0, L)
    ax1.set_xticks([])
    ax1.set_ylabel(f"f{fid}\nactivation", rotation=0, labelpad=40, va="center", fontsize=11)

    # Embedding annotation projection (SAE-independent)
    ax2.fill_between(xs, emb_proj, color="#8E8DF0", alpha=0.35)
    ax2.plot(xs, emb_proj, color="#8E8DF0", lw=0.8)
    shade_annotations(ax2)
    ax2.set_xlim(0, L)
    ax2.set_xticks([])
    sa_sign = "+" if struct_align >= 0 else ""
    ax2.set_ylabel(f"Emb. annot.\nproj. (SA={sa_sign}{struct_align:.2f})",
                   rotation=0, labelpad=40, va="center", fontsize=11)

    # Attention at best layer, logit delta, or smoothed activation fallback
    if attn_signal is not None:
        ax3.fill_between(xs, attn_signal, color="#3266AD", alpha=0.35)
        ax3.plot(xs, attn_signal, color="#3266AD", lw=0.8)
        layer_label = f"layer {attn_layer_used + 1}" if attn_layer_used is not None else "layer 33"
        ax3.set_ylabel(f"Attention\n({layer_label})", rotation=0, labelpad=40, va="center", fontsize=11)
    elif ldelta is not None:
        ax3.fill_between(xs, ldelta, color="#BA7517", alpha=0.35)
        ax3.plot(xs, ldelta, color="#BA7517", lw=0.8)
        ax3.axhline(0, color="#cccccc", lw=0.5)
        ax3.set_ylabel("Logit Δ\n(ESM LM)", rotation=0, labelpad=40, va="center", fontsize=11)
    else:
        window = max(3, L // 50)
        attn_proxy = np.convolve(acts, np.ones(window) / window, mode="same")
        attn_proxy = (attn_proxy - attn_proxy.min()) / (attn_proxy.max() - attn_proxy.min() + 1e-8)
        ax3.fill_between(xs, attn_proxy, color="#4F83C2", alpha=0.35)
        ax3.plot(xs, attn_proxy, color="#4F83C2", lw=0.8)
        ax3.set_ylabel("Activation\nsmoothed", rotation=0, labelpad=40, va="center", fontsize=11)
    shade_annotations(ax3)
    ax3.set_xlim(0, L)
    ax3.set_xticks([])

    # Physchem
    ax4.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax4.plot(xs, prop_arr, color="#C88719", lw=0.9)
    shade_annotations(ax4)
    ax4.set_xlim(0, L)
    ax4.set_xlabel("Residue position", fontsize=11)
    ax4.set_ylabel(prop_name.replace("_", "\n"), rotation=0, labelpad=40, va="center", fontsize=11)

    n_proteins = case.get("n_proteins")
    n_prot_str = f" | N={n_proteins} proteins" if n_proteins is not None else ""
    fig.suptitle(
        f"{acc} | {at} | feature f{fid} | L={L}{n_prot_str}",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    outpath = outdir / f"fig6_clean_{label}_{acc}.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  ATTENTION HELPER — extract layer-33 attention per residue
# =============================================================================

def _build_attn_scorer(esm2_model, converter, backend, device, layer=33, max_len=1022):
    """
    Returns a function  get_attn_scores(acc) -> np.ndarray [L]
    that computes per-residue mean attention received at the specified layer.
    
    For HuggingFace backend, loads a separate model with eager attention
    to support output_attentions=True.
    """
    _cache = {}
    _seq_cache = {}
    _attn_model = [None]  # mutable container for lazy init
    _default_layer = layer

    def _get_attn_model():
        if _attn_model[0] is not None:
            return _attn_model[0]
        if backend == "hf":
            from transformers import EsmModel
            model_name = esm2_model.config._name_or_path
            log.info(f"  Loading ESM-2 with eager attention for attention extraction ...")
            m = EsmModel.from_pretrained(
                model_name, attn_implementation="eager"
            ).to(device).eval()
            _attn_model[0] = m
        else:
            _attn_model[0] = esm2_model
        return _attn_model[0]

    def register_seq(acc, seq):
        _seq_cache[acc] = seq

    def get_attn_scores(acc, layer_override=None):
        use_layer = layer_override if layer_override is not None else _default_layer
        cache_key = (acc, use_layer)
        if cache_key in _cache:
            return _cache[cache_key]
        seq = _seq_cache.get(acc)
        if seq is None:
            return None
        seq = seq[:max_len]
        L = len(seq)

        attn_model = _get_attn_model()

        with torch.no_grad():
            if backend == "esm":
                _, _, tokens = converter([(acc, seq)])
                tokens = tokens.to(device)
                out = attn_model(tokens, repr_layers=[use_layer],
                                 return_contacts=False,
                                 need_head_weights=True)
                attn = out["attentions"]  # [1, n_layers, n_heads, L+2, L+2]
                layer_attn = attn[0, use_layer - 1, :, 1:L+1, 1:L+1]  # [n_heads, L, L]
                attn_to = layer_attn.mean(dim=0).sum(dim=0)  # [L]
                result = attn_to.cpu().numpy().astype(np.float32)
            else:
                inputs = converter(seq, return_tensors="pt", truncation=True,
                                   max_length=max_len + 2).to(device)
                out = attn_model(**inputs, output_attentions=True,
                                 output_hidden_states=False)
                # out.attentions: tuple of [1, n_heads, L_tok, L_tok] per layer
                layer_attn = out.attentions[use_layer - 1][0, :, 1:L+1, 1:L+1]  # [n_heads, L, L]
                attn_to = layer_attn.mean(dim=0).sum(dim=0)  # [L]
                result = attn_to.cpu().numpy().astype(np.float32)

        _cache[cache_key] = result
        return result

    return get_attn_scores, register_seq


# =============================================================================
#  MAIN
# =============================================================================

def main():
    global torch
    args = parse_args()

    try:
        import torch as _t
        torch = _t
    except ImportError:
        sys.exit("\n[ERROR] pip install torch\n")

    args.outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    log.info(f"Device: {device}")

    # ── load everything ───────────────────────────────────────────────────────
    log.info("Loading data ...")

    # SAE
    W_enc, b_enc, W_dec, b_dec = load_sae(args.checkpoint, device)

    # ESM-2 LM head for logit-delta ablation (optional)
    esm_lm_head   = None
    esm_tokenizer = None
    if args.esm_model:
        esm_lm_head, esm_tokenizer = load_esm_for_logits(
            args.esm_model, device, args.esm_cache_dir
        )

    # Proteins
    prot_df  = pd.read_csv(args.proteins, sep="\t", low_memory=False)
    prot_df  = prot_df[prot_df["split"] == args.split]
    seq_by_acc = dict(zip(prot_df["accession"], prot_df["sequence"]))

    # Annotations
    ann_df = pd.read_csv(args.annotations, sep="\t", low_memory=False)
    ann_df.columns = ann_df.columns.str.strip().str.lower()
    ann_df = ann_df[ann_df["split"] == args.split].copy()
    ann_df["start"] = ann_df["start"].astype(int)
    ann_df["end"]   = ann_df["end"].astype(int)

    # use annot_subtype as feature_type when available (matches multi_evidence.py)
    if ("annot_subtype" in ann_df.columns
            and ann_df["annot_subtype"].notna().any()
            and ann_df["annot_subtype"].nunique() > ann_df["feature_type"].nunique()):
        ann_df["feature_type"] = ann_df["annot_subtype"]
        log.info(f"  Using annot_subtype as feature_type "
                 f"({ann_df['feature_type'].nunique()} subtypes)")

    # ── load ESM-2 for on-the-fly embeddings ─────────────────────────────────
    log.info("Loading ESM-2 ...")
    esm2_model, converter, backend = load_esm2(args.esm2_model, device)
    get_emb = lambda acc: _get_embedding_live(
        acc, seq_by_acc[acc], esm2_model, converter, backend,
        device, args.esm2_layer, args.max_seq_len,
    )

    # ── build attention scorer (best layer per type, default 33) ────────────
    log.info("Building attention scorer ...")
    get_attn_scores, register_attn_seq = _build_attn_scorer(
        esm2_model, converter, backend, device,
        layer=args.esm2_layer, max_len=args.max_seq_len,
    )
    # register all sequences so the scorer can look them up
    for acc, seq in seq_by_acc.items():
        register_attn_seq(acc, seq)

    # Alignment results
    align_path = args.alignment_dir / "alignment_scores.parquet"
    align_df   = (pd.read_parquet(align_path)
                  if align_path.exists()
                  else pd.DataFrame())

    per_annot_path = args.alignment_dir / "per_annot_summary.tsv"
    per_annot_df   = (pd.read_csv(per_annot_path, sep="\t")
                      if per_annot_path.exists()
                      else pd.DataFrame())

    # Multi-evidence matrix
    ev_path  = args.multi_ev_dir / "full_evidence_matrix.tsv"
    ev_df    = (pd.read_csv(ev_path, sep="\t")
                if ev_path.exists()
                else pd.DataFrame())

    # Attention convergence
    attn_conv_path = args.attention_dir / "convergence_with_sae.tsv"
    attn_conv_df   = (pd.read_csv(attn_conv_path, sep="\t")
                      if attn_conv_path.exists()
                      else pd.DataFrame())

    attn_scores_path = args.attention_dir / "alignment_scores.parquet"
    attn_scores_df   = (pd.read_parquet(attn_scores_path)
                        if attn_scores_path.exists()
                        else pd.DataFrame())

    # ── generate figures ──────────────────────────────────────────────────────
    log.info("\nGenerating figures ...")

    if not align_df.empty:
        fig1_heatmap(align_df, args.outdir)
    else:
        log.warning("  Skipping Fig 1: alignment_scores.parquet not found")

    if not ev_df.empty:
        fig2_multi_evidence(ev_df, attn_conv_df, args.outdir,
                            ann_df=ann_df, seq_by_acc=seq_by_acc,
                            get_emb=get_emb,
                            get_attn_scores=get_attn_scores)
    else:
        log.warning("  Skipping Fig 2: missing multi-evidence data")

    if not attn_conv_df.empty:
        fig3_convergence(attn_conv_df, args.outdir,
                         ev_matrix=ev_df if not ev_df.empty else None,
                         ann_df=ann_df, seq_by_acc=seq_by_acc,
                         get_attn_scores=get_attn_scores,
                         sae_summary_df=per_annot_df)
    else:
        log.warning("  Skipping Fig 3: attention convergence not found")

    if not align_df.empty:
        ann_by_acc = {acc: grp for acc, grp in ann_df.groupby("accession")}
        fig4_structural(ann_df, get_emb, W_enc, b_enc,
                        align_df, args.outdir)
    else:
        log.warning("  Skipping Fig 4: alignment data not found")

    if not attn_scores_df.empty:
        fig5_layer_attention(attn_scores_df, args.outdir,
                             sae_summary_df=per_annot_df)
    else:
        log.warning("  Skipping Fig 5: attention scores not found")

    # Case studies
    forced = ([s.strip() for s in args.case_proteins.split(",")]
              if args.case_proteins else None)
    cases = _choose_case_study_proteins(
        ann_df, seq_by_acc,
        per_annot_df, args.multi_ev_dir,
        forced_proteins=forced,
    )

    log.info(f"  Case studies: {len(cases)} proteins selected")
    for case in cases:
        if "accession" not in case:
            log.warning(f"  No protein found for {case['annot_type']}")
            continue
        n_prot_info = f", N={case['n_proteins']} proteins in cluster" if case.get("n_proteins") is not None else ""
        log.info(f"  Case: {case['accession']} | f{case['feature_id']} | {case['annot_type']}{n_prot_info}")
        fig6_case_study(
            case, ann_df, seq_by_acc, get_emb,
            W_enc, b_enc, W_dec, b_dec,
            attn_scores_df, args.outdir,
            esm_lm_head=esm_lm_head,
            esm_tokenizer=esm_tokenizer,
            get_attn_scores=get_attn_scores,
            attn_conv_df=attn_conv_df,
        )

    # ── manifest ──────────────────────────────────────────────────────────────
    figs = sorted(args.outdir.glob("fig*.pdf"))
    log.info(f"\n✅ Done. {len(figs)} figures in {args.outdir}")
    for f in figs:
        log.info(f"    {f.name}")


if __name__ == "__main__":
    main()