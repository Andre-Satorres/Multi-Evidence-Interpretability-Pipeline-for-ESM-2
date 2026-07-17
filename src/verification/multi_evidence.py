"""
Multi-evidence analysis for top SAE features.

For each candidate (feature_id, annot_type) pair this script computes:

Evidence layer 1 — Physicochemical alignment
  For each of 8 residue properties (hydrophobicity, charge, volume,
  aromaticity, disorder propensity, helix propensity, sheet propensity,
  polarity), compute the mean property value inside vs outside the
  annotation, and test whether the annotated residues have a distinctive
  physicochemical signature.  This answers: "does f5229 prefer hydrophobic
  residues?" without relying on any external database.

Evidence layer 2 — Causal ablation
  Zero the activation of feature f in the SAE latent space and measure
  how much the reconstructed embedding changes at each residue position.
  Computes:
    ablation_delta[i] = ||x_reconstructed - x_ablated||_2  per residue i
  Tests whether delta is larger inside the annotation than outside.
  This is causal evidence: the feature is *necessary* for the representation
  of annotated regions.

Evidence layer 3 — Cross-feature convergence
  Identifies other features that co-activate with f on the same proteins
  and same regions.  High co-activation = multiple independent SAE features
  converge on the same biological concept.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import OUT_DIR, ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH

torch = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
#  PHYSICOCHEMICAL TABLES
# =============================================================================

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8,  'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5,
    'Q':-3.5,  'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5,
    'L': 3.8,  'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6,
    'S':-0.8,  'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2,
}

# Molecular volume (Å³, approximate)
VOLUME = {
    'A': 88.6,  'R':173.4, 'N':114.1, 'D':111.1, 'C':108.5,
    'Q':143.8,  'E':138.4, 'G': 60.1, 'H':153.2, 'I':166.7,
    'L':166.7,  'K':168.6, 'M':162.9, 'F':189.9, 'P':112.7,
    'S': 89.0,  'T':116.1, 'W':227.8, 'Y':193.6, 'V':140.0,
}

# Net charge at pH 7 (simplified)
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D':-1, 'C': 0,
    'Q': 0, 'E':-1, 'G': 0, 'H': 0, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}

# Aromaticity (1 = aromatic)
AROMATICITY = {aa: 1 if aa in ('F','Y','W') else 0
               for aa in 'ACDEFGHIKLMNPQRSTVWY'}

# Helix propensity (Pace & Scholtz, 1998) — kcal/mol, menor = mais favorável
# Nota: escala original não inclui Prolina (estruturalmente incompatível com hélice)
HELIX_PROP = {
    'A': 0.00, 'L': 0.21, 'R': 0.21, 'M': 0.24, 'K': 0.26,
    'Q': 0.39, 'E': 0.40, 'I': 0.41, 'W': 0.49, 'S': 0.50,
    'Y': 0.53, 'F': 0.54, 'V': 0.61, 'H': 0.61, 'N': 0.65,
    'T': 0.66, 'C': 0.68, 'D': 0.69, 'G': 1.00, 'P': 3.16
}

# Beta-sheet propensity (thermodynamic scale, Kim & Berg, 1993)
# Fonte: Kim, C.A. & Berg, J.M. (1993). Nature, 362, 267-270.
# AAindex: KIMC930101 — valores mais negativos = mais favorável à folha-beta
SHEET_PROP = {
    'A': -0.35, 'R': -0.44, 'N': -0.38, 'D': -0.41, 'C': -0.47,
    'Q': -0.40, 'E': -0.41, 'G':  0.00, 'H': -0.46, 'I': -0.56,
    'L': -0.48, 'K': -0.41, 'M': -0.46, 'F': -0.55, 'P': -0.23,
    'S': -0.39, 'T': -0.48, 'W': -0.48, 'Y': -0.50, 'V': -0.53,
}

# Disorder propensity (TOP-IDP scale, Campen et al., 2008)
# Fonte: Campen, A. et al. (2008). Protein & Peptide Letters, 15(9), 956-963.
# Valores mais negativos = promovem ordem; mais positivos = promovem desordem
DISORDER_PROP = {
    'W': -0.884, 'F': -0.697, 'Y': -0.510, 'I': -0.486, 'M': -0.397,
    'L': -0.326, 'V': -0.121, 'N':  0.007, 'C':  0.020, 'T':  0.059,
    'A':  0.060, 'G':  0.166, 'R':  0.180, 'D':  0.192, 'H':  0.303,
    'Q':  0.318, 'K':  0.586, 'S':  0.341, 'E':  0.736, 'P':  0.987,
}

PROPERTIES = {
    "hydrophobicity": HYDROPHOBICITY,
    "charge":         CHARGE,
    "volume":         VOLUME,
    "aromaticity":    AROMATICITY,
    "helix_prop":     HELIX_PROP,
    "sheet_prop":     SHEET_PROP,
    "disorder_prop":  DISORDER_PROP,
    # "polarity":       POLARITY, removida por ser redundante com hydrophobicity
}

def seq_to_properties(seq: str) -> dict[str, np.ndarray]:
    """Convert amino acid sequence to per-residue property arrays.

    Resíduos sem valor definido em uma escala específica (ex.: Prolina em
    HELIX_PROP; códigos especiais X/B/Z/U/O em qualquer propriedade) recebem
    NaN, e devem ser excluídos explicitamente do cálculo de evidência para
    aquela propriedade -- nunca substituídos por um valor numérico arbitrário.
    """
    out = {}
    for name, table in PROPERTIES.items():
        arr = np.array([table.get(aa, np.nan) for aa in seq], dtype=np.float32)
        out[name] = arr
    return out

# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-evidence analysis: physicochemical + causal ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--alignment",  type=Path,
                   default=OUT_DIR / "feature_alignment" / "per_annot_summary.tsv")
    p.add_argument("--all-scores", type=Path,
                   default=OUT_DIR / "feature_alignment" / "alignment_scores.parquet")
    p.add_argument("--esm2-model", default="esm2_t33_650M_UR50D",
                   help="ESM-2 model name for on-the-fly embedding.")
    p.add_argument("--esm2-layer", type=int, default=33,
                   help="ESM-2 transformer layer to extract embeddings from.")
    p.add_argument("--max-seq-len", type=int, default=1022,
                   help="Truncate sequences longer than this.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--annotations",type=Path,
                   default=ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH)
    p.add_argument("--proteins",   type=Path,
                   default=OUT_DIR.parent / "data" / "proteins_with_split.tsv")
    p.add_argument("--split",      default="test")
    p.add_argument("--top-pairs",  type=int, default=8)
    p.add_argument("--sort-by",    default="odds_ratio",
                   choices=["odds_ratio","auprc_mean","mean_contrast"],
                   help="Metric to rank pairs by.")
    p.add_argument("--activation-threshold", type=float, default=0.1)
    p.add_argument("--top-proteins",type=int, default=50,
                   help="Max proteins per pair to process.")
    p.add_argument("--outdir",     type=Path,
                   default=OUT_DIR / "multi_evidence")
    p.add_argument("--device",     default="auto")
    p.add_argument("--esm-model",   default=None,
               help="ESM-2 model name to enable logit-delta evidence layer. "
                    "E.g. facebook/esm2_t33_650M_UR50D. Optional.")
    p.add_argument("--esm-cache-dir", type=Path, default=None,
               help="HuggingFace cache directory for ESM-2 weights.")
    p.add_argument("--min-clusters-focal",  type=int, default=30)
    p.add_argument("--min-clusters-domain", type=int, default=5)
    p.add_argument("--rpp-threshold", type=float, default=None,
               help="Override GMM threshold for focal vs domain classification")
    return p.parse_args()


# =============================================================================
#  SAE HELPERS
# =============================================================================

def load_sae(checkpoint_path, device):
    """Load encoder + decoder weights. Returns (W_enc, b_enc, W_dec, b_dec)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model_state_dict"]
    W_enc = sd["encoder.weight"].float().to(device)  # [K, D]
    b_enc = sd["encoder.bias"].float().to(device)    # [K]
    W_dec = sd["decoder.weight"].float().to(device)  # [D, K]
    b_dec = sd["decoder.bias"].float().to(device)    # [D]
    K, D  = W_enc.shape
    log.info(f"  SAE: D={D}, K={K}")
    return W_enc, b_enc, W_dec, b_dec, K, D


def encode_decode(emb, W_enc, b_enc, W_dec, b_dec):
    """
    Full SAE forward pass.
    emb: [L, D] numpy float32
    Returns z [L, K] and x_hat [L, D], both numpy float32.
    """
    with torch.no_grad():
        x     = torch.tensor(emb).float().to(W_enc.device)
        z     = torch.relu(x @ W_enc.T + b_enc)        # [L, K]
        x_hat = z @ W_dec.T + b_dec                    # [L, D]
    return z.cpu().numpy(), x_hat.cpu().numpy()


def ablate_feature(emb, feature_id, W_enc, b_enc, W_dec, b_dec):
    """
    Ablate (zero) one feature and reconstruct.
    Returns (x_hat - x_hat_ablated) delta [L, D] numpy float32.
    """
    with torch.no_grad():
        x     = torch.tensor(emb).float().to(W_enc.device)
        z     = torch.relu(x @ W_enc.T + b_enc)        # [L, K]
        z_abl = z.clone()
        z_abl[:, feature_id] = 0.0                     # zero the feature
        x_abl = z_abl @ W_dec.T + b_dec                # [L, D]
        x_hat = z     @ W_dec.T + b_dec                # [L, D] original recon
    return (x_hat - x_abl).cpu().numpy()               # [L, D] delta


# =============================================================================
#  ESM LOGIT-DELTA HELPERS
# =============================================================================

def load_esm_for_logits(model_name, device, cache_dir=None):
    """
    Load the ESM-2 LM head (the masked-language-model prediction head).
    The stored embeddings are last hidden states of EsmModel, so lm_head
    can be applied to them directly to obtain per-residue AA log-probs.
    Returns (lm_head module, tokenizer).
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    log.info(f"  Loading ESM-2 LM head from {model_name} ...")
    kw = {"cache_dir": str(cache_dir)} if cache_dir else {}
    model = AutoModelForMaskedLM.from_pretrained(model_name, **kw).float().to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
    lm_head = model.lm_head
    return lm_head, tokenizer


def logit_delta(seq_emb, seq, feature_id, W_enc, b_enc, W_dec, b_dec,
                lm_head, tokenizer):
    """
    Per-residue drop in log-probability of the correct amino acid when
    feature_id's contribution is removed from the embedding.

    This is a downstream causal test: the effect is measured on ESM-2's
    masked-LM predictions, NOT on the SAE's own reconstruction, which
    would be circular.

    delta[i] > 0  → ablating this feature hurts prediction at residue i
    delta[i] < 0  → ablating improves prediction (feature was misleading)

    Returns: delta_lp np.ndarray [L] float32
    """
    import torch.nn.functional as F_func
    with torch.no_grad():
        x     = torch.tensor(seq_emb).float().to(W_enc.device)
        z     = torch.relu(x @ W_enc.T + b_enc)           # [L, K]
        # feature k's contribution to the SAE reconstruction
        # W_dec has shape [D, K]; W_dec[:,k] is the decoder direction [D]
        feat_dir   = W_dec[:, feature_id]                 # [D]
        feat_acts  = z[:, feature_id:feature_id+1]        # [L, 1]
        feat_contrib = feat_acts * feat_dir               # [L, D] broadcast
        # counterfactual embedding: remove feature k's reconstruction contrib
        x_cf = x - feat_contrib                           # [L, D]

        logits_orig = lm_head(x.unsqueeze(0)).squeeze(0)   # [L, vocab]
        logits_cf   = lm_head(x_cf.unsqueeze(0)).squeeze(0)

        enc_out = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
        token_ids = enc_out["input_ids"][0, :len(seq_emb)].to(W_enc.device)  # [L]

        lp_orig = F_func.log_softmax(logits_orig, dim=-1)
        lp_cf   = F_func.log_softmax(logits_cf,   dim=-1)
        idx = torch.arange(len(token_ids), device=W_enc.device)
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


# =============================================================================
#  EVIDENCE LAYER 1 — PHYSICOCHEMICAL
# =============================================================================

def physicochemical_evidence(
    accs:         list[str],
    annot_type:   str,
    seq_by_acc:   dict[str, str],
    ann_by_acc:   dict,
    threshold_act: float,
    feature_id:   int,
    get_emb,
    W_enc, b_enc,
) -> tuple[pd.DataFrame, dict]:
    """
    For each physicochemical property, compute:
      mean_in   : mean property value at annotated residues
      mean_out  : mean property value at non-annotated residues
      contrast  : mean_in - mean_out
      cohen_d   : effect size (mean_in - mean_out) / pooled_std

    Also tests whether high-activation residues preferentially have
    the physicochemical signature expected for the annotation type.
    """
    prop_in  = defaultdict(list)
    prop_out = defaultdict(list)
    # also: property values at high-activation vs low-activation residues
    prop_act_hi = defaultdict(list)
    prop_act_lo = defaultdict(list)

    for acc in accs:
        if acc not in seq_by_acc or acc not in ann_by_acc:
            continue
        seq = seq_by_acc[acc]
        L   = len(seq)

        ann_sub = ann_by_acc[acc][ann_by_acc[acc]["feature_type"] == annot_type]
        if ann_sub.empty:
            continue
        mask = build_mask(ann_sub, L)

        props = seq_to_properties(seq)

        # get activation for this feature
        emb  = get_emb(acc)
        L_emb = emb.shape[0]
        if L_emb != L:
            L = min(L, L_emb)
            mask = mask[:L]

        with torch.no_grad():
            x   = torch.tensor(emb[:L]).float().to(W_enc.device)
            z   = torch.relu(x @ W_enc.T + b_enc)
        acts = z[:, feature_id].cpu().numpy()

        hi_act = acts > threshold_act
        lo_act = ~hi_act

        for pname, parr in props.items():
            parr = parr[:L]
            in_idx  = mask == 1
            out_idx = mask == 0
            if in_idx.sum() > 0:
                prop_in[pname].extend(parr[in_idx].tolist())
            if out_idx.sum() > 0:
                prop_out[pname].extend(parr[out_idx].tolist())
            if hi_act.sum() > 0:
                prop_act_hi[pname].extend(parr[hi_act].tolist())
            if lo_act.sum() > 0:
                prop_act_lo[pname].extend(parr[lo_act].tolist())

    rows = []
    scores = {}
    for pname in PROPERTIES:
        vi = np.array(prop_in[pname])
        vo = np.array(prop_out[pname])
        va = np.array(prop_act_hi[pname])
        vb = np.array(prop_act_lo[pname])
    
        # contar apenas valores válidos (não-NaN) para decidir se há dado suficiente
        n_vi, n_vo = np.sum(~np.isnan(vi)), np.sum(~np.isnan(vo))
        if n_vi < 2 or n_vo < 2:
            continue  # dado insuficiente após excluir NaNs -- não confiável
    
        mi, mo = np.nanmean(vi), np.nanmean(vo)
        si, so = np.nanstd(vi) + 1e-8, np.nanstd(vo) + 1e-8
        pooled_std = np.sqrt((si**2 + so**2) / 2)
        cohen_d = (mi - mo) / pooled_std
    
        ma = np.nanmean(va) if np.sum(~np.isnan(va)) >= 2 else float("nan")
        mb = np.nanmean(vb) if np.sum(~np.isnan(vb)) >= 2 else float("nan")
        act_cohen_d = (ma - mb) / pooled_std if not (np.isnan(ma) or np.isnan(mb)) else float("nan")

        rows.append({
            "property":       pname,
            "mean_in":        float(mi),
            "mean_out":       float(mo),
            "contrast":       float(mi - mo),
            "cohen_d":        float(cohen_d),
            "mean_act_hi":    float(ma),
            "mean_act_lo":    float(mb),
            "act_cohen_d":    float(act_cohen_d),
            "n_in":           len(vi),
            "n_out":          len(vo),
        })
        scores[pname] = float(cohen_d)

    df = pd.DataFrame(rows).sort_values("cohen_d", key=abs, ascending=False)
    return df, scores


# =============================================================================
#  EVIDENCE LAYER 2 — CAUSAL ABLATION
# =============================================================================

def ablation_evidence(
    accs:         list[str],
    annot_type:   str,
    ann_by_acc:   dict,
    get_emb,
    feature_id:   int,
    W_enc, b_enc, W_dec, b_dec,
) -> tuple[pd.DataFrame, dict]:
    """
    For each protein, measure the L2 norm of the ablation delta at each
    residue, then compare inside vs outside the annotation.

    ablation_delta[i] = ||x_hat[i] - x_hat_ablated[i]||_2

    If the feature is necessary for the representation of annotated regions,
    delta_in >> delta_out.
    """
    records = []

    D = W_dec.shape[0]
    emb_in_sum  = np.zeros(D, dtype=np.float64)
    emb_out_sum = np.zeros(D, dtype=np.float64)
    n_in = 0; n_out = 0

    for acc in accs:
        if acc not in ann_by_acc:
            continue
        ann_sub = ann_by_acc[acc][ann_by_acc[acc]["feature_type"] == annot_type]
        if ann_sub.empty:
            continue

        emb  = get_emb(acc)
        L    = emb.shape[0]
        mask = build_mask(ann_sub, L)

        bool_mask = mask.astype(bool)
        if bool_mask.sum() > 0:
            emb_in_sum  += emb[:L][bool_mask].astype(np.float64).sum(0)
            n_in  += int(bool_mask.sum())
        if (~bool_mask).sum() > 0:
            emb_out_sum += emb[:L][~bool_mask].astype(np.float64).sum(0)
            n_out += int((~bool_mask).sum())

        # delta: [L, D]
        delta = ablate_feature(emb, feature_id, W_enc, b_enc, W_dec, b_dec)
        # L2 norm per residue: [L]
        delta_norm = np.linalg.norm(delta, axis=1)

        in_idx  = mask == 1
        out_idx = mask == 0

        mean_in  = float(delta_norm[in_idx].mean())  if in_idx.sum()  > 0 else 0.0
        mean_out = float(delta_norm[out_idx].mean()) if out_idx.sum() > 0 else 0.0
        std_in   = float(delta_norm[in_idx].std())   if in_idx.sum()  > 1 else 0.0
        std_out  = float(delta_norm[out_idx].std())  if out_idx.sum() > 1 else 0.0

        # effect size
        pooled = np.sqrt((std_in**2 + std_out**2) / 2) + 1e-8
        cohen_d = (mean_in - mean_out) / pooled

        records.append({
            "accession":    acc,
            "length":       L,
            "n_annot_res":  int(mask.sum()),
            "mean_delta_in":  mean_in,
            "mean_delta_out": mean_out,
            "contrast":       mean_in - mean_out,
            "cohen_d":        float(cohen_d),
        })

    if not records:
        return pd.DataFrame(), {}

    df = pd.DataFrame(records).sort_values("contrast", ascending=False)
    summary = {
        "mean_contrast":   float(df["contrast"].mean()),
        "median_contrast": float(df["contrast"].median()),
        "mean_cohen_d":    float(df["cohen_d"].mean()),
        "frac_positive":   float((df["contrast"] > 0).mean()),
        "n_proteins":      len(df),
    }

    if n_in > 0 and n_out > 0:
        v = (emb_in_sum / n_in) - (emb_out_sum / n_out)
        v_norm = np.linalg.norm(v) + 1e-8
        v = v / v_norm
        w_feat = W_dec[:, feature_id].cpu().float().numpy()   # [D]
        w_norm = np.linalg.norm(w_feat) + 1e-8
        structural_alignment = float(np.dot(w_feat / w_norm, v))
    else:
        structural_alignment = 0.0
    summary["structural_alignment"] = structural_alignment

    return df, summary


# =============================================================================
#  EVIDENCE LAYER 3 — FEATURE CONVERGENCE
# =============================================================================

def convergence_evidence(
    all_scores_df: pd.DataFrame,
    feature_id:    int,
    annot_type:    str,
    top_n:         int = 10,
) -> pd.DataFrame:
    """
    Find other features that also align with the same annotation type,
    sorted by AUPRC. This shows how many independent SAE features
    converge on the same biological concept.
    """
    same_annot = (all_scores_df[all_scores_df["annot_type"] == annot_type]
                  .sort_values("auprc_mean", ascending=False)
                  .head(top_n + 1))
    # exclude the focal feature
    same_annot = same_annot[same_annot["feature_id"] != feature_id]
    return same_annot[["feature_id","auprc_mean","odds_ratio","fisher_p","n_proteins"]].head(top_n)


def logit_delta_evidence(
    accs:          list[str],
    annot_type:    str,
    ann_by_acc:    dict,
    get_emb,
    seq_by_acc:    dict[str, str],
    feature_id:    int,
    W_enc, b_enc, W_dec, b_dec,
    lm_head, tokenizer,
) -> tuple[pd.DataFrame, dict]:
    """
    Measures the causal effect of feature_id on ESM-2 masked-LM predictions.
    Independent of SAE reconstruction: the effect is measured on amino-acid
    prediction log-probabilities, not on the SAE's own reconstruction delta.

    For each protein, compares mean logit-delta inside vs outside the annotation.
    If the feature genuinely encodes annotated-region information, ablating it
    should reduce prediction quality selectively at annotated positions.
    """
    records = []
    for acc in accs:
        if acc not in ann_by_acc or acc not in seq_by_acc:
            continue
        ann_sub = ann_by_acc[acc][ann_by_acc[acc]["feature_type"] == annot_type]
        if ann_sub.empty:
            continue

        seq = seq_by_acc[acc]
        emb = get_emb(acc)
        L   = emb.shape[0]
        seq = seq[:L]
        mask = build_mask(ann_sub, L)
        in_idx  = mask == 1
        out_idx = mask == 0
        if in_idx.sum() == 0:
            continue

        try:
            delta_lp = logit_delta(emb, seq, feature_id,
                                   W_enc, b_enc, W_dec, b_dec,
                                   lm_head, tokenizer)
        except Exception as exc:
            log.warning(f"    logit_delta failed for {acc}: {exc}")
            continue

        mean_in  = float(delta_lp[in_idx].mean())
        mean_out = float(delta_lp[out_idx].mean()) if out_idx.sum() > 0 else 0.0

        records.append({
            "accession":          acc,
            "mean_logitD_in":     mean_in,
            "mean_logitD_out":    mean_out,
            "contrast":           mean_in - mean_out,
            "n_annot_res":        int(in_idx.sum()),
        })

    if not records:
        return pd.DataFrame(), {}

    df = pd.DataFrame(records).sort_values("contrast", ascending=False)
    summary = {
        "logit_mean_contrast":  float(df["contrast"].mean()),
        "logit_frac_positive":  float((df["contrast"] > 0).mean()),
        "logit_mean_in":        float(df["mean_logitD_in"].mean()),
        "logit_mean_out":       float(df["mean_logitD_out"].mean()),
        "n_proteins":           len(df),
    }
    return df, summary


# =============================================================================
#  PLOTS
# =============================================================================

def plot_physicochemical(df: pd.DataFrame, feature_id: int,
                         annot_type: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── left: bar chart of cohen_d per property ──────────────────────────────
    ax = axes[0]
    df_sorted = df.sort_values("cohen_d")
    colors = ["#1D9E75" if v > 0 else "#E8593C" for v in df_sorted["cohen_d"]]
    ax.barh(df_sorted["property"], df_sorted["cohen_d"],
            color=colors, edgecolor="none")
    ax.axvline(0, color="#444441", lw=0.8)
    ax.axvline( 0.2, color="#888780", lw=0.5, linestyle="--")
    ax.axvline(-0.2, color="#888780", lw=0.5, linestyle="--")
    ax.set_xlabel("Cohen's d  (inside − outside annotation)", fontsize=10)
    ax.set_title(f"f{feature_id} × {annot_type}\nPhysicochemical contrast", fontsize=10)

    # ── right: mean_in vs mean_out per property (normalised) ─────────────────
    ax2 = axes[1]
    props = df["property"].tolist()
    x     = np.arange(len(props))
    # normalise each property to [0,1] for visual comparison
    def norm(col):
        rng = df[col].max() - df[col].min()
        return (df[col] - df[col].min()) / (rng + 1e-8)

    ax2.bar(x - 0.2, norm("mean_in"),  0.4, label="inside",
            color="#1D9E75", alpha=0.8)
    ax2.bar(x + 0.2, norm("mean_out"), 0.4, label="outside",
            color="#888780", alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(props, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("Normalised mean value", fontsize=10)
    ax2.set_title("Property profile (normalised)", fontsize=10)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(df: pd.DataFrame, feature_id: int,
                  annot_type: str, path: Path):
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # ── left: scatter mean_delta_in vs mean_delta_out ────────────────────────
    ax = axes[0]
    ax.scatter(df["mean_delta_out"], df["mean_delta_in"],
               alpha=0.5, s=20, color="#3266AD")
    lim = max(df["mean_delta_in"].max(), df["mean_delta_out"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#888780", lw=0.8, linestyle="--",
            label="delta_in = delta_out")
    ax.set_xlabel("Mean ablation Δ outside annotation", fontsize=10)
    ax.set_ylabel("Mean ablation Δ inside annotation", fontsize=10)
    ax.set_title(f"f{feature_id} × {annot_type}\nCausal ablation per protein", fontsize=10)
    ax.legend(fontsize=8)
    # fraction above diagonal
    frac = (df["mean_delta_in"] > df["mean_delta_out"]).mean()
    ax.text(0.05, 0.92, f"{frac*100:.0f}% proteins: Δin > Δout",
            transform=ax.transAxes, fontsize=9, color="#3266AD")

    # ── middle: histogram of contrast (delta_in - delta_out) ─────────────────
    ax2 = axes[1]
    ax2.hist(df["contrast"], bins=30, color="#3266AD", alpha=0.7, edgecolor="none")
    ax2.axvline(0, color="#E8593C", lw=1.5, linestyle="--", label="Δin = Δout")
    ax2.axvline(df["contrast"].mean(), color="#1D9E75", lw=1.5,
                linestyle="-", label=f"mean={df['contrast'].mean():.4f}")
    ax2.set_xlabel("Ablation contrast (Δin − Δout)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Distribution of ablation contrast", fontsize=10)
    ax2.legend(fontsize=8)

    # ── right: structural alignment bar ──────────────────────────────────────
    ax3 = axes[2]
    struct_align = df.attrs.get("structural_alignment", None)
    if struct_align is not None:
        color = "#1D9E75" if struct_align > 0.1 else ("#E8593C" if struct_align < -0.1 else "#888780")
        ax3.bar(["Struct.\nalignment"], [struct_align],
                color=color, alpha=0.85, edgecolor="none", width=0.4)
        ax3.axhline(0, color="#444441", lw=0.8)
        ax3.set_ylim(-1.05, 1.05)
        ax3.set_ylabel("cos(W_dec[:,k], v_annot)", fontsize=9)
        ax3.set_title("Decoder–annotation\ngeometric alignment", fontsize=9)
        ax3.text(0, struct_align + (0.05 if struct_align >= 0 else -0.1),
                 f"{struct_align:+.3f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=color)
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "structural_alignment\nnot available",
                 ha="center", va="center", transform=ax3.transAxes,
                 fontsize=9, color="#888780")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_logit_delta(df: pd.DataFrame, feature_id: int,
                     annot_type: str, path: Path):
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── left: scatter mean inside vs outside ─────────────────────────────
    ax = axes[0]
    ax.scatter(df["mean_logitD_out"], df["mean_logitD_in"],
               alpha=0.5, s=20, color="#BA7517")
    lim_max = max(df["mean_logitD_in"].abs().max(),
                  df["mean_logitD_out"].abs().max()) * 1.1 + 0.01
    ax.plot([-lim_max, lim_max], [-lim_max, lim_max],
            color="#888780", lw=0.8, linestyle="--", label="ΔlogP_in = ΔlogP_out")
    ax.axhline(0, color="#cccccc", lw=0.5)
    ax.axvline(0, color="#cccccc", lw=0.5)
    ax.set_xlabel("Mean ΔlogP outside annotation", fontsize=10)
    ax.set_ylabel("Mean ΔlogP inside annotation",  fontsize=10)
    ax.set_title(f"f{feature_id} × {annot_type}\nLogit delta (ESM-2 LM head)", fontsize=10)
    ax.legend(fontsize=8)
    frac = (df["mean_logitD_in"] > df["mean_logitD_out"]).mean()
    ax.text(0.05, 0.92, f"{frac*100:.0f}% proteins: ΔlogP_in > ΔlogP_out",
            transform=ax.transAxes, fontsize=9, color="#BA7517")

    # ── right: histogram of contrast ─────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(df["contrast"], bins=30, color="#BA7517", alpha=0.7, edgecolor="none")
    ax2.axvline(0, color="#E8593C", lw=1.5, linestyle="--", label="ΔlogP_in = ΔlogP_out")
    ax2.axvline(df["contrast"].mean(), color="#1D9E75", lw=1.5,
                label=f"mean={df['contrast'].mean():.4f}")
    ax2.set_xlabel("Logit-delta contrast (ΔlogP_in − ΔlogP_out)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Distribution of logit-delta contrast", fontsize=10)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_evidence_summary(pc_scores: dict, abl_summary: dict,
                          conv_df: pd.DataFrame, alignment_row: pd.Series,
                          feature_id: int, annot_type: str, path: Path,
                          logit_summary=None):
    """
    Single-page evidence summary figure — the paper figure.
    4 panels: physicochemical bar, ablation scatter, convergence bar,
              evidence scorecard.
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # ── top-left: top physicochemical properties ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    top_pc = sorted(pc_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
    names  = [x[0].replace("_", " ") for x in top_pc]
    vals   = [x[1] for x in top_pc]
    colors = ["#1D9E75" if v > 0 else "#E8593C" for v in vals]
    ax1.barh(names, vals, color=colors, edgecolor="none")
    ax1.axvline(0, color="#444441", lw=0.8)
    ax1.set_xlabel("Cohen's d", fontsize=9)
    ax1.set_title("Physicochemical signature\n(inside − outside)", fontsize=9)

    # ── top-right: ablation contrast histogram ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if abl_summary:
        ax2.bar(["Δ inside", "Δ outside"],
                [abl_summary.get("mean_contrast", 0) +
                 abl_summary.get("mean_delta_out_mean", 0),
                 abl_summary.get("mean_delta_out_mean", 0)],
                color=["#3266AD", "#888780"], edgecolor="none", alpha=0.8)
    ax2.set_ylabel("Mean ablation Δ (L2)", fontsize=9)
    ax2.set_title(f"Causal ablation\n({abl_summary.get('frac_positive',0)*100:.0f}% proteins: Δin > Δout)",
                  fontsize=9)

    # ── bottom-left: convergence (other features same annot) ─────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if not conv_df.empty:
        top_conv = conv_df.head(6)
        ax3.barh([f"f{fid}" for fid in top_conv["feature_id"]],
                 top_conv["auprc_mean"],
                 color="#BA7517", alpha=0.8, edgecolor="none")
        ax3.axvline(float(alignment_row.get("auprc_mean", 0)),
                    color="#E8593C", lw=1.5, linestyle="--",
                    label=f"f{feature_id} (focal)")
        ax3.set_xlabel("AUPRC", fontsize=9)
        ax3.set_title(f"Feature convergence\n(other features aligning with {annot_type})", fontsize=9)
        ax3.legend(fontsize=8)

    # ── bottom-right: evidence scorecard ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    scorecard = [
        ("Annotation alignment", f"AUPRC={alignment_row.get('auprc_mean',0):.3f}",
         f"OR={alignment_row.get('odds_ratio',0):.1f}",
         alignment_row.get('auprc_mean', 0) > 0.3),
        ("Physicochemical", f"top |d|={max(abs(v) for v in pc_scores.values()) if pc_scores else 0:.2f}",
         f"{sum(1 for v in pc_scores.values() if abs(v)>0.2)} props |d|>0.2",
         max(abs(v) for v in pc_scores.values()) > 0.2 if pc_scores else False),
        ("Causal ablation",
         f"mean Δcontrast={abl_summary.get('mean_contrast',0):.4f}",
         f"{abl_summary.get('frac_positive',0)*100:.0f}% proteins Δin>Δout",
         abl_summary.get("frac_positive", 0) > 0.6),
        ("Feature convergence",
         f"{len(conv_df)} other features AUPRC>0",
         f"top={conv_df['auprc_mean'].max():.3f}" if not conv_df.empty else "n/a",
         len(conv_df) >= 3),
        ("Struct. alignment",
         f"cos(decoder, v_annot)={abl_summary.get('structural_alignment', float('nan')):.3f}",
         "> 0.1 = geometrically consistent",
         abl_summary.get("structural_alignment", 0.0) > 0.1),
    ]

    if logit_summary:
        scorecard.append(
            ("Logit delta (ESM)",
             f"mean Δ contrast={logit_summary.get('logit_mean_contrast', float('nan')):.4f}",
             f"{logit_summary.get('logit_frac_positive', 0)*100:.0f}% proteins ΔlogP_in>out",
             logit_summary.get("logit_frac_positive", 0) > 0.6),
        )

    y = 0.88
    ax4.text(0.05, 0.95, "Evidence scorecard", fontsize=10,
             fontweight="bold", transform=ax4.transAxes, va="top")
    for name, val1, val2, passed in scorecard:
        color  = "#1D9E75" if passed else "#E8593C"
        symbol = "✓" if passed else "✗"
        ax4.text(0.05, y, f"{symbol} {name}", fontsize=9,
                 color=color, transform=ax4.transAxes)
        ax4.text(0.05, y - 0.07, f"   {val1}  |  {val2}", fontsize=8,
                 color="#888780", transform=ax4.transAxes)
        y -= 0.18

    fig.suptitle(f"f{feature_id} × {annot_type} — Multi-evidence summary",
                 fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

    # ── load SAE ─────────────────────────────────────────────────────────────
    log.info("Loading SAE ...")
    W_enc, b_enc, W_dec, b_dec, K, D = load_sae(args.checkpoint, device)

    esm_lm_head = None
    esm_tokenizer = None
    if args.esm_model:
        esm_lm_head, esm_tokenizer = load_esm_for_logits(
            args.esm_model, device, args.esm_cache_dir
        )

    # ── load proteins + sequences ─────────────────────────────────────────────
    log.info("Loading sequences ...")
    prot_df = pd.read_csv(args.proteins, sep="\t", low_memory=False)
    prot_df = prot_df[prot_df["split"] == args.split]
    seq_by_acc = dict(zip(prot_df["accession"], prot_df["sequence"]))
    log.info(f"  {len(seq_by_acc):,} sequences loaded")

    # ── load annotations ──────────────────────────────────────────────────────
    log.info("Loading annotations ...")
    ann_df = pd.read_csv(args.annotations, sep="\t", low_memory=False)
    ann_df.columns = ann_df.columns.str.strip().str.lower()
    ann_df = ann_df[ann_df["split"] == args.split].copy()
    ann_df["start"] = ann_df["start"].astype(int)
    ann_df["end"]   = ann_df["end"].astype(int)

    # use annot_subtype as feature_type when available
    # (annotations_enriched.tsv has specific subtypes like "Binding site: GTP")
    if ("annot_subtype" in ann_df.columns
            and ann_df["annot_subtype"].notna().any()
            and ann_df["annot_subtype"].nunique() > ann_df["feature_type"].nunique()):
        ann_df["feature_type"] = ann_df["annot_subtype"]
        log.info("  Using annot_subtype as feature_type column "
                 f"({ann_df['feature_type'].nunique()} subtypes)")

    ann_by_acc = {acc: grp for acc, grp in ann_df.groupby("accession")}

    # ── load alignment results ────────────────────────────────────────────────
    log.info("Loading alignment results ...")
    align_df = pd.read_csv(args.alignment, sep="\t")

    align_df["precision"] = align_df["tp"] / (align_df["tp"] + align_df["fp"] + 1e-9)

    # ── regime classification (dynamic, data-driven) ──────────────────────────
    align_df["residues_per_protein"] = (
        align_df["tp"] / (align_df["n_proteins"] + 1e-9)
    )

    # threshold = mediana da distribuicao de residuos por proteina
    if args.rpp_threshold is not None:
        rpp_threshold = args.rpp_threshold
        log.info(f"  Using manual threshold: {rpp_threshold:.1f}")
    else:
        rpp = align_df["residues_per_protein"].values.reshape(-1, 1)
        log_rpp = np.log1p(rpp)  # log porque a distribuicao eh skewed

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(log_rpp)

        # threshold = ponto de intersecao entre as duas gaussianas
        # aproximado pela media das duas medias
        means = np.expm1(gmm.means_.flatten())
        rpp_threshold = means.mean()

        log.info(f"  GMM means: {means}, threshold: {rpp_threshold:.1f} residues/protein")

    align_df["regime"] = (
        align_df["residues_per_protein"]
        .apply(lambda x: "focal" if x < rpp_threshold else "domain")
    )

    n_focal  = (align_df["regime"] == "focal").sum()
    n_domain = (align_df["regime"] == "domain").sum()
    log.info(f"  Focal pairs: {n_focal}, Domain pairs: {n_domain}")

    # ── filtros por regime ─────────────────────────────────────────────────────
    align_df["precision"] = (
        align_df["tp"] / (align_df["tp"] + align_df["fp"] + 1e-9)
    )

    # filtro de precision dinamico: remove bottom 10% de precision por regime
    focal_prec_thresh  = align_df.loc[align_df["regime"]=="focal",  "precision"].quantile(0.10)
    domain_prec_thresh = align_df.loc[align_df["regime"]=="domain", "precision"].quantile(0.10)

    focal_df = align_df[
        (align_df["regime"] == "focal") &
        (align_df["precision"] >= focal_prec_thresh) &
        (align_df["n_clusters"] >= args.min_clusters_focal)
    ].sort_values(["auprc_mean", "n_clusters"], ascending=False)

    domain_df = align_df[
        (align_df["regime"] == "domain") &
        (align_df["precision"] >= domain_prec_thresh) &
        (align_df["n_clusters"] >= args.min_clusters_domain)
    ].sort_values(["auprc_mean", "n_clusters"], ascending=False)

    # ── seleciona top pairs balanceados ───────────────────────────────────────
    n_each      = args.top_pairs // 2
    top_focal   = list(zip(focal_df["feature_id"].astype(int),
                        focal_df["annot_type"]))[:n_each]
    top_domain  = list(zip(domain_df["feature_id"].astype(int),
                        domain_df["annot_type"]))[:n_each]
    pairs = top_domain + top_focal

    log.info(f"  Selected {len(top_domain)} domain + {len(top_focal)} focal pairs")
    log.info(f"  Precision thresholds: focal>={focal_prec_thresh:.3f}, "
            f"domain>={domain_prec_thresh:.3f}")

    all_scores_df = None
    if args.all_scores.exists():
        all_scores_df = pd.read_parquet(args.all_scores)

    # ── load ESM-2 for on-the-fly embeddings ─────────────────────────────────
    log.info("Loading ESM-2 ...")
    esm_model, converter, backend = load_esm2(args.esm2_model, device)
    get_emb = lambda acc: _get_embedding_live(
        acc, seq_by_acc[acc], esm_model, converter, backend,
        device, args.esm2_layer, args.max_seq_len,
    )

    # ── analyse each pair ─────────────────────────────────────────────────────
    all_evidence = []

    for feature_id, annot_type in pairs:
        log.info(f"\n{'─'*54}")
        log.info(f"  f{feature_id} × {annot_type}")

        safe   = annot_type.lower().replace(" ", "_").replace("/", "_")
        pdir   = args.outdir / f"f{feature_id}_{safe}"
        pdir.mkdir(exist_ok=True)

        # proteins with this annotation
        ann_sub = ann_df[ann_df["feature_type"] == annot_type]
        accs    = [a for a in ann_sub["accession"].unique()
                   if a in seq_by_acc][:args.top_proteins]
        log.info(f"  {len(accs)} proteins")

        # alignment row for this pair
        align_row = align_df[
            (align_df["feature_id"].astype(int) == feature_id) &
            (align_df["annot_type"] == annot_type)
        ]
        align_row = align_row.iloc[0] if not align_row.empty else pd.Series()

        # ── layer 1: physicochemical ──────────────────────────────────────────
        log.info("  Layer 1: physicochemical ...")
        pc_df, pc_scores = physicochemical_evidence(
            accs, annot_type, seq_by_acc, ann_by_acc,
            args.activation_threshold, feature_id, get_emb,
            W_enc, b_enc,
        )
        pc_df.to_csv(pdir / "physicochemical.tsv", sep="\t", index=False)
        if not pc_df.empty:
            plot_physicochemical(pc_df, feature_id, annot_type,
                                 pdir / "physicochemical.png")
        log.info(f"    top property: "
                 f"{pc_df.iloc[0]['property'] if not pc_df.empty else 'n/a'} "
                 f"d={pc_df.iloc[0]['cohen_d']:.3f}" if not pc_df.empty else "")

        # ── layer 2: causal ablation ──────────────────────────────────────────
        log.info("  Layer 2: causal ablation ...")
        abl_df, abl_summary = ablation_evidence(
            accs, annot_type, ann_by_acc, get_emb,
            feature_id, W_enc, b_enc, W_dec, b_dec,
        )
        if not abl_df.empty:
            abl_df.to_csv(pdir / "ablation.tsv", sep="\t", index=False)
            # store mean_delta_out_mean for plotting
            abl_summary["mean_delta_out_mean"] = float(abl_df["mean_delta_out"].mean())
            abl_df.attrs["structural_alignment"] = abl_summary.get("structural_alignment")
            plot_ablation(abl_df, feature_id, annot_type,
                          pdir / "ablation.png")
            log.info(f"    ablation contrast: {abl_summary['mean_contrast']:.4f}  "
                     f"frac_pos: {abl_summary['frac_positive']:.2f}")

        # ── layer 2b: logit-delta evidence (ESM downstream) ──────────────
        logit_summary = {}
        logit_df = pd.DataFrame()
        if esm_lm_head is not None:
            log.info("  Layer 2b: logit-delta evidence ...")
            logit_df, logit_summary = logit_delta_evidence(
                accs, annot_type, ann_by_acc, get_emb,
                seq_by_acc, feature_id,
                W_enc, b_enc, W_dec, b_dec,
                esm_lm_head, esm_tokenizer,
            )
            if not logit_df.empty:
                logit_df.to_csv(pdir / "logit_delta.tsv", sep="\t", index=False)
                plot_logit_delta(logit_df, feature_id, annot_type,
                                 pdir / "logit_delta.png")
                log.info(f"    logit contrast: {logit_summary['logit_mean_contrast']:.4f}  "
                         f"frac_pos: {logit_summary['logit_frac_positive']:.2f}")

        # ── layer 3: convergence ──────────────────────────────────────────────
        conv_df = pd.DataFrame()
        if all_scores_df is not None:
            conv_df = convergence_evidence(all_scores_df, feature_id, annot_type)
            conv_df.to_csv(pdir / "convergence.tsv", sep="\t", index=False)
            log.info(f"    {len(conv_df)} other features align with {annot_type}")

        # ── summary figure ────────────────────────────────────────────────────
        plot_evidence_summary(
            pc_scores, abl_summary, conv_df, align_row,
            feature_id, annot_type, pdir / "evidence_summary.png",
            logit_summary=logit_summary,
        )

        # ── evidence record ───────────────────────────────────────────────────
        ev = {
            "feature_id":          feature_id,
            "annot_type":          annot_type,
            "auprc_mean":          float(align_row.get("auprc_mean", float("nan"))),
            "odds_ratio":          float(align_row.get("odds_ratio", float("nan"))),
            "fisher_p":            float(align_row.get("fisher_p", float("nan"))),
            "n_proteins":          len(accs),
            # physicochemical
            "pc_top_property":     pc_df.iloc[0]["property"] if not pc_df.empty else "",
            "pc_top_cohen_d":      float(pc_df.iloc[0]["cohen_d"]) if not pc_df.empty else float("nan"),
            "pc_n_strong":         int((pc_df["cohen_d"].abs() > 0.2).sum()) if not pc_df.empty else 0,
            # ablation
            "abl_mean_contrast":   abl_summary.get("mean_contrast", float("nan")),
            "abl_frac_positive":   abl_summary.get("frac_positive", float("nan")),
            "abl_mean_cohen_d":    abl_summary.get("mean_cohen_d", float("nan")),
            # structural alignment
            "abl_structural_align":  abl_summary.get("structural_alignment", float("nan")),
            # logit delta
            "logit_mean_contrast":   logit_summary.get("logit_mean_contrast", float("nan")),
            "logit_frac_positive":   logit_summary.get("logit_frac_positive", float("nan")),
            # convergence
            "conv_n_features":     len(conv_df),
            "conv_top_auprc":      float(conv_df["auprc_mean"].max()) if not conv_df.empty else float("nan"),
        }
        with open(pdir / "evidence_summary.json", "w") as f:
            json.dump(ev, f, indent=2)
        all_evidence.append(ev)

    # ── full evidence matrix ──────────────────────────────────────────────────
    if all_evidence:
        ev_df = pd.DataFrame(all_evidence)
        ev_path = args.outdir / "full_evidence_matrix.tsv"
        ev_df.to_csv(ev_path, sep="\t", index=False)
        log.info(f"\n{'═'*54}")
        log.info("  MULTI-EVIDENCE MATRIX")
        log.info(f"{'═'*54}")
        print(ev_df[[
            "feature_id","annot_type",
            "auprc_mean","odds_ratio",
            "pc_top_property","pc_top_cohen_d","pc_n_strong",
            "abl_mean_contrast","abl_frac_positive",
            "abl_structural_align",
            "logit_frac_positive",
            "conv_n_features",
        ]].to_string(index=False))
        log.info(f"\n✅ Done. Outputs in {args.outdir}")


if __name__ == "__main__":
    main()
