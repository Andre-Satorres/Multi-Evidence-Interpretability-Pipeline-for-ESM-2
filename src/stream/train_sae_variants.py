"""
train_sae_variants.py — PIPLV2
================================
Train SAE variants (Vanilla/L1, TopK, Matryoshka, T-SAE) in streaming mode.
No pre-computed embeddings needed — runs ESM-2 on-the-fly.

Memory/disk design:
  - Zero residue block storage (ESM-2 runs live)
  - Checkpoint only best.pt + last.pt (no per-epoch checkpoints)
  - Activations never cached to disk during training
  - RAM: ESM-2 (~2.5GB) + SAE (~80MB) + one batch buffer (~50MB)
  - Checkpoints: ~80MB each (only 2 kept per run)

Architectures
-------------
  vanilla   : L1 SAE (original) — ReLU encoder + L1 sparsity penalty
  topk      : TopK SAE — hard top-k sparsity, no L1 penalty
  matryoshka: Matryoshka SAE — multi-scale loss at K/8, K/4, K/2, K
  tsae      : Temporal SAE — L1 + contrastive loss on adjacent residues

Output format
-------------
All variants save best.pt in the SAME format as train_sae.py:
  {model_state_dict, optimizer_state_dict, epoch, config, best_val_loss}
This makes them drop-in compatible with feature_alignment.py and paper_figures.py.

Optimizations applied
---------------------
  - torch.compile (PyTorch 2.x):    ~10-30% throughput gain via kernel fusion
  - torch.autocast (bfloat16):      ~2x throughput on tensor cores, ~50% VRAM
  - GradScaler:                     numerically stable mixed-precision training
  - non_blocking=True transfers:    overlaps H2D copy with GPU compute
  - pin_memory tensors:             faster CPU→GPU DMA for batch buffers

Usage
-----
  # quick test (5k proteins, 2 epochs):
  python src/sae/train_sae_variants.py \
      --architecture vanilla \
      --esm2-model   esm2_t33_650M_UR50D \
      --fasta        data/proteins.fasta \
      --proteins     data/proteins_with_split.tsv \
      --max-proteins 5000 \
      --num-epochs   2 \
      --outdir       outputs/sae_variants/vanilla_5k

  # full run (50k proteins, 13 epochs):
  python src/sae/train_sae_variants.py \
      --architecture topk \
      --esm2-model   esm2_t33_650M_UR50D \
      --fasta        data/proteins.fasta \
      --proteins     data/proteins_with_split.tsv \
      --max-proteins 50000 \
      --num-epochs   13 \
      --outdir       outputs/sae_variants/topk_50k

  # T-SAE (needs sequential order — do NOT set --shuffle-proteins):
  python src/sae/train_sae_variants.py \
      --architecture tsae \
      --tsae-lambda  0.1 \
      --esm2-model   esm2_t33_650M_UR50D \
      --fasta        data/proteins.fasta \
      --proteins     data/proteins_with_split.tsv \
      --max-proteins 50000 \
      --num-epochs   13 \
      --outdir       outputs/sae_variants/tsae_50k

  # disable optimizations if you hit issues:
  python src/sae/train_sae_variants.py \
      --no-compile \
      --no-amp \
      ...
"""

import sys
import csv
import json
import time
import logging
import argparse
import random
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from constants import OUT_DIR

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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train SAE variants (Vanilla/TopK/Matryoshka/T-SAE) via ESM-2 streaming.",
    )
    # architecture
    p.add_argument("--architecture", required=True,
                   choices=["vanilla", "topk", "matryoshka", "tsae"],
                   help="SAE architecture to train.")
    p.add_argument("--latent-dim",   type=int,   default=8192)
    p.add_argument("--l1-lambda",    type=float, default=3e-5,
                   help="L1 penalty (vanilla/tsae only).")
    p.add_argument("--topk-k",       type=int,   default=128,
                   help="Number of active features per token (topk only).")
    p.add_argument("--topk-k-start", type=int,   default=0,
                   help="Initial k for TopK annealing (0 = no annealing, use --topk-k directly). "
                        "k is linearly decayed from --topk-k-start to --topk-k over "
                        "--l1-warmup-epochs, then held fixed. "
                        "Recommended: --topk-k-start 512 to let the dictionary form before "
                        "sparsity is enforced.")
    p.add_argument("--auxk-alpha",   type=float, default=1/32,
                   help="Weight of AuxK dead-feature loss (topk only). "
                        "Reconstructs the residual using top-k activations among dead "
                        "features so they receive gradient. 0 = disabled.")
    p.add_argument("--tsae-lambda",  type=float, default=0.1,
                   help="Weight of temporal contrastive loss (tsae only).")
    p.add_argument("--tsae-temp",    type=float, default=0.07,
                   help="Temperature for T-SAE contrastive loss.")
    p.add_argument("--matryoshka-scales", type=int, nargs="+",
                   default=None,
                   help="Latent scales for Matryoshka (default: K//8, K//4, K//2, K).")
    p.add_argument("--l1-warmup-epochs", type=int, default=2)
    p.add_argument("--target-l0",        type=int, default=0,
                   help="Target mean L0 per token for L1-based archs (vanilla/tsae/matryoshka). "
                        "When set, lambda is adjusted after each epoch to track this value. "
                        "Set equal to --topk-k for a fair cross-arch comparison. 0 = disabled.")

    # data
    p.add_argument("--esm2-model",  default="esm2_t33_650M_UR50D")
    p.add_argument("--esm2-layer",  type=int, default=33)
    p.add_argument("--fasta",       type=Path, required=True)
    p.add_argument("--proteins",    type=Path, required=True,
                   help="TSV with accession + split columns.")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split",   default="val")
    p.add_argument("--max-proteins", type=int, default=None,
                   help="Limit proteins per split (for quick test runs).")
    p.add_argument("--max-seq-len",  type=int, default=1022,
                   help="Truncate sequences longer than this.")
    p.add_argument("--shuffle-proteins", action="store_true",
                   help="Shuffle protein order each epoch. "
                        "Do NOT use with --architecture tsae.")

    # training
    p.add_argument("--num-epochs",   type=int,   default=13)
    p.add_argument("--batch-size",   type=int,   default=2048,
                   help="Residues per gradient step.")
    p.add_argument("--learning-rate",type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)

    # checkpointing — keep small
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Save numbered checkpoint every N epochs (0=disabled). "
                        "Each checkpoint is ~80MB — use sparingly.")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop if val_recon does not improve for N epochs. "
                        "0 = disabled.")

    # ── optimization flags ───────────────────────────────────────────────────
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile (useful if PyTorch < 2.0 or debugging).")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable automatic mixed precision (bfloat16). "
                        "Fall back to full fp32.")
    p.add_argument("--amp-dtype", default="bfloat16",
                   choices=["bfloat16", "float16"],
                   help="AMP dtype. bfloat16 is preferred on Ampere/Ada (A5500). "
                        "Use float16 on older Volta/Turing GPUs.")

    # output
    p.add_argument("--outdir",  type=Path,
                   default=OUT_DIR / "sae_variants" / "run")
    p.add_argument("--device",  default="auto")
    return p.parse_args()


# =============================================================================
#  ESM-2
# =============================================================================

def load_esm2(model_name, device):
    log.info(f"Loading ESM-2 ({model_name}) ...")
    try:
        import esm as _esm
        model, alphabet = _esm.pretrained.load_model_and_alphabet(model_name)
        batch_converter  = alphabet.get_batch_converter()
        model.eval().to(device)
        log.info("  Loaded via fair-esm")
        return model, batch_converter, "esm"
    except Exception:
        log.info("  fair-esm not found — trying HuggingFace transformers ...")
        from transformers import EsmModel, EsmTokenizer
        tokenizer = EsmTokenizer.from_pretrained(f"facebook/{model_name}")
        model     = EsmModel.from_pretrained(f"facebook/{model_name}")
        model.eval().to(device)
        log.info("  Loaded via HuggingFace")
        return model, tokenizer, "hf"


@torch.no_grad()
def get_embedding(acc, seq, esm_model, converter, backend, device, layer, max_len):
    """Returns [L, D] float32 numpy array. Never stores to disk."""
    seq = seq[:max_len]
    L   = len(seq)
    if backend == "esm":
        _, _, tokens = converter([(acc, seq)])
        tokens = tokens.to(device)
        out = esm_model(tokens, repr_layers=[layer], return_contacts=False)
        emb = out["representations"][layer][0, 1:L+1, :].cpu().float().numpy()
    else:
        inputs = converter(seq, return_tensors="pt", truncation=True,
                           max_length=max_len+2).to(device)
        out = esm_model(**inputs, output_hidden_states=True)
        # hidden_states[0] = embedding layer; hidden_states[layer] = transformer layer `layer`
        emb = out.hidden_states[layer][0, 1:L+1, :].cpu().float().numpy()
    return emb  # [L, D]


# =============================================================================
#  FASTA + PROTEIN LIST
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
                acc = header.split("|")[1] if "|" in header else header
                buf = []
            else:
                buf.append(line)
    if acc:
        seqs[acc] = "".join(buf)
    log.info(f"  Read {len(seqs):,} sequences")
    return seqs


def load_split_accs(proteins_path, split, seqs, max_proteins=None, seed=42):
    import pandas as pd
    df = pd.read_csv(proteins_path, sep="\t", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    accs = df[df["split"] == split]["accession"].tolist()
    accs = [a for a in accs if a in seqs]
    if max_proteins and len(accs) > max_proteins:
        rng = random.Random(seed)
        accs = rng.sample(accs, max_proteins)
    log.info(f"  {split}: {len(accs):,} proteins")
    return accs


# =============================================================================
#  INPUT NORMALIZATION
# =============================================================================

def estimate_normalization_stats(accs, seqs, esm_model, converter,
                                  backend, device, layer, max_len,
                                  n_proteins=500, seed=42):
    """
    Estimates per-dimension mean and std of ESM-2 embeddings from a small sample.
    Normalizing input to mean=0, std=1 makes the MSE and L1 penalty comparable
    in scale, so l1_lambda values in [1e-4, 1e-2] work as expected.

    Returns (mean, std) as float32 CPU tensors of shape [D].
    Saved to config.json and checkpoint so feature_alignment.py can apply
    the same normalization at inference time.
    """
    log.info(f"Estimating normalization stats from {n_proteins} proteins ...")
    rng    = random.Random(seed)
    sample = rng.sample(accs, min(n_proteins, len(accs)))

    embs = []
    for acc in sample:
        try:
            emb = get_embedding(acc, seqs[acc], esm_model, converter,
                                backend, device, layer, max_len)
            embs.append(emb)
        except Exception:
            continue

    all_embs = np.concatenate(embs, axis=0)          # [N_residues, D]
    mean = torch.tensor(all_embs.mean(axis=0), dtype=torch.float32)
    std  = torch.tensor(all_embs.std(axis=0),  dtype=torch.float32).clamp(min=1e-6)
    log.info(f"  Stats from {len(embs)} proteins, {all_embs.shape[0]:,} residues")
    log.info(f"  mean norm={mean.norm():.2f}  std mean={std.mean():.4f}  "
             f"std min={std.min():.4f}  std max={std.max():.4f}")
    return mean, std


# =============================================================================
#  SAE MODELS
# =============================================================================

def make_vanilla_sae(input_dim, latent_dim):
    """Standard L1 SAE: ReLU encoder + MSE + L1 penalty.

    Follows the Bricken et al. / Anthropic SAE recipe:
      - Decoder columns kept at unit norm (prevents L1-laundering via scale).
      - Encoder weight initialised as W_dec.T (tied-transpose init).
      - L1 penalty is sum over features per token, mean over tokens — this
        makes lambda scale-independent of latent_dim.
    """
    class VanillaSAE(nn.Module):
        arch = "vanilla"
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
            self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
            # decoder columns → unit norm; encoder init tied to decoder
            nn.init.xavier_uniform_(self.decoder.weight)
            with torch.no_grad():
                self.decoder.weight.copy_(
                    F.normalize(self.decoder.weight, dim=0))
            nn.init.zeros_(self.decoder.bias)
            # tied-transpose: encoder rows ≈ decoder columns (good geometry)
            with torch.no_grad():
                self.encoder.weight.copy_(self.decoder.weight.T)
            nn.init.zeros_(self.encoder.bias)

        def encode(self, x):
            return F.relu(self.encoder(x))

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            z = self.encode(x)
            return self.decoder(z), z

        def normalize_decoder(self):
            """Re-normalize decoder columns to unit norm after each gradient step."""
            with torch.no_grad():
                self.decoder.weight.copy_(
                    F.normalize(self.decoder.weight, dim=0))

    return VanillaSAE()


def make_topk_sae(input_dim, latent_dim, k, auxk_alpha=1/32):
    """TopK SAE: hard top-k sparsity with AuxK dead-feature loss.

    Decoder columns kept at unit norm — critical so that feature activation
    magnitudes are comparable across the dictionary.

    AuxK loss (Gao et al. 2024): reconstructs the main-path residual using
    the top-k activations among *dead* features (those not in the main top-k).
    This gives dead features gradient signal without distorting the primary
    reconstruction objective.  Controlled by auxk_alpha (default 1/32).
    """
    _k = k
    _auxk_alpha = auxk_alpha
    class TopKSAE(nn.Module):
        arch = "topk"
        def __init__(self):
            super().__init__()
            self.k = _k
            self.auxk_alpha = _auxk_alpha
            self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
            self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
            nn.init.xavier_uniform_(self.decoder.weight)
            with torch.no_grad():
                self.decoder.weight.copy_(
                    F.normalize(self.decoder.weight, dim=0))
            nn.init.zeros_(self.decoder.bias)
            with torch.no_grad():
                self.encoder.weight.copy_(self.decoder.weight.T)
            nn.init.zeros_(self.encoder.bias)
            # z_pre is stored during forward for auxk_loss computation
            self._z_pre = None

        def encode(self, x):
            z_pre = F.relu(self.encoder(x))           # [B, K]
            self._z_pre = z_pre                       # kept for auxk_loss
            # keep only top-k activations, zero the rest
            topk_vals, topk_idx = z_pre.topk(self.k, dim=-1)
            z = torch.zeros_like(z_pre)
            z.scatter_(-1, topk_idx, topk_vals)
            return z

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            z = self.encode(x)
            return self.decoder(z), z

        def auxk_loss(self, x, x_hat, z):
            """AuxK loss: reconstruct the residual using dead features.

            Dead features = those NOT in the main top-k for each sample.
            We take the top-k of z_pre among dead features, decode that,
            and compute MSE against the main-path residual.
            Returns 0 if auxk_alpha == 0 or z_pre unavailable.
            """
            if self.auxk_alpha == 0 or self._z_pre is None:
                return torch.tensor(0.0, device=x.device)

            z_pre = self._z_pre                       # [B, K]
            residual = x - x_hat                       # [B, D]

            # mask out main top-k features: set them to -inf in z_pre
            # so they can't be selected by the aux top-k
            dead_pre = z_pre.clone()
            dead_pre[z > 0] = 0.0                     # zero where main path is active

            # pick top-k among the remaining (dead) features
            k_aux = min(self.k, (dead_pre > 0).sum(dim=-1).min().item())
            if k_aux == 0:
                return torch.tensor(0.0, device=x.device)

            aux_topk_vals, aux_topk_idx = dead_pre.topk(k_aux, dim=-1)
            z_aux = torch.zeros_like(z_pre)
            z_aux.scatter_(-1, aux_topk_idx, aux_topk_vals)

            # decode the dead-feature activations and compute MSE vs residual
            x_aux = self.decoder(z_aux)
            return F.mse_loss(x_aux, residual)

        def normalize_decoder(self):
            with torch.no_grad():
                self.decoder.weight.copy_(
                    F.normalize(self.decoder.weight, dim=0))

    return TopKSAE()


def make_matryoshka_sae(input_dim, latent_dim, scales=None):
    """
    Matryoshka SAE: trains nested sub-dictionaries at multiple scales.
    Loss = sum of weighted MSE at each scale (K//8, K//4, K//2, K).
    Uses only the first S latents for scale S — nested structure.
    """
    if scales is None:
        scales = [latent_dim//8, latent_dim//4, latent_dim//2, latent_dim]
    scales = [s for s in scales if s <= latent_dim]
    _scales = scales

    class MatryoshkaSAE(nn.Module):
        arch = "matryoshka"
        def __init__(self):
            super().__init__()
            self.scales  = _scales
            # weights: smaller scales get higher weight (coarse features matter more)
            raw_w = [1.0 / (i+1) for i in range(len(_scales))]
            total = sum(raw_w)
            self.scale_weights = [w / total for w in raw_w]

            self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
            self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
            nn.init.xavier_uniform_(self.decoder.weight)
            with torch.no_grad():
                self.decoder.weight.copy_(
                    F.normalize(self.decoder.weight, dim=0))
            nn.init.zeros_(self.decoder.bias)
            with torch.no_grad():
                self.encoder.weight.copy_(self.decoder.weight.T)
            nn.init.zeros_(self.encoder.bias)

        def encode(self, x):
            return F.relu(self.encoder(x))

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            z = self.encode(x)
            return self.decoder(z), z

        def normalize_decoder(self):
            with torch.no_grad():
                self.decoder.weight.copy_(
                    F.normalize(self.decoder.weight, dim=0))

        def matryoshka_loss(self, x, l1_lambda=0.0):
            """Compute weighted multi-scale reconstruction loss."""
            z_full = self.encode(x)                    # [B, K]
            total_loss = torch.tensor(0.0, device=x.device)
            for scale, w in zip(self.scales, self.scale_weights):
                z_s    = z_full.clone()
                z_s[:, scale:] = 0.0                   # zero out latents > scale
                x_hat_s = self.decoder(z_s)
                recon_s = F.mse_loss(x_hat_s, x)
                # L1: sum per token, mean over batch (scale-independent of latent_dim)
                l1_s = z_s[:, :scale].abs().sum(dim=-1).mean() if l1_lambda > 0 else 0.0
                total_loss = total_loss + w * (recon_s + l1_lambda * l1_s)
            return total_loss, z_full

    return MatryoshkaSAE()


def make_tsae(input_dim, latent_dim, tsae_lambda, temperature):
    """
    Temporal SAE: L1 SAE + contrastive loss encouraging adjacent residues
    to have similar high-level feature activations.

    Contrastive loss: for each pair (z_i, z_{i+1}) of adjacent residues,
    maximise cosine similarity of their normalised activations.
    This is a self-supervised objective — no labels needed.

    The key insight from the T-SAE paper: semantic content evolves smoothly
    over adjacent tokens (residues). A feature that fires on one residue of
    an alpha helix should also fire on the neighbouring helix residues.
    """
    _lam  = tsae_lambda
    _temp = temperature

    class TSAE(nn.Module):
        arch = "tsae"
        def __init__(self):
            super().__init__()
            self.tsae_lambda = _lam
            self.temperature = _temp
            self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
            self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
            nn.init.xavier_uniform_(self.decoder.weight)
            with torch.no_grad():
                self.decoder.weight.copy_(
                    F.normalize(self.decoder.weight, dim=0))
            nn.init.zeros_(self.decoder.bias)
            with torch.no_grad():
                self.encoder.weight.copy_(self.decoder.weight.T)
            nn.init.zeros_(self.encoder.bias)

        def encode(self, x):
            return F.relu(self.encoder(x))

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            z = self.encode(x)
            return self.decoder(z), z

        def normalize_decoder(self):
            with torch.no_grad():
                self.decoder.weight.copy_(
                    F.normalize(self.decoder.weight, dim=0))

        def temporal_contrastive_loss(self, z, protein_lengths):
            """
            Contrastive loss on adjacent residues within each protein.

            z               : [B_total, K] activations (concatenated proteins)
            protein_lengths : list of int — length of each protein in the batch
                              sum(protein_lengths) == B_total

            For each protein, computes cosine similarity between adjacent
            residue pairs and maximises it (NT-Xent style).

            Returns scalar loss tensor.
            """
            if len(protein_lengths) == 0 or z.shape[0] < 2:
                return torch.tensor(0.0, device=z.device)

            losses = []
            offset = 0
            for L in protein_lengths:
                if L < 2:
                    offset += L
                    continue
                z_prot = z[offset:offset+L]             # [L, K]

                # normalise for cosine similarity
                z_norm = F.normalize(z_prot, dim=-1)    # [L, K]

                # anchor = all residues except last, positive = next residue
                z_anc = z_norm[:-1]                     # [L-1, K]
                z_pos = z_norm[1:]                      # [L-1, K]

                # cosine similarity matrix: [L-1, L-1]
                sim = torch.mm(z_anc, z_pos.T) / self.temperature

                # NT-Xent: diagonal = positive pairs
                labels = torch.arange(L-1, device=z.device)
                loss_i = F.cross_entropy(sim, labels)
                losses.append(loss_i)
                offset += L

            if not losses:
                return torch.tensor(0.0, device=z.device)
            return torch.stack(losses).mean()

    return TSAE()


# =============================================================================
#  STREAMING BATCH BUILDER
# =============================================================================

class ProteinBatchAccumulator:
    """
    Accumulates residue embeddings from multiple proteins until
    batch_size is reached, then yields the batch.

    For T-SAE: also tracks protein_lengths for the contrastive loss.
    Critically: never stores to disk, uses ~50MB RAM per batch max.

    Optimization: tensors allocated with pin_memory=True for faster
    CPU→GPU DMA, and transferred with non_blocking=True to overlap
    H2D copy with GPU compute on the previous batch.
    """
    def __init__(self, batch_size, device, arch):
        self.batch_size = batch_size
        self.device     = device
        self.arch       = arch
        self._buf       = []          # list of [L_i, D] tensors (pinned)
        self._lengths   = []          # protein lengths (for T-SAE)
        self._n         = 0           # current residue count

    def add(self, emb_np):
        """Add one protein's embeddings ([L, D] numpy float32)."""
        # pin_memory=True: allocates in page-locked memory for faster DMA
        t = torch.from_numpy(emb_np).pin_memory()
        self._buf.append(t)
        self._lengths.append(t.shape[0])
        self._n += t.shape[0]

    def ready(self):
        return self._n >= self.batch_size

    def _to_device(self, t):
        """Transfer to device with non_blocking for H2D overlap."""
        return t.to(self.device, non_blocking=True)

    def flush(self):
        """
        Returns (x, protein_lengths) where x is [B, D] on device.
        Keeps remainder in buffer. protein_lengths only set for tsae.
        """
        x_all = torch.cat(self._buf, dim=0)            # [N_total, D] — still CPU/pinned
        lengths_all = self._lengths

        if self._n <= self.batch_size:
            # flush everything
            x = self._to_device(x_all)
            self._buf = []; self._lengths = []; self._n = 0
            return x, lengths_all

        # split at batch_size — for T-SAE we need to split on protein boundaries
        if self.arch == "tsae":
            # find the protein boundary closest to batch_size without going over
            cum = 0
            split_prot = 0
            for i, L in enumerate(lengths_all):
                if cum + L > self.batch_size:
                    break
                cum += L
                split_prot = i + 1

            if split_prot == 0:
                # single protein longer than batch_size — yield it whole
                split_prot = 1
                cum = lengths_all[0]

            x_batch    = self._to_device(torch.cat(self._buf[:split_prot], dim=0))
            len_batch  = lengths_all[:split_prot]
            # keep remainder
            self._buf      = self._buf[split_prot:]
            self._lengths  = lengths_all[split_prot:]
            self._n       -= cum
            return x_batch, len_batch
        else:
            # simple split at batch_size — order doesn't matter for non-T-SAE
            x_batch   = self._to_device(x_all[:self.batch_size])
            remainder = x_all[self.batch_size:]
            self._buf     = [remainder] if remainder.shape[0] > 0 else []
            self._lengths = []   # not tracked for non-T-SAE
            self._n       = remainder.shape[0]
            return x_batch, []

    def flush_all(self):
        """Flush whatever is left (end of epoch)."""
        if self._n == 0:
            return None, []
        x = self._to_device(torch.cat(self._buf, dim=0))
        lengths = self._lengths
        self._buf = []; self._lengths = []; self._n = 0
        return x, lengths


# =============================================================================
#  LOSS FUNCTIONS
# =============================================================================

def compute_loss(model, x, arch, l1_lambda, protein_lengths=None):
    """Unified loss computation for all architectures."""
    if arch == "matryoshka":
        loss, z = model.matryoshka_loss(x, l1_lambda)
        x_hat   = model.decode(z)
    elif arch == "tsae":
        x_hat, z = model(x)
        recon    = F.mse_loss(x_hat, x)
        l1       = z.abs().sum(dim=-1).mean()   # sum over features per token, mean over tokens
        temporal = model.temporal_contrastive_loss(z, protein_lengths or [])
        loss     = recon + l1_lambda * l1 + model.tsae_lambda * temporal
    elif arch == "topk":
        x_hat, z = model(x)
        recon    = F.mse_loss(x_hat, x)   # no L1 — sparsity via hard top-k
        raw = getattr(model, "_orig_mod", model)
        aux  = raw.auxk_loss(x, x_hat, z) if hasattr(raw, "auxk_loss") else 0.0
        loss = recon + raw.auxk_alpha * aux
    else:  # vanilla
        x_hat, z = model(x)
        recon    = F.mse_loss(x_hat, x)
        l1       = z.abs().sum(dim=-1).mean()   # sum over features per token, mean over tokens
        loss     = recon + l1_lambda * l1

    return loss, x_hat, z


# =============================================================================
#  METRICS
# =============================================================================

def batch_metrics(x, x_hat, z, dead_threshold=1e-3):
    with torch.no_grad():
        recon    = F.mse_loss(x_hat, x).item()
        l1       = z.abs().mean().item()
        # L0: mean number of active features per residue (correct sparsity metric)
        l0       = (z > 0).float().sum(dim=-1).mean().item()
        sparsity = l0 / z.shape[-1]               # normalized for CSV logging
        unit_mean = z.abs().mean(dim=0)
        dead_frac = (unit_mean < dead_threshold).float().mean().item()
        ss_res = ((x - x_hat)**2).sum().item()
        ss_tot = ((x - x.mean(0, keepdim=True))**2).sum().item()
        r2     = 1.0 - ss_res / (ss_tot + 1e-8)
        unit_act_sum = z.abs().sum(dim=0).cpu()
    return {"recon": recon, "l1": l1, "sparsity": sparsity, "l0": l0,
            "dead_frac": dead_frac, "r2": r2}, unit_act_sum


def epoch_stats(unit_act_sums, n_samples, dead_threshold=1e-3):
    mean_act = unit_act_sums / max(n_samples, 1)
    alive    = (mean_act >= dead_threshold).float().mean().item()
    ever     = (mean_act > 0).float().mean().item()
    return {"alive_frac": alive, "ever_active_frac": ever,
            "dead_frac": 1.0 - alive}


def resample_dead_features(model, unit_act_sums, n_samples, dead_threshold=1e-3):
    """
    Re-initialise encoder/decoder weights for features that were dead across
    the whole epoch (mean activation < dead_threshold).

    Uses the Anthropic SAE recipe: sample a random live-feature direction,
    add small noise, normalise to unit norm for the decoder column, set the
    encoder row to the same direction scaled by the encoder norm mean.
    This gives dead features a fresh start without disrupting live ones.

    Only operates on the underlying (non-compiled) model parameters.
    Returns the number of features resampled.
    """
    raw_model = getattr(model, "_orig_mod", model)
    if not hasattr(raw_model, "decoder"):
        return 0

    mean_act   = unit_act_sums / max(n_samples, 1)          # [K]
    dead_mask  = (mean_act < dead_threshold)                 # [K] bool
    n_dead     = dead_mask.sum().item()
    if n_dead == 0:
        return 0

    with torch.no_grad():
        W_dec = raw_model.decoder.weight   # [D, K]
        W_enc = raw_model.encoder.weight   # [K, D]
        device = W_dec.device

        dead_idx  = dead_mask.nonzero(as_tuple=True)[0].to(device)
        alive_idx = (~dead_mask).nonzero(as_tuple=True)[0].to(device)

        if alive_idx.numel() == 0:
            return 0  # everything dead — can't resample from live features

        # sample random alive decoder columns as seed directions
        sample_idx = alive_idx[torch.randint(len(alive_idx), (len(dead_idx),))]
        new_dirs   = W_dec[:, sample_idx].clone()                    # [D, n_dead]
        new_dirs  += 0.01 * torch.randn_like(new_dirs)               # small noise
        new_dirs   = F.normalize(new_dirs, dim=0)                    # unit norm

        # decoder: replace dead columns
        W_dec[:, dead_idx] = new_dirs

        # encoder: replace dead rows; scale by mean alive encoder norm
        alive_enc_norm = W_enc[alive_idx].norm(dim=1).mean()
        W_enc[dead_idx] = new_dirs.T * alive_enc_norm

        # zero encoder bias for resampled features
        if raw_model.encoder.bias is not None:
            raw_model.encoder.bias[dead_idx] = 0.0

    return int(n_dead)


# =============================================================================
#  TRAIN ONE EPOCH
# =============================================================================

def run_epoch(
    model, arch, accs, seqs,
    esm_model, converter, backend, device, layer, max_len,
    optimizer, scaler, l1_lambda, batch_size,
    grad_clip, training, shuffle, seed, epoch_idx,
    amp_ctx,
    emb_mean, emb_std,
    dead_threshold=1e-3,
):
    """
    Stream through proteins, accumulate residues into batches, compute loss.
    Returns (metrics_dict, unit_act_sums, n_samples).
    """
    if training:
        model.train()
    else:
        model.eval()

    rng = random.Random(seed + epoch_idx * 997)
    order = list(accs)
    if shuffle:
        rng.shuffle(order)

    accum = ProteinBatchAccumulator(batch_size, device, arch)
    agg   = {"recon": 0., "l1": 0., "sparsity": 0.,
             "dead_frac": 0., "r2": 0., "l0": 0.}
    unit_act_sums = None
    n_batches = 0
    n_samples = 0
    n_failed  = 0

    split_name = "train" if training else "val"
    pbar = (_tqdm(order, desc=f"  {split_name}", unit="prot",
                  dynamic_ncols=True)
            if _tqdm else order)

    for acc in pbar:
        seq = seqs[acc]
        try:
            emb = get_embedding(acc, seq, esm_model, converter,
                                backend, device, layer, max_len)
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                log.warning(f"  ESM-2 failed for {acc}: {e}")
            continue

        accum.add(emb)

        # yield batches as they fill up
        while accum.ready():
            x, prot_lens = accum.flush()
            x = (x - emb_mean) / emb_std          # normalize: mean=0, std=1
            _step(model, arch, x, prot_lens, optimizer, scaler,
                  l1_lambda, grad_clip, training, amp_ctx)
            with torch.no_grad(), amp_ctx:
                _, z = model(x) if arch != "matryoshka" else (None, model.encode(x))
                x_hat = model.decode(z)
                # cast back to fp32 for stable metrics
                m, u = batch_metrics(x.float(), x_hat.float(), z.float(),
                                     dead_threshold)
            for k in agg: agg[k] += m[k]
            unit_act_sums = u if unit_act_sums is None else unit_act_sums + u
            n_batches += 1
            n_samples += x.shape[0]
            if _tqdm and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    recon=f"{m['recon']:.4f}",
                    l0=f"{m['l0']:.0f}",
                    dead=f"{m['dead_frac']:.3f}",
                    r2=f"{m['r2']:.3f}",
                    refresh=False,
                )
            # free GPU memory immediately
            del x, z, x_hat

    # flush remainder
    x, prot_lens = accum.flush_all()
    if x is not None and x.shape[0] > 0:
        x = (x - emb_mean) / emb_std              # normalize
        _step(model, arch, x, prot_lens, optimizer, scaler,
              l1_lambda, grad_clip, training, amp_ctx)
        with torch.no_grad(), amp_ctx:
            _, z = model(x) if arch != "matryoshka" else (None, model.encode(x))
            x_hat = model.decode(z)
            m, u = batch_metrics(x.float(), x_hat.float(), z.float(),
                                 dead_threshold)
        for k in agg: agg[k] += m[k]
        unit_act_sums = u if unit_act_sums is None else unit_act_sums + u
        n_batches += 1
        n_samples += x.shape[0]
        del x, z, x_hat

    if n_failed > 0:
        log.warning(f"  {n_failed} proteins failed ESM-2 inference")

    if n_batches == 0:
        return {k: float("nan") for k in agg}, None, 0

    avg = {k: v / n_batches for k, v in agg.items()}
    return avg, unit_act_sums, n_samples


def _step(model, arch, x, prot_lens, optimizer, scaler,
          l1_lambda, grad_clip, training, amp_ctx):
    """
    Single gradient step (or eval forward) with AMP + GradScaler.
    Frees intermediate tensors.
    """
    if not training:
        return
    optimizer.zero_grad(set_to_none=True)
    with amp_ctx:
        loss, _, _ = compute_loss(model, x, arch, l1_lambda, prot_lens)
    scaler.scale(loss).backward()
    # unscale before grad clip so the clip threshold is in fp32 scale
    scaler.unscale_(optimizer)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    # keep decoder columns at unit norm so activation magnitudes are comparable
    # unwrap torch.compile wrapper (same pattern as save_ckpt) before calling method
    raw_model = getattr(model, "_orig_mod", model)
    if hasattr(raw_model, "normalize_decoder"):
        raw_model.normalize_decoder()
    del loss


# =============================================================================
#  CHECKPOINTING
# =============================================================================

def save_ckpt(path, model, optimizer, epoch, config, best_val_loss):
    """
    Save checkpoint in the SAME format as train_sae.py for compatibility.
    If the model was compiled with torch.compile, saves the underlying
    _orig_mod state dict so it stays loadable without torch.compile.
    """
    raw_model = getattr(model, "_orig_mod", model)
    torch.save({
        "model_state_dict":     raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
        "config":               config,
        "best_val_loss":        best_val_loss,
    }, path)
    log.info(f"  Saved: {path.name} ({path.stat().st_size/1e6:.1f} MB)")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    global torch, nn, F

    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu"))
    log.info(f"Device: {device}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.outdir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ── validate args ─────────────────────────────────────────────────────────
    if args.architecture == "tsae" and args.shuffle_proteins:
        log.warning("T-SAE with --shuffle-proteins breaks temporal structure. "
                    "Removing shuffle for T-SAE.")
        args.shuffle_proteins = False

    # ── AMP setup ─────────────────────────────────────────────────────────────
    use_amp = (not args.no_amp) and (device.type == "cuda")
    if use_amp:
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        amp_ctx   = torch.autocast(device_type="cuda", dtype=amp_dtype)
        # GradScaler is only needed for float16 (bfloat16 has full range, no underflow)
        scaler    = (torch.cuda.amp.GradScaler()
                     if amp_dtype == torch.float16
                     else _NoOpScaler())
        log.info(f"AMP enabled: dtype={args.amp_dtype}  "
                 f"GradScaler={'yes' if amp_dtype == torch.float16 else 'no (bfloat16)'}")
    else:
        amp_ctx = _null_ctx()
        scaler  = _NoOpScaler()
        log.info("AMP disabled — running fp32")

    # ── load ESM-2 ────────────────────────────────────────────────────────────
    esm_model, converter, backend = load_esm2(args.esm2_model, device)

    # ── load sequences ────────────────────────────────────────────────────────
    log.info("Loading sequences ...")
    seqs = read_fasta(args.fasta)

    log.info("Loading protein splits ...")
    train_accs = load_split_accs(args.proteins, args.train_split, seqs,
                                 args.max_proteins, args.seed)
    val_accs   = load_split_accs(args.proteins, args.val_split,   seqs,
                                 max(args.max_proteins // 5, 1) if args.max_proteins else None,
                                 args.seed)

    # ── infer input dim from first protein ───────────────────────────────────
    log.info("Inferring input dim ...")
    test_emb = get_embedding(train_accs[0], seqs[train_accs[0]],
                             esm_model, converter, backend, device,
                             args.esm2_layer, args.max_seq_len)
    input_dim = test_emb.shape[1]
    log.info(f"  Input dim: {input_dim}")
    del test_emb

    # ── normalization stats ───────────────────────────────────────────────────
    emb_mean, emb_std = estimate_normalization_stats(
        train_accs, seqs, esm_model, converter, backend, device,
        args.esm2_layer, args.max_seq_len,
        n_proteins=500, seed=args.seed,
    )
    # move to device for fast in-place normalization during training
    emb_mean = emb_mean.to(device)
    emb_std  = emb_std.to(device)

    # ── build model ───────────────────────────────────────────────────────────
    arch = args.architecture
    if arch == "vanilla":
        model = make_vanilla_sae(input_dim, args.latent_dim)
    elif arch == "topk":
        model = make_topk_sae(input_dim, args.latent_dim, args.topk_k,
                              auxk_alpha=args.auxk_alpha)
    elif arch == "matryoshka":
        scales = args.matryoshka_scales or [
            args.latent_dim//8, args.latent_dim//4,
            args.latent_dim//2, args.latent_dim]
        model = make_matryoshka_sae(input_dim, args.latent_dim, scales)
        log.info(f"  Matryoshka scales: {model.scales}")
    elif arch == "tsae":
        model = make_tsae(input_dim, args.latent_dim,
                          args.tsae_lambda, args.tsae_temp)
        log.info(f"  T-SAE λ_temporal={args.tsae_lambda}, temp={args.tsae_temp}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Model: {arch}, K={args.latent_dim}, params={n_params/1e6:.1f}M")

    # ── torch.compile ─────────────────────────────────────────────────────────
    if (not args.no_compile) and hasattr(torch, "compile"):
        log.info("Compiling model with torch.compile (mode='reduce-overhead') ...")
        log.info("  First epoch will be slower due to compilation — this is expected.")
        # reduce-overhead: best for fixed-shape workloads (our batches are ~fixed size)
        model = torch.compile(model, mode="reduce-overhead")
    else:
        log.info("torch.compile disabled (use PyTorch >= 2.0 and remove --no-compile)")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    # ── config (saved with checkpoint) ───────────────────────────────────────
    config = {
        "model":         f"SAE_{arch}",
        "architecture":  arch,
        "input_dim":     input_dim,
        "latent_dim":    args.latent_dim,
        "l1_lambda":     args.l1_lambda,
        "l1_warmup_epochs": args.l1_warmup_epochs,
        "target_l0":     args.target_l0 if args.target_l0 > 0 else None,
        "learning_rate": args.learning_rate,
        "weight_decay":  args.weight_decay,
        "batch_size":    args.batch_size,
        "num_epochs":    args.num_epochs,
        "topk_k":        args.topk_k if arch == "topk" else None,
        "topk_k_start":  args.topk_k_start if arch == "topk" else None,
        "auxk_alpha":    args.auxk_alpha if arch == "topk" else None,
        "tsae_lambda":   args.tsae_lambda if arch == "tsae" else None,
        "tsae_temp":     args.tsae_temp if arch == "tsae" else None,
        "matryoshka_scales": model.scales if arch == "matryoshka" else None,
        "n_train":       len(train_accs),
        "n_val":         len(val_accs),
        "seed":          args.seed,
        "amp":           use_amp,
        "amp_dtype":     args.amp_dtype if use_amp else "fp32",
        "compiled":      (not args.no_compile) and hasattr(torch, "compile"),
        "normalized":    True,
        "emb_mean":      emb_mean.cpu().tolist(),
        "emb_std":       emb_std.cpu().tolist(),
        "timestamp":     datetime.now().isoformat(),
    }
    with open(args.outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── CSV metrics log ───────────────────────────────────────────────────────
    csv_path = args.outdir / "metrics.csv"
    csv_fields = ["epoch", "l1_eff", "train_recon", "train_l1", "train_sparsity",
                  "train_l0", "train_dead_frac", "val_recon", "val_r2",
                  "alive_frac", "ever_active_frac", "elapsed_s"]
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    best_epoch       = 0
    _warmup_done     = False          # True once warmup finishes; resets tracking
    # adaptive lambda state: starts at args.l1_lambda, adjusted each epoch if --target-l0 set
    l1_lambda_cur    = args.l1_lambda
    _l1_arch         = arch in ("vanilla", "tsae", "matryoshka")
    # TopK k annealing: linear decay from topk_k_start → topk_k over warmup epochs
    _topk_anneal     = (arch == "topk" and args.topk_k_start > args.topk_k)

    log.info(f"\nTraining {arch} SAE for {args.num_epochs} epochs ...")
    log.info(f"  Train: {len(train_accs):,} proteins")
    log.info(f"  Val:   {len(val_accs):,} proteins")
    if args.target_l0 > 0 and _l1_arch:
        log.info(f"  Adaptive lambda: targeting l0={args.target_l0}")
    if _topk_anneal:
        log.info(f"  TopK k annealing: {args.topk_k_start} → {args.topk_k} "
                 f"over {args.l1_warmup_epochs} epochs")

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        # L1 warmup (applied on top of adaptive lambda)
        if args.l1_warmup_epochs <= 1:
            l1_eff = l1_lambda_cur
        else:
            l1_eff = l1_lambda_cur * min(epoch / args.l1_warmup_epochs, 1.0)

        # TopK k annealing: linearly decay from k_start to k over warmup epochs
        if _topk_anneal:
            warmup_frac = min(epoch / args.l1_warmup_epochs, 1.0)
            k_eff = int(round(args.topk_k_start + (args.topk_k - args.topk_k_start) * warmup_frac))
            raw_model = getattr(model, "_orig_mod", model)
            raw_model.k = k_eff
            log.info(f"\nEpoch {epoch}/{args.num_epochs}  k={k_eff}")
        else:
            log.info(f"\nEpoch {epoch}/{args.num_epochs}  λ={l1_eff:.2e}")

        # train
        train_m, train_u, train_n = run_epoch(
            model, arch, train_accs, seqs,
            esm_model, converter, backend, device,
            args.esm2_layer, args.max_seq_len,
            optimizer, scaler, l1_eff, args.batch_size,
            args.grad_clip, training=True,
            shuffle=args.shuffle_proteins,
            seed=args.seed, epoch_idx=epoch,
            amp_ctx=amp_ctx,
            emb_mean=emb_mean, emb_std=emb_std,
        )

        # val
        val_m, val_u, val_n = run_epoch(
            model, arch, val_accs, seqs,
            esm_model, converter, backend, device,
            args.esm2_layer, args.max_seq_len,
            optimizer, scaler, l1_eff, args.batch_size,
            args.grad_clip, training=False,
            shuffle=False,
            seed=args.seed, epoch_idx=0,
            amp_ctx=amp_ctx,
            emb_mean=emb_mean, emb_std=emb_std,
        )

        elapsed = time.time() - t0

        # epoch-level stats
        sp = epoch_stats(train_u, train_n) if train_u is not None else {}

        # dead feature resampling (L1-based archs; TopK uses AuxK loss instead)
        if _l1_arch and train_u is not None and sp.get("dead_frac", 0) > 0.05:
            n_resampled = resample_dead_features(model, train_u, train_n)
            if n_resampled > 0:
                log.info(f"  resampled {n_resampled} dead features "
                         f"({n_resampled/args.latent_dim*100:.1f}%)")

        log.info(
            f"  train recon={train_m['recon']:.5f}  "
            f"l0={train_m['l0']:.0f}  "
            f"sparsity={train_m['sparsity']:.4f}  "
            f"dead={train_m['dead_frac']:.3f}  "
            f"alive={sp.get('alive_frac', 0):.3f}  "
            f"val_recon={val_m['recon']:.5f}  "
            f"val_r2={val_m['r2']:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # CSV
        csv_writer.writerow({
            "epoch":          epoch,
            "l1_eff":         f"{l1_eff:.2e}",
            "train_recon":    f"{train_m['recon']:.6f}",
            "train_l1":       f"{train_m['l1']:.6f}",
            "train_sparsity": f"{train_m['sparsity']:.4f}",
            "train_l0":       f"{train_m['l0']:.1f}",
            "train_dead_frac":f"{sp.get('dead_frac', train_m['dead_frac']):.4f}",
            "val_recon":      f"{val_m['recon']:.6f}",
            "val_r2":         f"{val_m['r2']:.4f}",
            "alive_frac":     f"{sp.get('alive_frac', 0):.4f}",
            "ever_active_frac":f"{sp.get('ever_active_frac', 0):.4f}",
            "elapsed_s":      f"{elapsed:.0f}",
        })
        csv_file.flush()

        # ── reset tracking once warmup finishes ─────────────────────────────
        # During warmup (L1 ramp-up or TopK k-annealing) reconstruction is
        # artificially low because sparsity is weaker.  Comparing those epochs
        # with post-warmup ones poisons early-stopping and best-checkpoint
        # selection.  We reset best_val_loss the first epoch after warmup so
        # that all further comparisons are fair.
        _has_warmup = (args.l1_warmup_epochs > 1) or _topk_anneal
        if _has_warmup and not _warmup_done and epoch > args.l1_warmup_epochs:
            _warmup_done   = True
            best_val_loss  = float("inf")
            best_epoch     = epoch
            log.info("  [warmup done — resetting best_val_loss for fair comparison]")

        # checkpoint: best.pt + last.pt only (no per-epoch unless requested)
        val_loss = val_m["recon"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            save_ckpt(ckpt_dir / "best.pt", model, optimizer,
                      epoch, config, best_val_loss)

        # adaptive lambda: adjust l1_lambda_cur to track target_l0.
        # Guard: only increase lambda when reconstruction is healthy (within 20% of
        # best). If recon has already degraded, back off lambda instead so the model
        # can recover before sparsity pressure is reapplied.
        if (args.target_l0 > 0 and _l1_arch
                and epoch >= args.l1_warmup_epochs
                and train_m["l0"] > 0):
            dead_frac = sp.get("dead_frac", 0.0)
            ratio = train_m["l0"] / args.target_l0
            recon_ratio = val_loss / best_val_loss  # 1.0 = at best, >1 = degraded
            if dead_frac > 0.3 and ratio > 1.0:
                # >30% features dead — lambda too high, back off to let resampling work
                multiplier = 0.5
                note = f"  [dead={dead_frac:.2f} — backing off]"
            elif recon_ratio > 2.0 and ratio > 1.0:
                # Reconstruction badly degraded (>2× best) — back off to recover
                multiplier = 0.5
                note = f"  [recon={recon_ratio:.2f}× best — backing off]"
            elif recon_ratio > 1.2 and ratio > 1.0:
                # Reconstruction moderately degraded — hold lambda, don't increase pressure
                multiplier = 1.0
                note = f"  [recon={recon_ratio:.2f}× best — holding]"
            else:
                # cap to [0.33, 2.0] per epoch (was 3.0 — too aggressive for large gaps)
                multiplier = max(0.33, min(2.0, ratio ** 0.5))
                note = ""
            l1_lambda_cur *= multiplier
            l1_lambda_cur  = max(1e-7, min(1e-1, l1_lambda_cur))
            log.info(f"  adaptive λ: l0={train_m['l0']:.0f} → target={args.target_l0}  "
                     f"×{multiplier:.2f}  new λ={l1_lambda_cur:.2e}" + note)

        save_ckpt(ckpt_dir / "last.pt", model, optimizer,
                  epoch, config, best_val_loss)

        # optional numbered checkpoint (disk usage warning)
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            log.warning(f"  Saving numbered checkpoint (each is "
                        f"~{(ckpt_dir/'best.pt').stat().st_size/1e6:.0f}MB)")
            save_ckpt(ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer,
                      epoch, config, best_val_loss)

        # free GPU cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # early stopping — only active after warmup to avoid unfair comparisons
        if args.early_stop_patience > 0 and epoch > args.l1_warmup_epochs:
            epochs_no_improve = epoch - best_epoch
            if epochs_no_improve >= args.early_stop_patience:
                log.info(f"\nEarly stopping: no improvement for {epochs_no_improve}/{args.early_stop_patience} epochs.")
                break
            elif epochs_no_improve > 0:
                log.info(f"  No improvement for {epochs_no_improve}/"
                         f"{args.early_stop_patience} epochs")

    csv_file.flush()
    csv_file.close()

    log.info(f"\n✅ Done. Best val_recon={best_val_loss:.6f} at epoch {best_epoch}")
    log.info(f"   Checkpoints: {ckpt_dir}")
    log.info(f"   Disk used: "
             f"{sum(f.stat().st_size for f in ckpt_dir.glob('*.pt'))/1e6:.0f} MB")


# =============================================================================
#  AMP HELPERS
# =============================================================================

class _NoOpScaler:
    """Drop-in replacement for GradScaler when AMP is off or using bfloat16."""
    def scale(self, loss):     return loss
    def unscale_(self, opt):   pass
    def step(self, opt):       opt.step()
    def update(self):          pass


class _null_ctx:
    """Null context manager — used when AMP is disabled."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


if __name__ == "__main__":
    main()