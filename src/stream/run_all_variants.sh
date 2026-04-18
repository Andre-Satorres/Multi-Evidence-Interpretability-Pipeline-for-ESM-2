#!/bin/bash
# Usage:
#   ./run_all_variants.sh                          # run all 4 archs, 50k proteins
#   ./run_all_variants.sh vanilla topk             # run only vanilla and topk
#   ./run_all_variants.sh all 10000                # all archs, 10k proteins
#   ./run_all_variants.sh vanilla 5000             # vanilla only, 5k proteins
#   ./run_all_variants.sh tsae matryoshka 20000    # two archs, 20k proteins
#
# Arguments (any order):
#   Architecture names: vanilla | topk | matryoshka | tsae
#   A plain integer:    sets --max-proteins (default: 50000)
#   "all":              run all four architectures (default if no arch given)
set -e

# ── parse args ────────────────────────────────────────────────────────────────
RUN_VANILLA=0; RUN_TOPK=0; RUN_MATRYOSHKA=0; RUN_TSAE=0
MAX_PROTEINS=50000
ANY_ARCH=0

for arg in "$@"; do
    case "$arg" in
        vanilla)     RUN_VANILLA=1;    ANY_ARCH=1 ;;
        topk)        RUN_TOPK=1;       ANY_ARCH=1 ;;
        matryoshka)  RUN_MATRYOSHKA=1; ANY_ARCH=1 ;;
        tsae)        RUN_TSAE=1;       ANY_ARCH=1 ;;
        all)         ANY_ARCH=2 ;;   # sentinel: run all
        ''|*[!0-9]*) echo "Unknown argument: $arg" >&2; exit 1 ;;
        *)           MAX_PROTEINS=$arg ;;
    esac
done

# default: run all if no arch specified
if [ "$ANY_ARCH" -eq 0 ] || [ "$ANY_ARCH" -eq 2 ]; then
    RUN_VANILLA=1; RUN_TOPK=1; RUN_MATRYOSHKA=1; RUN_TSAE=1
fi

# ── shared config ─────────────────────────────────────────────────────────────
TARGET_L0=512
L1_INIT=1e-3
TOPK_K=512
ESM_LAYER=33

BASE="python3 src/stream/train_sae_variants.py \
    --esm2-model esm2_t33_650M_UR50D \
    --esm2-layer $ESM_LAYER \
    --fasta data/proteins.fasta \
    --proteins data/proteins_with_split.tsv \
    --max-proteins $MAX_PROTEINS \
    --num-epochs 20 \
    --early-stop-patience 10 \
    --l1-warmup-epochs 3 \
    --latent-dim 16384 \
    --shuffle-proteins"

echo "Max proteins: $MAX_PROTEINS"
ARCH_LIST=""
[ "$RUN_VANILLA" -eq 1 ]    && ARCH_LIST="$ARCH_LIST vanilla"
[ "$RUN_TOPK" -eq 1 ]       && ARCH_LIST="$ARCH_LIST topk"
[ "$RUN_MATRYOSHKA" -eq 1 ] && ARCH_LIST="$ARCH_LIST matryoshka"
[ "$RUN_TSAE" -eq 1 ]       && ARCH_LIST="$ARCH_LIST tsae"
echo "Architectures:$ARCH_LIST"

# ── runs ──────────────────────────────────────────────────────────────────────
if [ "$RUN_VANILLA" -eq 1 ]; then
    echo "=== Vanilla ===" && $BASE \
        --architecture vanilla \
        --l1-lambda    $L1_INIT \
        --target-l0    $TARGET_L0 \
        --outdir outputs/sae_variants/vanilla_${MAX_PROTEINS}
fi

if [ "$RUN_MATRYOSHKA" -eq 1 ]; then
    echo "=== Matryoshka ===" && $BASE \
        --architecture matryoshka \
        --l1-lambda    $L1_INIT \
        --target-l0    $TARGET_L0 \
        --outdir outputs/sae_variants/matryoshka_${MAX_PROTEINS}
fi

if [ "$RUN_TSAE" -eq 1 ]; then
    # T-SAE: no --shuffle-proteins (temporal structure must be preserved)
    echo "=== T-SAE ===" && python3 src/stream/train_sae_variants.py \
        --esm2-model esm2_t33_650M_UR50D \
        --esm2-layer $ESM_LAYER \
        --fasta data/proteins.fasta \
        --proteins data/proteins_with_split.tsv \
        --max-proteins $MAX_PROTEINS \
        --num-epochs 13 \
        --early-stop-patience 5 \
        --l1-warmup-epochs 3 \
        --latent-dim 16384 \
        --architecture tsae \
        --tsae-lambda  0.1 \
        --l1-lambda    $L1_INIT \
        --target-l0    $TARGET_L0 \
        --outdir outputs/sae_variants/tsae_${MAX_PROTEINS}
fi

if [ "$RUN_TOPK" -eq 1 ]; then
    echo "=== TopK ===" && $BASE \
        --architecture topk \
        --topk-k       $TOPK_K \
        --topk-k-start 512 \
        --auxk-alpha   0.03125 \
        --outdir outputs/sae_variants/topk_${MAX_PROTEINS}
fi

echo "=== All done ==="
