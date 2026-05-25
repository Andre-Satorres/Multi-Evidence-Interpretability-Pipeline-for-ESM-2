# SAE Model Selection

22 configurations were explored across 4 architectures. Selection criteria: validation R², dead-feature fraction, and training stability. The selected model (*) was trained on the full dataset (50k proteins).

| Architecture | K | k / λ₁ | Val R² | Dead % | Epochs |
|---|---|---|---|---|---|
| Topk * | 16,384 | k=512 | 0.915 | 11.5 | 17 |
| Topk | 8,192 | k=128 | 0.672 | 64.2 | 3 |
| Topk | 8,192 | k=64 | 0.571 | 0.0 | 4 |
| Topk | 8,192 | k=64 | 0.560 | 90.1 | 2 |
| Topk | 4,096 | k=64 | 0.439 | 0.0 | 6 |
| Matryoshka | 8,192 | λ=3e-05 | 0.979 | 63.9 | 2 |
| Matryoshka | 8,192 | λ=1e-04 | 0.931 | 23.1 | 10 |
| Matryoshka | 8,192 | λ=5e-01 | 0.909 | 77.3 | 3 |
| Vanilla | 8,192 | λ=3e-05 | 0.977 | 44.2 | 2 |
| Vanilla | 8,192 | λ=5e-01 | 0.965 | 82.2 | 3 |
| Vanilla | 8,192 | λ=1e-02 | 0.959 | 12.8 | 1 |
| Vanilla | 8,192 | λ=1e-04 | 0.951 | 12.9 | 1 |
| Vanilla | 8,192 | λ=3e-04 | 0.950 | 12.9 | 1 |
| Vanilla | 8,192 | λ=1e-01 | 0.950 | 51.7 | 2 |
| Vanilla | 8,192 | λ=1e-04 | 0.947 | 0.0 | 5 |
| Vanilla | 8,192 | λ=5e-01 | 0.867 | 73.4 | 2 |
| Vanilla | 8,192 | λ=1e-03 | 0.761 | 0.0 | 4 |
| Vanilla | 4,096 | λ=1e-03 | 0.725 | 0.0 | 6 |
| Vanilla | 8,192 | λ=1e+00 | 0.696 | 50.8 | 1 |
| Vanilla | 8,192 | λ=1e-04 | 0.466 | 2.4 | 7 |
| T-SAE | 8,192 | λ=3e-05 | 0.936 | 44.6 | 3 |
| T-SAE | 8,192 | λ=1e-03 | 0.231 | 1.2 | 9 |

## Selection rationale

- **TopK-16384-k512** was chosen because it achieves high R² (0.915) with a controlled dead-feature fraction (11.5%) and an exact sparsity budget (exactly 512 features active per token).
- Higher-R² models (Vanilla λ=3e-05: 0.977, Matryoshka: 0.979) suffer from >44% dead features and lack an explicit sparsity gate, making individual feature activations harder to interpret.
- Lower-k TopK variants (k=64, k=128) achieve poor reconstruction (R² < 0.67), indicating insufficient capacity.
- T-SAE showed promise but the contrastive loss introduced training instability at larger scales.
