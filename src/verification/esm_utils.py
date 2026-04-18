"""
esm_utils.py — shared ESM-2 helpers for verification scripts.

Mirrors the load_esm2 / get_embedding functions from src/stream/train_sae_variants.py
so verification scripts can run on-the-fly without pre-extracted shards.
"""

import torch


def load_esm2(model_name: str, device: torch.device):
    """
    Load ESM-2 model. Tries fair-esm first, falls back to HuggingFace.
    Returns (model, converter, backend) where backend is 'esm' or 'hf'.
    """
    import logging
    log = logging.getLogger(__name__)
    log.info(f"Loading ESM-2 ({model_name}) ...")
    try:
        import esm as _esm
        model, alphabet = _esm.pretrained.load_model_and_alphabet(model_name)
        batch_converter = alphabet.get_batch_converter()
        model.eval().to(device)
        log.info("  Loaded via fair-esm")
        return model, batch_converter, "esm"
    except Exception:
        log.info("  fair-esm not found — trying HuggingFace transformers ...")
        from transformers import EsmModel, EsmTokenizer
        hf_name = model_name if "/" in model_name else f"facebook/{model_name}"
        tokenizer = EsmTokenizer.from_pretrained(hf_name)
        model = EsmModel.from_pretrained(hf_name)
        model.eval().to(device)
        log.info("  Loaded via HuggingFace")
        return model, tokenizer, "hf"


@torch.no_grad()
def get_embedding(acc: str, seq: str, esm_model, converter, backend: str,
                  device: torch.device, layer: int, max_len: int = 1022):
    """
    Compute per-residue ESM-2 embeddings for a single protein.
    Returns [L, D] float32 numpy array. Never stores to disk.
    """
    seq = seq[:max_len]
    L = len(seq)
    if backend == "esm":
        _, _, tokens = converter([(acc, seq)])
        tokens = tokens.to(device)
        out = esm_model(tokens, repr_layers=[layer], return_contacts=False)
        emb = out["representations"][layer][0, 1:L + 1, :].cpu().float().numpy()
    else:
        inputs = converter(seq, return_tensors="pt", truncation=True,
                           max_length=max_len + 2).to(device)
        out = esm_model(**inputs, output_hidden_states=True)
        emb = out.hidden_states[layer][0, 1:L + 1, :].cpu().float().numpy()
    return emb  # [L, D]
