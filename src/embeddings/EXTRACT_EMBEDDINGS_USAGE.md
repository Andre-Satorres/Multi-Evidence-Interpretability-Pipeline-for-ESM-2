# default 10% subset, all splits
python src/embeddings/extract_embeddings.py

# debug run, 2% subset
python src/embeddings/extract_embeddings.py \
    --input outputs/subsets/proteins_subset_2pct.tsv \
    --max-tokens 2048 --shard-size 128

# resume interrupted run
python src/embeddings/extract_embeddings.py --resume

# load a shard
data = torch.load("outputs/embeddings/esm2_650m/train_shard_000.pt",
                  weights_only=False)
emb = data["embeddings"][0]   # Tensor[L, 1280]