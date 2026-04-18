# Visão geral

## Objetivo do paper

Contributions
* Homology-aware evaluation protocol for interpretability in protein language models
* Residue-level and region-level validation of SAE features against curated annotations
* Comparative analysis between SAE, attention, and mutation-based attribution methods
* A quantitative discovery score for identifying novel functional regions
* Structural validation using AlphaFold

alta F1 (global) mas baixa localização (resíduo) ????
pega feature deles
calcula:
F1 proteína (igual deles)
AUROC por resíduo e compara

F1 alto
AUROC médio/baixo -> limitação do InterPLM

Comparar **SAE**, **attention** e **mutation sensitivity** como métodos de interpretabilidade sobre embeddings/PLMs de proteínas, avaliando se recuperam **anotações biológicas locais** e se o **superam baselines aleatórios**.

## Estratégia em 3 escalas

* **Fase 1:** 5k proteínas -> validar pipeline e métricas
* **Fase 2:** 50k proteínas -> resultado principal robusto
* **Fase 3:** 500k proteínas -> escalar treino do SAE e fortalecer paper, se der tempo

# Fase 0 — definição do experimento antes de escalar

## 0.1 Definir as tasks/alvos

Escolher poucas annotations locais e defensáveis.

* **HELIX**
* **STRAND / BETA_SHEET**
* **TURN** se existir bem curada
* **TRANSMEM**
* **BINDING**
* **ACTIVE_SITE**
* **METAL_BINDING**

Talvez separar em dois grupos:

* **estruturais**: helix, strand, transmem
* **funcionais**: binding, active site, metal binding

## 0.2 Definir a unidade de avaliação

* **principal:** por resíduo
* **secundária:** por segmento/região

## 0.3 Definir o baseline aleatório

* random embedding dimensions
* random projection
* shuffled scores

---

# Fase 1 — subset 5k

## 1) Baixar dados

* accession/id
* sequence
* annotations Swiss-Prot

Idealmente salvar em algo tipo:

* `proteins.tsv`
* `annotations.tsv`

### Campos mínimos de annotations

* accession
* feature_type
* start
* end
* description
* evidence se existir

---

## 2) Escolher o modelo

**ProtBert ou ESM-2 intermediário**, para artigo em 2 semanas, maximizar:

* embedding por resíduo
* facilidade de extração
* estabilidade
* reprodutibilidade

**ProtBert** para o artigo.

---

## 3) Pegar 5k proteínas e split train/val/test por homologia

* cluster por homologia com CD-HIT ou MMseqs2
* split por cluster
* algo como:
  * 70% train
  * 15% val
  * 15% test

### Importante

No subset 5k, o split tem que ser já "de verdade", igual ao experimento final.

---

## 4) Extrair embeddings

Salvar por proteína:

* accession
* `embedding[L, D]`
* `outputs/embeddings/{accession}.pt`

Salvar também um arquivo de metadados geral com:
* modelo usado
* dimensão D
* tokenizer/config
---

## 5) Treinar SAE com os embeddings

### Quantos neurônios?

* `latent_dim = 2 * D`
* `latent_dim = 4 * D`
* `latent_dim = 8 * D`

Se ProtBert tiver `D=1024`, isso daria:

* 2048
* 4096
* 8192

### Como escolher?

No conjunto de validação, compare:

* reconstruction loss
* sparsity real
* número de features mortas
* interpretabilidade preliminar
* desempenho contra annotations

---

# Fase 1.5 — definir score por resíduo para cada método

## 6) Para SAE, attention e mutation sensitivity, gerar score por resíduo

### SAE

#### Score agregado por resíduo

```text
score_sae(i) = max_k z[i,k]
```

#### Score geral por feature específica

Para avaliar "feature helix-like", por exemplo:

```text
score_sae_k(i) = z[i,k]
```

### Attention

Agregação simples e fixa:

* média da atenção recebida
* média sobre heads
* talvez último layer e média dos últimos layers

Exemplo:

```text
score_attn(i) = média da atenção recebida por i
```

### Mutation sensitivity

Output-alvo:

* mudança no embedding?
* mudança na reconstrução do SAE?
* mudança em um score downstream?

```text
score_mut(i) = || emb(seq) - emb(seq_mask_i) ||
```

# Fase 1.6 — avaliação

## 7) Comparar com annotations e calcular métricas

Para cada método e annotation:

* AUROC
* Average Precision
* enrichment no top-k%
* Mann–Whitney U / permutation test

### A. análise global por método

"attention vs mutation vs SAE geral"

### B. análise por feature SAE

"quais neurônios/features latentes estão mais associados a helix/binding/etc"

---

## 8) Comparar os métodos entre si

### 8.1 Método vs annotation

* qual método bate melhor com Swiss-Prot?

### 8.2 Método vs método

* correlação entre mapas por resíduo
* overlap top residues
* agreement em regiões

### 8.3 SAE vs random

* este precisa ser uma seção própria

---

# Estrutura corrigida do plano

## Etapa A — Dados

1. Baixar Swiss-Prot com sequência + annotations locais
2. Filtrar feature types alvo
3. Clusterizar por homologia
4. Construir splits train/val/test por cluster

## Etapa B — Modelo base

5. Escolher PLM base (preferência: ProtBert ou ESM-2 maior)
6. Extrair embeddings por resíduo

## Etapa C — SAE

7. Treinar SAE no train
8. Selecionar hiperparâmetros via val
9. Gerar latents no test

## Etapa D — Interpretabilidade comparativa

10. Gerar score por resíduo com:

* SAE
* attention
* mutation sensitivity

11. Gerar baselines aleatórios

## Etapa E — Avaliação

12. Comparar cada método contra annotations
13. Comparar métodos entre si
14. Comparar SAE vs random
15. Identificar features SAE específicas associadas a:

* helix
* strand
* transmem
* binding
* active site

## Etapa F — Casos qualitativos

16. Escolher proteínas bem estudadas
17. Visualizar regiões no AlphaFold
18. Mostrar agreement entre métodos e estrutura

## Etapa G — Escalonamento

19. Repetir pipeline em 50k
20. Se der tempo, expandir treino do SAE para 500k

---

    python src/verification/multi_evidence.py \
        --alignment    outputs/feature_alignment_streaming/per_annot_summary.tsv \
        --all-scores   outputs/feature_alignment_streaming/alignment_scores.parquet \
        --shard-dir    outputs/embeddings/esm2_650m \
        --checkpoint   outputs/sae_variants/matryoshka_50000/checkpoints/best.pt \
        --annotations  data/annotations_enriched.tsv \
        --proteins     data/proteins_with_split.tsv \
        --split        test \
        --top-pairs    8 \
        --activation-threshold 0.1 \
        --outdir       outputs/multi_evidence_enriched

python src/verification/paper_figures.py     --checkpoint      outputs/sae_variants/matryoshka_50000/checkpoints/best.pt     --alignment-dir   outputs/feature_alignment_streaming     --multi-ev-dir    outputs/multi_evidence_enriched     --attention-dir   outputs/attention_enriched     --shard-dir       outputs/embeddings/esm2_650m     --annotations     data/annotations_enriched.tsv     --proteins        data/proteins_with_split.tsv     --split           test     --case-proteins   "P0CQ65:492:Modified residue: N-acetylalanine:n_acetylalanine,Q6ZQJ5:707:Binding site: [4Fe-4S] cluster:fe_s_binding,Q5XI64:976:Active site: Charge relay system:charge_relay_site,A2AGA4:976:Active site: Nucleophile:nucleophile,Q91XS8:976:Active site: Proton acceptor:proton_acceptor,C5BCM3:976:Binding site: A divalent metal cation:a_divalent_metal_cation,Q9I993:976:Binding site: Fe cation:fe_cation"     --outdir          outputs/paper_figures_enriched_v4

python src/verification/paper_figures.py     --checkpoint      outputs/sae_variants/matryoshka_50000/checkpoints/best.pt     --alignment-dir   outputs/feature_alignment_streaming     --multi-ev-dir    outputs/multi_evidence_enriched     --attention-dir   outputs/attention_enriched     --shard-dir       outputs/embeddings/esm2_650m     --annotations     data/annotations_enriched.tsv     --proteins        data/proteins_with_split.tsv     --split           test     --case-proteins   "P49594:70:Domain: PPM-type phosphatase:phosphatase,P24733:122:Domain: Myosin motor:myosin_motor,Q61574:435:DNA binding: Fork-head:fork_head,Q9U903:565:DNA binding: T-box:t_box,Q92155:744:Domain: IF rod:if_rod,Q81EK9:754:Domain: NodB homology:nodb_homology,Q9P5X8:1011:Domain: RNase H type-2:rnase_h_type_2"     --outdir          outputs/paper_figures_enriched_v4

---

## Comandos para regenerar resultados

Checkpoint novo: outputs/sae_variants/topk_50000/checkpoints/best.pt

1. Feature Alignment — todas as K features (principal tabela do paper)

python3 src/stream/feature_alignment_streaming.py \
    --checkpoint  outputs/sae_variants/topk_50000/checkpoints/best.pt \
    --fasta       data/proteins.fasta \
    --proteins    data/proteins_with_split.tsv \
    --annotations data/annotations_enriched.tsv \
    --split       test \
    --all-features \
    --outdir      outputs/feature_alignment_topk50k \
    2>&1 | tee logs/feature_alignment_topk50k.log

Isso é o mais importante e o mais longo (~horas). Avalia todos os 16k features contra 439 tipos de anotação.

2. Multi-evidence — análise causal dos top 8 pares

python3 src/verification/multi_evidence.py \
    --alignment    outputs/feature_alignment_topk512/per_annot_summary.tsv \
    --all-scores   outputs/feature_alignment_topk512/alignment_scores.parquet \
    --checkpoint   outputs/sae_variants/topk_50000/checkpoints/best.pt \
    --annotations  data/annotations_enriched.tsv \
    --proteins     data/proteins_with_split.tsv \
    --split        test \
    --top-pairs    8 \
    --outdir       outputs/multi_evidence_topk512 \
    2>&1 | tee logs/multi_evidence_topk512.log

3. Feature qualitative — heatmaps por proteína

python3 src/verification/feature_qualitative.py \
    --alignment    outputs/feature_alignment_topk512/per_annot_summary.tsv \
    --all-scores   outputs/feature_alignment_topk512/alignment_scores.parquet \
    --checkpoint   outputs/sae_variants/topk_50000/checkpoints/best.pt \
    --annotations  data/annotations_enriched.tsv \
    --proteins     data/proteins_with_split.tsv \
    --split        test \
    --top-pairs    10 \
    --outdir       outputs/feature_qualitative_topk512 \
    2>&1 | tee logs/feature_qualitative_topk512.log

4. Paper figures — todas as figuras (fig1–fig6)

python3 src/verification/paper_figures.py \
    --checkpoint      outputs/sae_variants/topk_50000/checkpoints/best.pt \
    --alignment-dir   outputs/feature_alignment_topk512 \
    --multi-ev-dir    outputs/multi_evidence_topk512 \
    --attention-dir   outputs/attention_enriched \
    --annotations     data/annotations_enriched.tsv \
    --proteins        data/proteins_with_split.tsv \
    --split           test \
    --case-proteins   "A9KGD3|15747|Domain: RNase H type-2|rnase_h_type_2,Q61574|14069|DNA binding: Fork-head|fork_head,Q863A2|8657|DNA binding: T-box|t_box,Q9PUQ1|8537|DNA binding: ETS|ets,P32451|7808|Binding site: [4Fe-4S] cluster|fe_s_cluster,Q9NRS4|979|Transmembrane: Helical; Signal-anchor for type II membrane protein|transmem_signal_anchor,Q22307|15450|Binding site: Mn(2+)|mn_binding,Q8NHH9|5260|Binding site: GTP|gtp_binding" \
    --outdir          outputs/paper_figures_topk512 \
    2>&1 | tee logs/paper_figures_topk512.log

Os --case-proteins são os mesmos do v6 — pode ajustar se quiser outros exemplos qualitativos.

Ordem de dependência

1. feature_alignment  (independente — rodar primeiro)
        ↓
2. multi_evidence     (depende do alignment)
3. feature_qualitative (depende do alignment)
        ↓
4. paper_figures      (depende de alignment + multi_evidence)

Passos 2 e 3 podem rodar em paralelo (terminais separados) depois que o 1 acabar.

Nota: Os comandos já usam o novo código sem shards — ESM2 roda on-the-fly. Não precisa reextrair embeddings


-----------------

# Parallel feature alignment

# clean up old checkpoint files
rm -f outputs/feature_alignment_topk512/pass2_checkpoint*.pkl
rm -f outputs/feature_alignment_topk512/pass2_checkpoint*.tmp
rm -f outputs/feature_alignment_topk512/pass2_shard*.pkl
# (keep pass1_stats.npz !)

# restart both
CUDA_VISIBLE_DEVICES=2 nohup python src/stream/feature_alignment_streaming.py \
    --checkpoint outputs/sae_variants/topk_50000/checkpoints/best.pt \
    --fasta data/proteins.fasta --proteins data/proteins_with_split.tsv \
    --annotations data/annotations_enriched.tsv --split test --all-features \
    --select-by variance --min-proteins 30 --min-annot-residues 500 \
    --esm-batch-size 16 --align-batch-size 128 \
    --num-shards 2 --shard-idx 0 \
    --outdir outputs/feature_alignment_topk512 \
    > logs/feature_alignment_topk512_s0.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python src/stream/feature_alignment_streaming.py \
    --checkpoint outputs/sae_variants/topk_50000/checkpoints/best.pt \
    --fasta data/proteins.fasta --proteins data/proteins_with_split.tsv \
    --annotations data/annotations_enriched.tsv --split test --all-features \
    --select-by variance --min-proteins 30 --min-annot-residues 500 \
    --esm-batch-size 16 --align-batch-size 128 \
    --num-shards 2 --shard-idx 1 \
    --outdir outputs/feature_alignment_topk512 \
    > logs/feature_alignment_topk512_s1.log 2>&1 &