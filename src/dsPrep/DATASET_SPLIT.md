# Homology Split

Impedir que proteínas muito parecidas caiam em treino e teste.

* agrupar proteínas por similaridade de sequência
* colocar clusters inteiros em train, val ou test
* só depois extrair embeddings e treinar

## Clustering

mmseqs createdb proteins.fasta proteins_db

mmseqs cluster proteins_db clusters tmp \
  --min-seq-id 0.3 \
  -c 0.8 \
  --cov-mode 0

--min-seq-id 0.3 = mínimo de 30% de identidade

-c 0.8 = cobertura de 80%

--cov-mode 0 = cobertura aplicada de forma simétrica

## Export clusters to tsv

mmseqs createtsv proteins_db proteins_db clusters clusters.tsv

## Split clusters into train, val, test

python split_clusters.py --clusters clusters.tsv --train 0.7 --val 0.15

## Add split info to proteins

python add_split_info.py

# Analysis

## A) Split Percentage

```text
train: 402,254 (70.0%), 54,845 clusters (71.1%), 4,039,787 annotations (71.4%)
val:    86,972 (15.1%), 11,025 clusters (14.3%),   775,311 annotations (13.7%)
test:   85,401 (14.9%), 11,244 clusters (14.6%),   839,426 annotations (14.8%)
```

O split ficou **muito bom** em proteínas e em clusters:

* proteínas: praticamente 70/15/15
* clusters: 71.1 / 14.3 / 14.6, bem próximo
* annotations: também perto, embora val tenha um pouco menos e train um pouco mais

Isso sugere que o particionamento por homologia **não distorceu o volume dos dados** entre os splits. 
Conseguimos respeitar os clusters sem destruir o balanceamento global.

* o split é **homology-aware**
* e ao mesmo tempo **quantitativamente balanceado**
* então diferenças de desempenho entre train/val/test têm mais chance de refletir generalização real, não só artefato do split

> Homology-aware partitioning yielded balanced train, validation, and test sets, containing 70.0%, 15.1%, and 14.9% of proteins, respectively, while preserving a similar distribution of homology clusters and annotations across splits.

---

## B) Comprimento das proteínas por split

```text
train mean 367.6 | median 299
val   mean 341.8 | median 281
test  mean 361.8 | median 292
```

As distribuições são **parecidas**, mas não idênticas.

* train é um pouco mais longo
* val é o mais curto
* test fica no meio, mais próximo de train que de val
* a média é bem maior que a mediana em todos os splits -> cauda longa de proteínas grandes

Isso é biologicamente plausível e esperado. O mais importante aqui é que:

* não há colapso grotesco, tipo test só com proteínas minúsculas ou gigantes
* as medianas estão relativamente próximas: 281–299 aa

Mas, a validação ficou com proteínas um pouco mais curtas, em média e mediana. 
Métodos como mutation sensitivity e talvez attention podem responder diferente ao comprimento.

> Protein length distributions were broadly comparable across splits, with median lengths ranging from 281 to 299 amino acids, although the validation split showed slightly shorter proteins on average.

Se alguma métrica mudar no val mas não no test, pode ser partly comprimento e não só generalização.

---

## C) Estrutura dos clusters

```text
Total clusters: 77,114
Singleton clusters: 42,512 (55.1%)
Largest cluster: 1,670 proteins
```

Mais da metade dos clusters tem **uma única proteína**. Isso mostra duas coisas ao mesmo tempo:

1. o dataset tem **muita diversidade**
2. o threshold de homologia foi suficientemente rígido para não fundir tudo em supergrupos artificiais

Sobre o maior cluster, não é absurdo para esse tamanho de dataset.
Mas mostra que ainda existem famílias muito representadas. Essas famílias grandes podem influenciar treino e contagens agregadas.

* o conjunto tem uma longa cauda de proteínas "isoladas"
* mas também algumas famílias grandes e super-representadas
* então é importante reportar resultados por proteína e talvez normalizar algumas análises, para famílias grandes não dominarem

> Clustering at the selected homology threshold produced 77,114 clusters, of which 55.1% were singletons, indicating substantial sequence diversity, while the largest cluster contained 1,670 proteins.

---

## D) Sanity checks / leakage

```text
cluster_leakage_count = 0
dup_accessions_proteins = 0
accessions_in_multiple_splits = 0
dup_rows_proteins = 0
dup_rows_annotations = 74068
orphan_annotations = 0
```

Boa. Mas,

```text
dup_rows_annotations = 74068
```

Pode ser uma destas coisas:

1. duplicatas reais no arquivo de annotations
2. linhas idênticas biologicamente plausíveis, mas repetidas por diferentes evidências
3. erro de parsing do GFF
4. duplicação causada por merge/join

> To avoid inflating annotation counts due to multiple evidence sources describing the same residue-level feature, duplicate annotations were collapsed based on protein accession, feature type, and residue coordinates.

Após gerar o annotation_dedup.tsv, temos 0 duplicatas.

---

## E) Distribuição de feature types por split

```text
Feature type       -  train,  val,  test
Active site        - 124907, 25672, 28766
Beta strand        - 274345, 51246, 53184
Binding site       - 864199, 174325, 174353
Helix              - 268896, 50174, 55753
Region             - 229633, 43284, 56842
Topological domain - 113237, 18608, 23006
Modified residue   - 179048, 31958, 41883
...
Transmembrane      - 268787, 53431, 63361
```

As contagens parecem razoavelmente estáveis em escala grande, mas há **desbalanceamentos específicos** entre feature types.

Exemplos visíveis:

* **Helix**: test tem mais que val de forma perceptível
* **Region**: test tem muito mais que val
* **Topological domain**: test > val
* **Transmembrane**: test > val
* **Modified residue**: test > val também

Isso sugere que o split ficou bem balanceado no total, **mas não perfeitamente estratificado por tipo de anotação**.
Em split por homologia, algum desbalanceamento por feature é esperado, porque famílias de proteínas concentram certos padrões biológicos.

* globalmente balanceados
* com alguma heterogeneidade residual por feature type

> Although global split proportions were well preserved, some annotation categories showed moderate deviations across splits, likely reflecting family-specific biological enrichment retained by homology-aware partitioning.

---

## F) O "max annotation proportion deviation from expected (using actual annotation split proportions): 0.092"

Sequence uncertainty: 0.092
Non-standard residue: 0.086

São:
* raros
* meio "ruído biológico"    
* não centrais para o problema

Non-adjacent residues: 0.066
Site: 0.042
* também não são o foco principal

Disulfide bond: 0.039
* é biologicamente relevante
* mas o desvio é bem pequeno

As features mais interessantes para o problema aparecem ter desvio menor, o que é bom.

---

## G) Pontos fortes claros

### a) split por homologia funcionou

### b) os splits ficaram bem balanceados em volume

### c) há diversidade substancial

### d) o dataset é biologicamente heterogêneo

## H) Pontos que merecem cautela

### a) leve diferença de comprimento em val

### b) composição por feature type não é perfeitamente uniforme

### c) alguns desvios pequenos na métrica de balanceamento por feature type

## I) Resumo final

> The homology-aware partitioning produced balanced train, validation, and test sets containing 402254, 86972, and 85401 proteins, respectively, distributed across 77114 sequence-homology clusters. No cluster leakage or cross-split accession duplication was detected. Protein length distributions were broadly comparable across splits, with median lengths ranging from 281 to 299 amino acids. More than half of the clusters (55.1%) were singletons, indicating substantial sequence diversity. While overall annotation counts were proportional across splits, some feature categories exhibited moderate split-specific deviations, likely due to family-level biological enrichment.
