# 1) Qual modelo usar agora

O `facebook/esm2_t33_650M_UR50D` é um modelo de linguagem de proteínas amplamente usado e adequado para tarefas downstream em proteínas ([Hugging Face][1]). Já o `Rostlab/prot_bert` é mais antigo e o card enfatiza principalmente o pré-processamento específico de sequência em letras maiúsculas ([Hugging Face][2]). ESM-3 open, por sua vez, é apresentado como **generative model** condicionado por sequência/estrutura/função, o que complica seu uso se o foco imediato é **representação por resíduo + attention + mutational sensitivity** ([Hugging Face][3]). Além disso, a própria EvolutionaryScale posiciona **ESM C** como família voltada a representações e diz que ela busca melhorar representações em relação ao ESM2, enquanto o ESM3 foca geração controlável ([Hugging Face][4]).

Para **agora**:

* **PLM principal:** `facebook/esm2_t33_650M_UR50D`
* **Por quê:** embeddings por resíduo, attention acessível, ecossistema simples, comparação justa com o pipeline atual, menos atrito técnico ([Hugging Face][1])

Usar o ProtBert como **baseline secundário**, só se sobrar tempo.

---

# 2) Escopo do subset inicial

Usar **10% de cada split**. Aproximadamente:

* train: **40,225**
* val: **8,697**
* test: **8,540**

Gerar:
* `proteins_subset_10pct.tsv`
* `annotations_subset_10pct.tsv`

---

# 3) Extração de embeddings

Para cada proteína:

* accession
* split
* length
* embedding `[L, D]`

Salvar em **shards** para evitar vários arquivos pequenos ou um arquivo gigante:

* `train_shard_000.pt`
* `val_shard_000.pt`
* `test_shard_000.pt`

Cada shard contendo algo como:

```python
{
  "accessions": [...],
  "lengths": [...],
  "embeddings": [tensor_1, tensor_2, ...]
}
```

## O que extrair do modelo

Para cada proteína:

* hidden states da última camada por resíduo
* optionally mean-pooled por sequência, mas isso é secundário
* attention maps do último layer
* talvez média dos últimos 4 layers

Separação:
* embeddings + SAE em 10%
* attention em subset intermediário
* mutation sensitivity em subset pequeno, tipo:
  * 200–500 proteínas no piloto
  * depois 1k, se tudo estiver estável

## Mutation Sensivity Score

* muta posição `i`
* recalcula embedding
* passa no SAE
* mede `||z_original - z_mutado||` local ou global

Isso conversa muito bem com seu artigo.

---

# 4) Treino do SAE

* cada resíduo = uma amostra
* input dim = hidden size do ESM-2

## Entrada do SAE

Embedding por resíduo da última camada. A API/model cards enfatizam embedding por aminoácido e sequence-level output, com limite máximo de 1022 embeddings por sequência nas implementações publicadas ([Hugging Face][5]).

## Pré-processamento

* remover tokens especiais
* opcionalmente layer norm / centering global
* talvez amostragem uniforme de resíduos para treino inicial

---

# 5) Hiperparâmetros relevantes do SAE

## a) `latent_dim`!!!

* `2xD`
* `4xD`
* `8xD`
* `16xD`

Sendo `D` o hidden size do embedding.

## b) `l1_lambda`

Testar em escala log:

* `1e-5`
* `3e-5`
* `1e-4`
* `3e-4`

## c) learning rate

* `1e-3`
* `3e-4`

## d) batch size

O maior que couber sem matar a GPU.

## e) tied vs untied decoder

* começar com untied, que é mais flexível
* se tiver tempo, testar tied para ver se ajuda a regularizar

## f) activation

* **ReLU + L1**
* simples, reproduzível, rápido

---

# 6) Métricas importantes para escolher hiperparâmetros

## Bloco 1 — reconstrução

* MSE / explained variance no val

## Bloco 2 — esparsidade

* fração média de latents ativos por resíduo
* L0 médio aproximado
* % de features mortas

## Bloco 3 — utilidade interpretável

* associação entre top features SAE e annotations no val
* AUROC / AP para:
  * HELIX
  * STRAND
  * TRANSMEM
  * BINDING

## Bloco 4 — estabilidade

* a mesma latent feature aparece de forma coerente entre proteínas?
* top activations são localizadas?
* não colapsa tudo em poucas features?

---

# 7) Critério de seleção do melhor SAE

## Escolher o modelo que:

* mantém reconstrução boa
* mantém sparsity razoável
* tem poucas dead features
* **maximiza associação com annotations no val**

Queremos o SAE que **reconstrói bem o suficiente e produz latentes biologicamente úteis**.

---

# 8) Métrica de annotation recovery no val

Para cada latent feature `k`:

* score por resíduo = `z[:, k]`
* annotation mask binária por feature type

Calcular:

* AUROC
* Average Precision
* enrichment no top 1%, 5%, 10%

Depois, por annotation:

* pegar as top `m` features
* medir média no val

---

# 9) Attention nessa fase

Extrair embeddings e treinar SAE. Depois, para o mesmo subset:

* score_attn por resíduo
* comparar com annotations
* comparar com SAE

Objetivo: não bloquear o pipeline inteiro esperando attention/mutation.

---

# Referências 

[1]: https://huggingface.co/facebook/esm2_t33_650M_UR50D "facebook/esm2_t33_650M_UR50D"
[2]: https://huggingface.co/Rostlab/prot_bert "Rostlab/prot_bert"
[3]: https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1 "EvolutionaryScale/esm3-sm-open-v1"
[4]: https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12 "EvolutionaryScale/esmc-600m-2024-12"
[5]: https://huggingface.co/nvidia/esm2_t33_650M_UR50D "nvidia/esm2_t33_650M_UR50D"
