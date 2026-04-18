# 1 — Distribuição de Features

* "Binding site" domina absurdamente (~1.28M)
* Depois: Chain, Transmembrane, Beta strand, Helix
* Cauda longa de features raras

Interpretação:

* Dataset é **altamente desbalanceado**
* Features pontuais (binding) >> estruturais >> raras
* "Chain" e "Region" são **ruído estrutural (não local)**

Binding site ser dominante é curioso:

* são features **1-resíduo**
* isso favorece métodos "pontuais" (mutation)

Implicação:

* filtrar features
* ou avaliar por tipo separadamente

> The dataset exhibits a highly skewed distribution of feature types, with binding site annotations dominating, followed by structural elements such as helices and beta strands. This imbalance motivates evaluating interpretability methods per feature type.

---

# 2 — Tamanho das Features (log)

Distribuição multimodal:

* Binding / Active site → ~1 aa
* Helix / strand → ~5–20 aa
* Transmembrane → ~20–30 aa
* Domain → 100–1000 aa

Interpretação: **3 regimes completamente diferentes**:

1. **pontual (1 aa)** → binding
2. **local (5–30 aa)** → helix/strand
3. **global (100+)** → domain

> cada método vai performar diferente em cada regime

* mutation → ótimo em 1 aa
* SAE → ótimo em padrões locais
* attention → difuso, pode pegar domínios

Implicação: **separar análise por regime**.

> Feature lengths span multiple regimes, from single-residue annotations to large domains, highlighting the need to evaluate interpretability methods across different spatial scales.

---

# 3 - Cobertura por proteína

Top proteínas têm:

* 500–1500 features (!)

Essas proteínas são:

* hiper-estudadas (o 1º é P53, antígeno tumoral mais famoso)
* extremamente anotadas

Portanto, essas proteínas:

* dominam métricas globais
* enviesam avaliação

Implicação

* normalizar por proteína
* ou evitar que essas dominem

> Annotation density varies widely across proteins, with a subset of highly studied proteins exhibiting orders of magnitude more annotations, potentially biasing evaluation if not controlled.

---

# 4 — Composição do dataset (pizza)

* Binding site ~23%
* Chain ~10%
* Transmembrane ~7%
* Beta strand / helix ~6–7%
* "Others" ~24%

Dataset é uma mistura de:

* features locais úteis
* features estruturais
* features globais irrelevantes

Implicação:

* filtrar fortemente
* ou segmentar análise

> A substantial portion of annotations falls into heterogeneous or generic categories, reinforcing the need for careful feature selection.

---

# 5 — Densidade por proteína

O que mostra
Features por 100 aa:

* Beta strand / helix → ~2–4
* Binding → ~1–2
* Active site → <1

Interpretação:

* estrutura secundária é densa
* função é rara

> estrutura = padrão contínuo
> função = evento raro

Implicação:

* SAE deve capturar melhor estrutura
* mutation deve capturar melhor função

> Structural features exhibit higher density along protein sequences, whereas functional sites such as active or binding sites are sparse.

---

# 6 — Posição relativa (N → C)

Distribuição ao longo da proteína:

* Glycosylation → mais no início
* Alguns no meio
* Alguns nas pontas
* outros mais distribuídos

Interpretação:

* algumas features têm **posição preferencial**
* outras são distribuídas

Se um método "aprende posição" em vez de biologia → viés

Cuidado com modelos que capturam só posição relativa

> Certain feature types exhibit positional biases along the protein sequence, which must be considered when interpreting model-derived signals.

---

# 7 — Co-ocorrência

* Chain co-ocorre com tudo
* Binding co-ocorre com domain
* Helix co-ocorre com strand

Features não são independentes

Ex:

* SAE ativa em helix, mas na verdade era domain

Implicação:

* cuidado com causalidade
* avaliar por feature isolada

> Feature types exhibit strong co-occurrence patterns, which may confound interpretability analyses if not accounted for.

---

# 8 — Comprimento proteína x feature

* muitos pontos em 1 aa
* features médias ~10–30 aa
* poucos longos

Interpretação:

* maioria das features é pequena
* proteínas grandes não têm proporcionalmente mais features grandes

Escala de features não cresce linearmente com proteína!

Implicação:

* métodos não podem depender só de comprimento

> Feature lengths remain largely independent of protein length, with most annotations concentrated in short local regions.

---

# 9 — Comprimento das proteínas

* mediana ~295 aa
* distribuição log-normal
* dataset bem comportado biologicamente

Implicação:

* bom para batching
* mutation viável

> Protein lengths follow a typical log-normal distribution with a median around 300 amino acids.

---

# 10 — Composição de aminoácidos

* Distribuição esperada (L, A, G, etc)
* Nada estranho → dataset saudável
* sem viés forte de composição

> Amino acid composition aligns with expected biological distributions, indicating no major dataset bias.

---

# 11 — Comprimento x nº de features

* correlação positiva
* mas não linear
* proteínas maiores = mais features
* mas saturação
* densidade de anotação diminui com tamanho

Implicação:

* normalizar por comprimento pode ajudar

> The number of annotations increases with protein length, though sub-linearly, suggesting decreasing annotation density in larger proteins.

---

# CONCLUSÃO GERAL

> The dataset exhibits strong heterogeneity across feature types, scales, densities, and positional distributions. This diversity necessitates careful experimental design, including feature filtering and stratified evaluation, to ensure fair comparison of interpretability methods.

---

# PRINCIPAIS DECISÕES

## 1. Filtrar features

* HELIX
* STRAND
* TRANSMEM
* BINDING
* ACTIVE_SITE
* METAL_BINDING

---

## 2. Separar análise por tipo

* estrutural vs funcional

---

## 3. Controlar viés por proteína

* normalizar
  ou
* amostrar

---

## 4. Métricas por tipo

* uma tabela por feature

---