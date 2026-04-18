1. Introdução: PLMs aprendem biologia implicitamente.
   Pergunta: é possível tornar isso explícito e mensurável?

2. Dataset e SAE: 50k proteínas, split homológico,
   SAE sobre ESM-2 650M, ~N features ativas.

3. Alinhamento resíduo-nível (Fase 1+2):
   - UniProt + Pfam + BioLiP
   - propriedades físico-químicas
   - métricas: AUPRC, odds ratio, enrichment
   - baseline: permutação circular

4. Convergência multi-evidência (Fase 3):
   - features SAE × attention heads
   - quais features têm convergência forte?
   - taxonomia emergente das features

5. Perturbação (Fase 4):
   - top-k features candidatas
   - evidência causal

6. Discussão: o que o ESM-2 aprendeu?
   Limitações: viés de anotação, features mortas.