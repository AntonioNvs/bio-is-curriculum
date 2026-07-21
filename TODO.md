# TODO — Paper ACL (1 mês)

Objetivo do mês: ter um **pacote demonstrável para supervisores** — não submissão ACL completa, mas evidência sólida de que a contribuição é publicável, com narrativa clara, experimentos reproduzíveis e comparações mínimas com a literatura.

**Tese central (proposta):** curriculum learning para fine-tuning de PLMs guiado pelos sinais bi-objetivos do BIOIS (redundância + ruído/entropia) melhora a **fronteira eficiência–qualidade** em classificação de texto, indo além de CL baseado em confiança univariada ou heurísticas fracas.

**Estado atual do repo (baseline honesta):**
- Pipeline `run.py` + matriz IS×CL implementada (`raw`, `is`, `cl`, `is_cl`, `b1`).
- Variantes de CL: `biois_discrete`, `spcl_soft`, `spcl_loss`.
- Resultados parciais em WebKB e Reuters90 (`results/*20260711-022935*`).
- **Alerta:** em WebKB, `raw` (macro-F1 ≈ 0.83) > `is` ≈ 0.82 > `cl` ≈ 0.79 > `is_cl` ≈ 0.75 — ganho de eficiência existe, mas **perda de F1 precisa ser explicada ou corrigida** antes de apresentar aos supervisores.
- Baselines NLP (AnnealCR, AnnealTD, etc.) ainda **não implementados** (`EXPERIMENTS.md`).

---

## Cronograma sugerido (4 semanas)

| Semana | Foco | Entregável para supervisores |
|--------|------|------------------------------|
| 1 | Enquadramento + diagnóstico | 1-pager de contribuição + plano experimental; hipóteses sobre queda de F1 |
| 2 | Experimento fatorial núcleo | Tabela 2² (raw/is/cl/is_cl) em 3–4 datasets com IC 95% |
| 3 | Baselines + análises | ≥1 baseline NLP + figura eficiência–F1 + 1 ablação |
| 4 | Narrativa | Slides ou seção draft (intro + método + resultados preliminares) |

---

## Semana 1 — Enquadramento acadêmico e diagnóstico

### 1.1 Definir a contribuição em uma frase testável
- [ ] Escrever **claim principal** e **claims secundários** (máx. 3), cada um com métrica e comparação explícita.
  - Exemplo de claim principal: *"IS+CL com sinais BIOIS atinge macro-F1 dentro de δ do treino completo usando ≤50% dos dados e ≤60% do tempo."*
  - Exemplo secundário: *"O sinal bi-objetivo supera confidence-paced CL (Bengio 2009) e heurísticas length/loss em ≥2 de 4 datasets."*
- [ ] Decidir se o paper enfatiza **eficiência** (Pareto tempo×F1), **robustez** (hard-slice, classes raras) ou **ambos** — isso orienta figuras e texto.

**Validação acadêmica:** revisores ACL rejeitam papers cujo claim é vago ("melhora o treino") ou não mensurável. Cada claim precisa de baseline, métrica e hipótese nula.

### 1.2 Revisão de literatura — pontos de busca

Use estas queries (Google Scholar, ACL Anthology, Semantic Scholar). Para cada linha, registrar: método, sinal de dificuldade, modelo, tarefas, conclusão sobre eficácia.

| Área | Queries / termos | O que validar |
|------|------------------|---------------|
| **CL clássico** | `"curriculum learning" Bengio 2009`, `"self-paced learning" Jiang SPCL` | Diferenciar SPCL (loss reweighting) do pacing discreto BIOIS; citar Eqs. relevantes se usar `spcl_loss`. |
| **CL em NLP / PLM** | `"curriculum learning" BERT fine-tuning`, `AnnealCR ACL 2020`, `training dynamics curriculum EMNLP 2022`, `competence-based curriculum Platanios 2019` | Quais métodos são SOTA em NLU; se usam o mesmo orçamento de épocas/steps. |
| **CL negativo / crítico** | `Soviany curriculum insights ACL 2022`, `"curriculum learning" "does not" BERT` | **Obrigatório:** posicionar por que BIOIS não é "mais uma heurística". Incluir controles length/loss. |
| **Instance selection** | `confidence-based instance selection SIGIR 2023 Cunha`, `bi-objective instance selection`, `data pruning language models` | Separar contribuição **IS** (TOIS, em revisão) da contribuição **IS+CL para PLM** (este paper). |
| **Eficiência em fine-tune** | `data-efficient fine-tuning`, `subset training transformer classification` | Mostrar que eficiência não é só "menos épocas" — incluir `data_efficiency` e `compute_proxy`. |
| **Desbalanceamento** | `class imbalance text classification`, `rare class curriculum` | Justificar `class_balanced_loss` e upsampling de classes raras no pipeline. |

**Artigos âncora (ler, não só citar):**
1. Bengio et al. (ICML 2009) — confidence-paced CL → baseline `b1` já no repo.
2. Jiang et al. (AAAI 2015) SPCL → `spcl_loss`.
3. Xu et al. (ACL 2020) AnnealCR — **baseline prioritária**.
4. Christopoulou et al. (EMNLP 2022) AnnealTD — **baseline prioritária**.
5. Soviany et al. (ACL Insights 2022) — controles negativos.
6. Cunha et al. (SIGIR 2023) — IS com Transformers (trabalho anterior do grupo).
7. Platanios et al. (2019) — competence + length heuristic.

**O que precisa estar explícito na related work (evita rejeição por "falta de comparação"):**
- [ ] Tabela "posicionamento": método × usa PLM? × sinal de dificuldade × reduz dados? × pacing discreto/contínuo?
- [ ] Parágrafo "Por que não é redundante com SIGIR'23": lá é IS para eficiência; aqui é **reuso dos mesmos sinais para ordenar dificuldade no curriculum** durante fine-tune.
- [ ] Parágrafo "Por que não é só SPCL": SPCL usa loss do modelo em treino; BIOIS usa sinais **pré-treino** (classificador fraco) desacoplados do PLM — hipótese: generaliza melhor no início do fine-tune.

### 1.3 Diagnóstico dos resultados atuais (WebKB / Reuters90)
- [ ] Consolidar resultados existentes:
  ```sh
  uv run python summary.py --compare --run-prefix 20260711-022935 \
      --datasets webkb reuters90 \
      --output summary-compare-20260711-022935.xlsx
  ```
- [ ] Investigar por que `is_cl` < `raw` em macro-F1:
  - Hiperparâmetros `beta`, `theta`, `epochs-per-phase`, `curriculum-q`?
  - Redução excessiva de classes raras (`instance_selection.json`)?
  - Classificador fraco (TF-IDF+LR) mal calibrado para RoBERTa?
- [ ] Rodar grid mínimo de sensibilidade (1 dataset, 2 folds):
  - `beta` ∈ {0.1, 0.2, 0.3}, `theta` ∈ {0.1, 0.2, 0.3}
  - `epochs-per-phase` ∈ {1, 2, 3}
  - `curriculum-method` ∈ {biois_discrete, spcl_soft}
- [ ] Documentar conclusão: "com config X, trade-off aceitável" ou "método só vence em cenário Y (dataset grande / muitas classes)".

**Validação acadêmica:** um resultado onde o método proposto perde em todas as métricas em todos os datasets é fatal. É aceitável **perder F1 absoluto** se ganhar em **Pareto eficiência** com análise honesta de quando isso vale a pena.

### 1.4 Escolher datasets do núcleo (3–4 para o mês)
- [ ] **Tier A (obrigatório):** `webkb`, `reuters90` — já têm runs; completar folds faltantes.
- [ ] **Tier B (escolher 1–2):** `ohsumed` (muitas classes, desbalanceado), `mpqa` (pequeno), `agnews` (grande, mais fácil).
- [ ] Critério: diversidade em |C|, tamanho e desbalanceamento — revisores perguntam "funciona só em um dataset?".

Registrar em `experiments/paper_core.yaml` (criar) com seeds, folds e modos fixos.

---

## Semana 2 — Experimento fatorial núcleo (2²)

### 2.1 Matriz principal IS × CL
- [ ] Executar `raw`, `is`, `cl`, `is_cl` com **mesmo** `roberta-base`, mesmos folds, mesmas épocas totais:
  ```sh
  uv run python run.py experiments/tier2_base.yaml --dataset webkb
  uv run python run.py experiments/tier2_base.yaml --dataset reuters90
  uv run python run.py experiments/tier2_base.yaml --dataset ohsumed
  ```
- [ ] Garantir que **orçamento de treino é comparável**:
  - `raw`/`is`: `--epochs 6`
  - `cl`/`is_cl`: `--epochs-per-phase 2` × 3 fases = 6 épocas
  - Registrar `compute_proxy` e `train_time_s` em toda comparação.

**Validação acadêmica:** comparação injusta (menos épocas para baseline ou mais dados para o proposto) é motivo clássico de rejeição.

### 2.2 Ablação mínima (1 dataset, CV completa)
- [ ] **Só redundância:** `beta>0, theta=0` (modo `is` com flags; pode exigir pequena alteração no CLI se não existir).
- [ ] **Só ruído:** `beta=0, theta>0`.
- [ ] **CL sem IS:** modo `cl` (já existe).
- [ ] **IS sem CL:** modo `is` (já existe).

Objetivo: mostrar que ganho vem da **combinação** e que cada sinal contribui.

### 2.3 Variantes de curriculum (subconjunto)
- [ ] Comparar `biois_discrete` vs `spcl_soft` vs melhor `spcl_loss` scheme em **1 dataset** (WebKB).
- [ ] Escolher **uma** variante para o restante do mês (evitar explosão combinatória).

### 2.4 Baseline mínimo da literatura
- [ ] Rodar `b1` (Bengio) nos mesmos datasets:
  ```sh
  uv run python run.py experiments/tier2_base_baselines.yaml
  ```

### 2.5 Agregação e testes estatísticos
- [ ] Script ou notebook: para cada par (método A, método B), **paired t-test** ou Wilcoxon nos folds sobre `macro_f1` e `efficiency_score`.
- [ ] Reportar média ± IC 95% (já em `summary.csv`) **e** p-value no material para supervisores.

**Validação acadêmica:** ACL espera significância ou, no mínimo, intervalos sobre múltiplos folds — não um único seed.

---

## Semana 3 — Baselines NLP e análises

### 3.1 Implementar 1 baseline NLP prioritária
Escolher **uma** (não as três no mês):

| Opção | Prós | Contras |
|-------|------|---------|
| **AnnealCR** (ACL 2020) | Muito citado em NLU | Mais complexo (teachers, subsets) |
| **AnnealTD** (EMNLP 2022) | Sinal de training dynamics | Requer treino parcial para estatísticas |
| **Length / loss heuristic** (Soviany) | Rápido, controle negativo forte | Fraco — bom para mostrar que BIOIS não é heurística |

- [ ] Implementar em `src/baselines/` com interface igual a `baseline1.py`.
- [ ] Registrar como `b2` (ou token em `run_experiment.py`).
- [ ] Mesmo orçamento de épocas e mesmo split.

**Validação acadêmica:** sem baseline NLP recente, revisores dirão "comparação só com Bengio 2009 é insuficiente para PLMs".

### 3.2 Controles negativos (rápidos)
- [ ] **Random curriculum:** mesma estrutura de fases, ordem aleatória de instâncias (implementação ~50 linhas).
- [ ] **Length-based pacing:** ordenar por comprimento da sequência (Platanios 2019).

Comparar com `cl` (BIOIS). Se BIOIS não bater length-based, o claim precisa ser restrito.

### 3.3 Análises (sem novos treinos pesados)
- [ ] **Figura Pareto:** eixo X = `train_time_s` ou `compute_proxy`, eixo Y = `macro_f1`; pontos = modos; usar `analysis/analysis.ipynb`.
- [ ] **Hard-slice macro-F1:** já calculado — mostrar se CL ajuda nos exemplos difíceis (quantil 0.8 de entropia).
- [ ] **Classes raras:** correlacionar `n_rare_classes_pinned` / remoção por classe (`instance_selection.json`) com Δ F1 por classe.
- [ ] **Estudo de caso qualitativo:** 5 exemplos removidos por ruído vs 5 mantidos na fase Hard — 1 página para o apêndice.

### 3.4 Verificar reprodutibilidade
- [ ] `config.json` grava commit git — confirmar que todos os runs do núcleo têm `experiment-id` único e config congelada.
- [ ] README curto "como reproduzir a tabela principal" (3 comandos).

---

## Semana 4 — Narrativa para supervisores

### 4.1 Documento de 4–6 páginas (draft interno)
- [ ] **Introdução (1 pág.):** problema (custo de fine-tune), lacuna (CL em PLM fraco vs sinais bi-objetivos externos), contribuição em bullets.
- [ ] **Método (1–1.5 pág.):** diagrama IS → sinais (r, e) → fases CL; não repetir paper TOIS inteiro — referenciar SIGIR/TOIS.
- [ ] **Experimental setup (0.5 pág.):** datasets, CV, RoBERTa-base, métricas, baselines.
- [ ] **Resultados (1.5 pág.):** tabela principal + figura Pareto + 1 ablação.
- [ ] **Discussão (0.5 pág.):** quando funciona / quando não; ameaças à validade.
- [ ] **Limitações (obrigatório):** classificador fraco TF-IDF; β/θ fixos; só roberta-base no mês.

### 4.2 Slides (10–12 slides)
- [ ] Motivação → método (1 slide com pipeline) → matriz experimental → resultado principal → Pareto → "próximos passos para ACL".

### 4.3 Lista explícita de gaps para submissão ACL real
- [ ] +3 datasets, +2 baselines NLP, roberta-large ou outro PLM, significância completa, human evaluation (se claim for qualitativo), código público anonimizado, página de ética (datasets públicos — declaração curta).

---

## Checklist — o que revisores ACL vão atacar

| Risco de rejeição | Mitigação neste mês | Pode ficar para depois |
|-------------------|---------------------|-------------------------|
| Novidade incremental ("IS + CL óbvio") | Ablação 2² + sinal bi-objetivo vs confidence/length | Teoria formal |
| Baselines fracas | `b1` + 1 NLP + random/length | AnnealCR + AnnealTD + self-adaptive PLM |
| Comparação injusta | Igualar épocas totais; reportar tempo e % dados | Matched compute budget com early-stop |
| Poucos datasets | 3–4 com perfis diferentes | 8–11 já suportados no repo |
| Resultado negativo (F1 cai) | Narrativa Pareto + diagnóstico + tuning β/θ | SOTA absoluto em F1 |
| Falta de significância | IC 95% + teste pareado nos folds | Correção Bonferroni |
| Confusão com SIGIR'23 / TOIS | Parágrafo de posicionamento explícito | — |
| Heurística (Soviany 2022) | Controles length/loss/random | — |
| Reprodutibilidade | YAML + `config.json` + commit hash | Artifact ACL |
| Escrita em inglês | Draft pode ser PT para supervisores; paper final EN | — |

---

## Comandos úteis (referência rápida)

```sh
# Smoke antes de batch grande
uv run python run.py experiments/smoke.yaml

# Núcleo fatorial
uv run python run.py experiments/tier2_base.yaml --dataset webkb

# Comparar batch Docker
uv run python summary.py --compare --run-prefix <PREFIX> \
    --datasets webkb reuters90 --output summary-compare.xlsx
```

---

## Definição de "pronto para mostrar aos supervisores"

Considerar o mês bem-sucedido se houver:

1. **1-pager** com claim, related work (5–7 refs) e posicionamento vs SIGIR'23.
2. **Tabela principal** (≥3 datasets × 4 modos + `b1`) com IC 95%.
3. **1 figura Pareto** eficiência vs macro-F1.
4. **1 ablação** (redundância vs ruído vs ambos).
5. **Diagnóstico honesto** dos casos em que `is_cl` perde para `raw`.
6. **Roadmap** claro do que falta para submissão ACL (estimativa: +2–3 meses após este mês).

---

## Referências iniciais (BibTeX a completar na semana 1)

- Bengio et al., ICML 2009 — curriculum by difficulty.
- Jiang et al., AAAI 2015 — Self-Paced Curriculum Learning.
- Xu et al., ACL 2020 — AnnealCR.
- Christopoulou et al., EMNLP 2022 — curriculum from training dynamics.
- Soviany et al., ACL 2022 — curriculum in vision/NLP often fails vs random.
- Platanios et al., 2019 — competence-based CL.
- Cunha et al., SIGIR 2023 — confidence-based IS for Transformers.
- Cunha et al., TOIS (under review) — BIOIS bi-objective IS.
