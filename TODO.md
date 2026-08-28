# TODO — Paper ACL (1 mês)

Objetivo do mês: ter um **pacote demonstrável para supervisores** — não submissão ACL completa, mas evidência sólida de que a contribuição é publicável, com narrativa clara, experimentos reproduzíveis e comparações mínimas com a literatura.

**Tese central (proposta):** curriculum learning para fine-tuning de PLMs guiado pelos sinais bi-objetivos do BIOIS (redundância + ruído/entropia) melhora a **fronteira eficiência–qualidade** em classificação de texto, indo além de CL baseado em confiança univariada ou heurísticas fracas.

**Estado atual do repo (baseline honesta):**
- Pipeline `bio-experiment` + matriz IS×CL implementada (`raw`, `is`, `cl`, `is_cl`, `b1`).
- Variantes de CL: `biois_discrete`, `spcl_soft`, `spcl_loss`.
- Resultados parciais em WebKB e Reuters90 (`results/*20260711-022935*`) — **úteis para diagnóstico**, mas datasets pequenos (~5K–13K docs) limitam o ganho visível de IS (redução de ~30% ≈ poucos mil exemplos).
- **Hipótese central para datasets grandes:** em `yelp_2013` (335K), `agnews` (128K) e `medline` (860K), a redução BIOIS (~30–40% no TOIS/SIGIR) deve aparecer em `data_efficiency`, `train_time_s` e `compute_proxy` — é daí que vem o argumento de eficiência para supervisores.
- Config pronta: `experiments/campaigns/large_datasets_5cv.yaml`.
- **Alerta:** em WebKB, `raw` (macro-F1 ≈ 0.83) > `is` ≈ 0.82 > `cl` ≈ 0.79 > `is_cl` ≈ 0.75 — ganho de eficiência existe, mas **perda de F1 precisa ser explicada ou corrigida**; em datasets grandes o trade-off pode inverter (menos overfitting em ruído/redundância).
- Baselines NLP (AnnealCR, AnnealTD, etc.) ainda **não implementados** (`EXPERIMENTS.md`).

---

## Cronograma sugerido (4 semanas)

| Semana | Foco | Entregável para supervisores |
|--------|------|------------------------------|
| 1 | Enquadramento + diagnóstico | 1-pager de contribuição + plano experimental; hipóteses sobre queda de F1 em datasets pequenos vs. ganho em grandes |
| 2 | Experimento fatorial núcleo | Tabela 2² em **≥2 datasets grandes** (Tier L) + 1 dataset médio de controle; IC 95% |
| 3 | Baselines + análises | ≥1 baseline NLP + **figura Pareto com escala log de tempo** (grandes) + 1 ablação |
| 4 | Narrativa | Slides ou seção draft (intro + método + resultados preliminares); claim ancorado em redução % e speedup |

---

## Semana 1 — Enquadramento acadêmico e diagnóstico

### 1.1 Definir a contribuição em uma frase testável
- [ ] Escrever **claim principal** e **claims secundários** (máx. 3), cada um com métrica e comparação explícita.
  - Exemplo de claim principal: *"IS+CL com sinais BIOIS atinge macro-F1 dentro de δ do treino completo em agnews/yelp_2013/medline usando ≤70% dos dados e ≥1.3× speedup em tempo de treino."*
  - Exemplo secundário: *"O sinal bi-objetivo supera confidence-paced CL (Bengio 2009) e heurísticas length/loss em ≥2 de 3 datasets Tier L."*
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
- [ ] **Smoke em 1 dataset grande** antes do batch (1 fold, LR ou RoBERTa com `--folds 0`):
  ```sh
  uv run bio-experiment experiments/campaigns/large_datasets_5cv.yaml --dataset agnews --folds 0
  ```

**Validação acadêmica:** um resultado onde o método proposto perde em todas as métricas em todos os datasets é fatal. É aceitável **perder F1 absoluto em datasets pequenos** se ganhar em **Pareto eficiência em datasets grandes** — onde a redução é material (centenas de milhares de exemplos removidos).

### 1.4 Estratégia de datasets — pequenos para diagnóstico, grandes para o claim

O claim de eficiência **depende de escala**. BIOIS com β=0.3, θ=0.2 remove ~30–40% do treino (Cunha et al., TOIS/SIGIR); em WebKB isso são ~2K docs — em MEDLINE são ~250K. A narrativa do mês deve ancorar resultados nos **Tier L**.

**Esquema obrigatório (Zenodo / atcBench):** `texts.txt` + `score.txt` + `splits/split_{5,10}.pkl` + `tfidf/*.gz` — mesmo layout de `download_datasets.py`. Referência: [atcBench](https://github.com/waashk/atcBench) (22 datasets Li et al. 2022 + 3 grandes; Cunha et al., arXiv:2504.01930).

#### Tier S — diagnóstico rápido (já no repo, 10-fold)
| Dataset | \|train\| aprox. | \|C\| | Papel no mês |
|---------|----------------|------|--------------|
| `webkb` | 8K | 7 | Debug β/θ, variantes CL, ablação 2² |
| `reuters90` | 13K | 90 | Classes raras, desbalanceamento extremo |

Completar folds faltantes do batch `20260711-022935`; **não** usar como única evidência de eficiência.

#### Tier L — núcleo do claim (obrigatório, 5-fold)
| Dataset | \|train\| aprox. | \|C\| | Skew | Literatura / por quê |
|---------|----------------|------|------|----------------------|
| `agnews` | 128K | 4 | balanceado | Zhang et al. 2015; baseline CL/IS em SIGIR'23 e TOIS; tarefa fácil → isola efeito de redução |
| `yelp_2013` | 335K | 6 | desbalanceado | Zhang et al. 2015; reviews longas; redundância lexical alta |
| `medline` | 860K | 7 | extremamente desbalanceado | Domínio biomédico; Cunha cita 80 GPU-h XLNet — **melhor cenário para speedup** |

Config: `experiments/large_datasets_roberta_base_5cv.yaml`. Ordem sugerida: `agnews` (validar pipeline) → `yelp_2013` → `medline`.

#### Tier L+ — extensão grande (mesmo esquema Zenodo; adicionar ao repo se houver tempo)
| Dataset | \|train\| aprox. | \|C\| | Zenodo | Por quê incluir |
|---------|----------------|------|--------|-----------------|
| `imdb_reviews` | 348K | 10 | [5257310](https://doi.org/10.5281/zenodo.5257310) | Sentimento multi-classe; complementa topic dos outros grandes |
| `sogou` | 510K | 5 | [5259056](https://doi.org/10.5281/zenodo.5259056) | Zhang et al. 2015; notícias; denso (588 docs/classe); testa generalização fora de inglês "padrão" |

Prioridade para ACL completa: **1 dos dois** basta além do Tier L.

#### Tier M — ponte médio→grande e diversidade (já no atcBench; baixar via Zenodo)
| Dataset | \|train\| aprox. | \|C\| | Zenodo | Papel |
|---------|----------------|------|--------|-------|
| `dblp` | 38K | 10 | [7555264](https://doi.org/10.5281/zenodo.7555264) | Prototipar β/θ com custo menor que Tier L |
| `books` | 34K | 8 | [7555256](https://doi.org/10.5281/zenodo.7555256) | Textos longos (densidade 269); bom para ruído vs. redundância |
| `acm` | 25K | 11 | [7555249](https://doi.org/10.5281/zenodo.7555249) | Tópico acadêmico; entre médio e grande |
| `ohsumed` | 18K | 23 | já no repo | Muitas classes; já listado em `tier2_base` |
| `wos-11967` | 12K | 33 | [7555385](https://doi.org/10.5281/zenodo.7555385) | Kowsari et al. 2017; 33 classes balanceadas |

**Não priorizar no mês:** `mpqa`, `twitter`, `sst1` — pequenos demais para evidenciar redução; úteis só como sanity check.

#### Seleção mínima para supervisores (mês 1)
- [ ] **Tier L (3):** `agnews`, `yelp_2013`, `medline` — matriz 2² completa, 5-fold.
- [ ] **Tier S (1):** `webkb` — diagnóstico + ablação β/θ (já parcialmente feito).
- [ ] **Tier M (1, opcional):** `dblp` ou `books` — validar que tuning de WebKB transfere antes de gastar GPU em `medline`.

Registrar em `experiments/paper_core.yaml` (criar) com: tiers, seeds, folds (5 para L, 10 para S), modos fixos, e `β`/`θ` por tier se necessário.

---

## Semana 2 — Experimento fatorial núcleo (2²) — foco em datasets grandes

### 2.0 Pré-requisitos
- [ ] Baixar Tier L: `uv run python download_datasets.py agnews yelp_2013 medline`
- [ ] Confirmar `instance_selection.json` reporta `reduction_rate` ≥ 25% em pelo menos 2 dos 3 grandes (senão revisar β/θ antes do batch).

### 2.1 Matriz principal IS × CL (prioridade Tier L)
- [ ] Executar `raw`, `is`, `cl`, `is_cl` com **mesmo** `roberta-base`, mesmos folds, mesmas épocas totais:
  ```sh
  # Núcleo — datasets grandes (5-fold)
  uv run bio-experiment experiments/campaigns/large_datasets_5cv.yaml --dataset agnews
  uv run bio-experiment experiments/campaigns/large_datasets_5cv.yaml --dataset yelp_2013
  uv run bio-experiment experiments/campaigns/large_datasets_5cv.yaml --dataset medline

  # Diagnóstico / ablação — dataset pequeno (10-fold, se faltar)
  uv run bio-experiment experiments/tier2_base.yaml --dataset webkb
  ```
- [ ] **Métricas obrigatórias em datasets grandes:** além de `macro_f1`, registrar e comparar:
  - `data_efficiency` (fração de dados usada após IS)
  - `train_time_s` e `compute_proxy` (argumento principal para supervisores)
  - `reduction_rate` de `instance_selection.json` por fold
- [ ] Garantir que **orçamento de treino é comparável**:
  - `raw`/`is`: `--epochs 6`
  - `cl`/`is_cl`: `--epochs-per-phase 2` × 3 fases = 6 épocas
  - Registrar `compute_proxy` e `train_time_s` em toda comparação.

**Validação acadêmica:** comparação injusta (menos épocas para baseline ou mais dados para o proposto) é motivo clássico de rejeição. Em datasets grandes, revisores esperam ver **speedup** explícito (ex.: "1.4× menos tempo com F1 dentro de δ").

### 2.2 Ablação mínima
- [ ] **Só redundância:** `beta>0, theta=0` (modo `is` com flags; pode exigir pequena alteração no CLI se não existir).
- [ ] **Só ruído:** `beta=0, theta>0`.
- [ ] **CL sem IS:** modo `cl` (já existe).
- [ ] **IS sem CL:** modo `is` (já existe).

**Onde rodar:** ablação completa em `webkb` (rápido); **confirmar tendência** em 1 fold de `agnews` ou `yelp_2013` (redundância vs. ruído em escala).

Objetivo: mostrar que ganho vem da **combinação** e que cada sinal contribui — especialmente onde há massa de dados redundantes/ruidosos.

### 2.3 Variantes de curriculum (subconjunto)
- [ ] Comparar `biois_discrete` vs `spcl_soft` vs melhor `spcl_loss` scheme em **1 dataset** (WebKB).
- [ ] Escolher **uma** variante para o restante do mês (evitar explosão combinatória).

### 2.4 Baseline mínimo da literatura
- [ ] Rodar `b1` (Bengio) nos mesmos datasets:
  ```sh
  uv run bio-experiment experiments/tier2_base_baselines.yaml
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
- [ ] **Figura Pareto (2 painéis):** (a) Tier S `webkb`; (b) Tier L `agnews`+`yelp_2013`+`medline` — eixo X = `train_time_s` ou `compute_proxy` (**escala log** nos grandes), eixo Y = `macro_f1`; pontos = modos; usar `analysis/analysis.ipynb`.
- [ ] **Tabela de redução:** dataset × `n_before` × `n_after` × `reduction_rate` × Δ tempo × Δ macro-F1 (dados de `instance_selection.json` + `summary.csv`).
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
- [ ] +Tier L+ (`imdb_reviews`, `sogou`) e +Tier M (`dblp`, `wos-11967`) no `download_datasets.py`
- [ ] +2 baselines NLP, roberta-large ou outro PLM, significância completa, human evaluation (se claim for qualitativo), código público anonimizado, página de ética (datasets públicos — declaração curta)

---

## Checklist — o que revisores ACL vão atacar

| Risco de rejeição | Mitigação neste mês | Pode ficar para depois |
|-------------------|---------------------|-------------------------|
| Novidade incremental ("IS + CL óbvio") | Ablação 2² + sinal bi-objetivo vs confidence/length | Teoria formal |
| Baselines fracas | `b1` + 1 NLP + random/length | AnnealCR + AnnealTD + self-adaptive PLM |
| Comparação injusta | Igualar épocas totais; reportar tempo e % dados | Matched compute budget com early-stop |
| Poucos datasets | Tier L (3 grandes) + 1 pequeno de diagnóstico | Tier L+ e Tier M completos (atcBench) |
| Resultado negativo (F1 cai) | Narrativa Pareto em **grandes** + diagnóstico em pequenos + tuning β/θ | SOTA absoluto em F1 |
| Claim de eficiência sem escala | `medline`/`yelp_2013` com % redução e speedup reportados | `sogou`, `imdb_reviews` |
| Falta de significância | IC 95% + teste pareado nos folds | Correção Bonferroni |
| Confusão com SIGIR'23 / TOIS | Parágrafo de posicionamento explícito | — |
| Heurística (Soviany 2022) | Controles length/loss/random | — |
| Reprodutibilidade | YAML + `config.json` + commit hash | Artifact ACL |
| Escrita em inglês | Draft pode ser PT para supervisores; paper final EN | — |

---

## Comandos úteis (referência rápida)

```sh
# Smoke antes de batch grande
uv run bio-experiment experiments/smoke.yaml

# Download datasets grandes
uv run python download_datasets.py agnews yelp_2013 medline

# Núcleo fatorial — Tier L (5-fold)
uv run bio-experiment experiments/campaigns/large_datasets_5cv.yaml --dataset agnews
uv run bio-experiment experiments/campaigns/large_datasets_5cv.yaml --dataset yelp_2013
uv run bio-experiment experiments/campaigns/large_datasets_5cv.yaml --dataset medline

# Diagnóstico — Tier S (10-fold)
uv run bio-experiment experiments/tier2_base.yaml --dataset webkb

# Docker batch datasets grandes
uv run bio-experiment experiments/campaigns/large_datasets_5cv.yaml

# Comparar batch Docker
uv run python summary.py --compare --run-prefix <PREFIX> \
    --datasets agnews yelp_2013 medline webkb reuters90 \
    --output summary-compare.xlsx
```

---

## Definição de "pronto para mostrar aos supervisores"

Considerar o mês bem-sucedido se houver:

1. **1-pager** com claim, related work (5–7 refs) e posicionamento vs SIGIR'23 — claim ancorado em **redução % + speedup em datasets grandes**.
2. **Tabela principal** (≥3 datasets Tier L × 4 modos + `b1`) com IC 95%, colunas de `data_efficiency` e tempo.
3. **1 figura Pareto** eficiência vs macro-F1 (painel com escala log nos grandes).
4. **1 ablação** (redundância vs ruído vs ambos) — pelo menos confirmada em 1 dataset grande.
5. **Diagnóstico honesto** dos casos em que `is_cl` perde para `raw` em Tier S, e quando ganha em Tier L.
6. **Roadmap** claro do que falta para submissão ACL (Tier L+, baselines NLP, +2–3 meses).

---

## Referências iniciais (BibTeX a completar na semana 1)

- Bengio et al., ICML 2009 — curriculum by difficulty.
- Jiang et al., AAAI 2015 — Self-Paced Curriculum Learning.
- Xu et al., ACL 2020 — AnnealCR.
- Christopoulou et al., EMNLP 2022 — curriculum from training dynamics.
- Soviany et al., ACL 2022 — curriculum in vision/NLP often fails vs random.
- Platanios et al., 2019 — competence-based CL.
- Cunha et al., SIGIR 2023 — confidence-based IS for Transformers.
- Cunha et al., TOIS 2024 — BIOIS bi-objective IS (redundância + ruído).
- Li et al., TIST 2022 — survey ATC; suíte de 22 datasets de referência.
- Cunha et al., arXiv:2504.01930 — atcBench (22+3 datasets, Zenodo).
- Zhang et al., 2015 — AG News, Yelp 2013 (Character-level CNNs).
- Kowsari et al., ICMLA 2017 — Web of Science (WOS).
