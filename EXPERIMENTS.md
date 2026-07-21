# Experimentos

Lista abstrata dos experimentos para o paper. Cada bloco isola um eixo; todos compartilham o mesmo setup (datasets, CV, modelo, métricas).

**Setup comum:** classificação de texto com Transformer (RoBERTa-base), cross-validation, macro-F1 + tempo de treino + fração de dados usada.

**Contribuição central:** curriculum learning guiado pelas métricas BIOIS (redundância, ruído, entropia) — não apenas seleção de dados.

---

## 1. Baseline — sem IS, sem CL

Treino padrão no conjunto completo.

- **Objetivo:** referência de acurácia e custo computacional.
- **Modo:** `raw`

---

## 2. Apenas instance selection (BIOIS)

Redução do dataset por redundância e ruído, sem curriculum.

- **Objetivo:** medir o efeito isolado da seleção de instâncias.
- **Modo:** `is`
- **Ablação (opcional):** redundância-only vs. ruído-only vs. ambos.

---

## 3. Apenas curriculum learning (sinais BIOIS)

Organização do treino em fases (fácil → difícil) usando as métricas BIOIS como sinal de dificuldade, sem reduzir o dataset.

- **Objetivo:** medir o efeito isolado do curriculum com o sinal proposto.
- **Modo:** `cl`
- **Variantes internas:** BIOIS-discrete (clean → diverse → hard), SPCL soft, SPCL loss

---

## 4. Instance selection + curriculum learning (método proposto)

BIOIS reduz o dataset e o curriculum opera sobre o subset.

- **Objetivo:** resultado principal — eficiência com F1 competitivo.
- **Modo:** `is_cl`
- **Variantes de CL:** discrete, SPCL soft, SPCL loss (mesma IS, schedules diferentes)

---

## 5. Baselines de curriculum learning (literatura)

Comparação com métodos de CL que **pacingam ou ponderam instâncias** por sinais de dificuldade alternativos — sem as métricas bi-objetivas do BIOIS.

- **Objetivo:** mostrar que CL guiado por redundância + ruído + entropia (BIOIS) supera CL recente baseado em heurísticas, dinâmica de treino ou confiança univariada.
- **Escopo:** mesmo orçamento de treino e mesmo scheduler de fases quando aplicável; comparar `cl`/`is_cl` (BIOIS) vs. cada baseline.

### Fundacionais (referência histórica)

| Baseline | Sinal de dificuldade | Status no repo |
|----------|---------------------|----------------|
| Confidence-paced CL (Bengio et al., 2009) | confiança no rótulo (classificador fraco) | `b1` |
| SPCL canônico (Jiang et al., 2015) | região Ψ + prior de confiabilidade | `spcl_loss` |

### NLP / fine-tuning — prioridade para o paper

Métodos desenhados para PLMs em tarefas de NLU (classificação, NLI, etc.):

| Baseline | Sinal de dificuldade | Referência | Status |
|----------|---------------------|------------|--------|
| Cross-Review + Annealing (AnnealCR) | votos de modelos-teacher em subsets do treino | Xu et al., ACL 2020 | a implementar |
| Training Dynamics CL (AnnealTD) | estatísticas de incerteza durante o treino (easy / ambiguous / hard) | Christopoulou et al., EMNLP 2022 | a implementar |
| Competence-based CL | competência crescente do modelo (função de época) | Platanios et al., 2019 | a implementar |
| CL-LRC | comprimento + raridade + comprehensibility (LRC) | Ranaldi et al., RANLP 2023 | a implementar |
| Self-adaptive CL | dificuldade predita pelo próprio PLM (confiança do modelo) | ACL SRW 2025 | a implementar |
| SPDCL | dificuldade linguística + features dependentes da tarefa (desbalanceamento) | arXiv 2210.14724 | a implementar |

### Heurísticas e controles negativos

Importante incluir porque Soviany et al. (ACL Insights 2022) mostram que muitas curricula baseadas em heurísticas **não superam amostragem aleatória** em BERT/T5 — servem de controle para posicionar o BIOIS:

| Baseline | Sinal de dificuldade | Referência |
|----------|---------------------|------------|
| Length-based | comprimento da sequência (proxy de complexidade) | Platanios et al., 2019 |
| Loss-based pacing | loss de treino como dificuldade | padrão em SPL |
| Perplexity / TF-IDF rank | complexidade lexical estática | Soviany et al., 2022 |

### Opcional (appendix ou extensão)

| Baseline | Sinal de dificuldade | Nota |
|----------|---------------------|------|
| Influence-driven CL | influência de cada exemplo no loss de outros | foco em pré-treino; arXiv 2025 |
| Pacing contínuo (SPCL soft) | soft-pacing sobre sinais BIOIS | já no repo como variante interna |

**Comparações-chave para o paper:**

- `is_cl` vs. **AnnealCR** e **AnnealTD** — BIOIS vs. os CL mais citados em NLU fine-tuning
- `is_cl` vs. **self-adaptive PLM** — sinal bi-objetivo externo vs. dificuldade auto-reportada pelo Transformer
- `cl` (BIOIS) vs. **length / loss** — BIOIS supera heurísticas que a literatura recente considera fracas
- `is_cl` vs. `b1` — ganho além de confidence-paced clássico

---

## 6. Análise (pós-experimentos)

Não são runs novos de treino; derivam dos resultados acima.

- Fronteira eficiência: macro-F1 vs. tempo de treino
- Impacto em classes raras
- Quando o sinal do classificador fraco transfere para o Transformer
- Estudos de caso: exemplos removidos vs. mantidos

---

## Matriz resumida

| Experimento              | IS | CL | Sinal de dificuldade | Papel no paper          |
|--------------------------|----|----|----------------------|-------------------------|
| Baseline                 | ✗  | ✗  | —                    | Referência              |
| Only IS                  | ✓  | ✗  | —                    | Ablação IS              |
| Only CL (BIOIS)          | ✗  | ✓  | BIOIS                | Ablação CL              |
| IS + CL (proposto)       | ✓  | ✓  | BIOIS                | **Resultado principal** |
| CL SOTA baselines        | ✗/✓| ✓  | TD, AnnealCR, LRC, PLM… | Comparação com literatura NLP |
| Análise                  | —  | —  | —                    | Figuras e discussão     |

---

## Prioridade de execução

1. Baseline + Only IS + Only CL + IS+CL (fatorial 2²)
2. IS+CL com variantes de CL (discrete, SPCL soft, SPCL loss)
3. Baselines NLP: AnnealCR (ACL 2020) → AnnealTD (EMNLP 2022) → self-adaptive PLM → heurísticas (length/loss)
4. Análises
