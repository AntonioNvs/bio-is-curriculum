# biO-IS-Curriculum

Treinamento curricular guiado por redundância e ruído para classificação de texto com Transformers.

## Instalação

```sh
uv sync
```

## 4 modos de execução (matriz IS × CL)

A flag `--mode` seleciona uma das 4 combinações de teste. Use `--model lr` para trocar RoBERTa por Regressão Logística (mais rápido para testes rápidos).

### raw — sem IS, sem CL

Fine-tuning padrão no conjunto de treino completo (modelo "cru", sem nenhum tratamento).

```sh
uv run python main.py webkb --data_dir datasets --fold 0 \
    --mode raw --epochs 6
```

### is — com IS, sem CL

BIOIS reduz o dataset; treino único no subset resultante.

```sh
uv run python main.py webkb --data_dir datasets --fold 0 \
    --mode is --epochs 6 --beta 0.3 --theta 0.2
```

### cl — sem IS, com CL

BIOIS é executado apenas para gerar os sinais (beta=0, theta=0); curriculum organiza o treino em 3 fases sobre o conjunto completo.

```sh
uv run python main.py webkb --data_dir datasets --fold 0 \
    --mode cl --epochs-per-phase 2
```

### is_cl — com IS e CL (default)

BIOIS reduz o dataset e o curriculum opera sobre o subset reduzido.

```sh
uv run python main.py webkb --data_dir datasets --fold 0 \
    --mode is_cl --epochs-per-phase 2 --beta 0.3 --theta 0.2
```

## Resultados

Cada execução gera uma pasta `results/<mode>-<timestamp>-<hex6>/` com:

| Arquivo | Conteúdo |
|---|---|
| `config.json` | Todos os hiperparâmetros, dataset, fold e commit git |
| `timings.csv` | `name, seconds` — tempos de IS, CL, treino, total |
| `phase_metrics.csv` | `phase, n_samples, n_iter, train_time_s, pred_time_s, micro_f1, macro_f1, accuracy, hard_slice_macro_f1` |
| `train_history.csv` | `phase, epoch, step, loss, lr` — uma linha por step de treino |
| `predictions_test.csv` | `idx, y_true, y_pred, pred_entropy` — predições finais no teste |

Para comparar os 4 modos, basta carregar os `phase_metrics.csv` de cada pasta.

## Opções principais

```
--mode {raw,is,cl,is_cl}        Modo de execução (default: is_cl)
--model {lr,roberta}             Modelo (default: roberta)
--hf-model                       Checkpoint HuggingFace (default: roberta-base)
--epochs                         Épocas para treino único / raw / is (default: 6)
--epochs-per-phase               Épocas por fase do curriculum (default: 2)
--batch-size                     Batch de treino (default: 16)
--max-length                     Comprimento máximo de tokenização (default: 256)
--beta / --theta                 Taxas de redução do BIOIS (default: 0.3 / 0.2)
--curriculum-beta                Peso de redundância na Fase Hard: w=1-beta*r (default: 0.5)
--results-dir                    Diretório base de resultados (default: results/)
```

TO-DO para 26/05 (entregáveis da reunião)

Código
- [X] Implementar flag `--baseline N` (índice de baseline da literatura — ver `BASELINES.md`). Começar com `--baseline 1` (Bengio et al. 2009 confidence-paced CL) reusando `_probaEveryone` do BIOIS, sem mask de ruído nem peso de redundância

Experimentos (em ordem de prioridade)
- [ ] webkb 10cv × {raw, baseline=1, is, cl, is_cl} — 50 runs, foco no 2² + ablação CL ingênuo
- [ ] reuters90 5cv × 5 modos, começando por `FOLDS="0 1 2"` (`--n-splits 5` para caber no tempo; manter `upsample_min_per_class` ligado pelas classes <5 ex.)
- [ ] mpqa 10cv × 5 modos

Análises pro slide
- [ ] Tabela macro-f1 média ± IC95 com Wilcoxon pareado (`raw` vs cada modo, `cl` vs `is_cl`, `--baseline 1` vs `is_cl`)
- [ ] Gráfico Pareto tempo×macro-f1 (`timings.csv::model_train_time_s`)
- [ ] Bar chart macro-f1 por classe ordenada por frequência, evidenciando ganho nas raras
- [ ] Dois números-âncora pro título: macro-f1 nas raras (`raw` → `is_cl`) e redução de tempo (cl → is_cl)

Adiados (não cabem em 1 dia)
- [ ] Variação de balanceamento/weighting no CL — segundo experimento, sem ele o 2² já fica limpo
- [ ] Baseline externo SOTA — comparação interessante é interna (cl vs is_cl vs cl_bengio); SOTA fica como next-step