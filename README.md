# biO-IS-Curriculum

Treinamento curricular guiado por redundância e ruído para classificação de texto com Transformers.

## Instalação

```sh
uv sync
```

### Datasets

Os datasets são baixados do Zenodo e organizados em `datasets/<nome>/`:

```sh
uv run python download_datasets.py              # todos
uv run python download_datasets.py webkb reuters90  # subset
```

Datasets suportados: `webkb`, `reuters90`, `mpqa`, `twitter`, `sst1`, `yelp_reviews`, `ohsumed`, `20ng`, `yelp_2013`, `agnews`, `medline`.

## Execução rápida (recomendado)

A interface principal é `run.py`: um comando, um YAML, sumário automático com IC 95%.

```sh
# Smoke test (2 folds, LR rápido)
uv run python run.py experiments/smoke.yaml

# Experimento completo (webkb 10-fold, todos os modos)
uv run python run.py experiments/webkb.yaml

# Override de folds
uv run python run.py experiments/tier2_base.yaml --folds 0 1 2

# Mesmo config, outro dataset
uv run python run.py experiments/tier2_base.yaml --dataset reuters90
```

Configs prontas em `experiments/`:

| Arquivo | Descrição |
|---|---|
| `smoke.yaml` | Validação rápida do pipeline (LR, 2 folds) |
| `webkb.yaml` | Exemplo mínimo com todos os modos |
| `tier2_base.yaml` | Produção RoBERTa-base, fatorial 2² |
| `tier2_base_baselines.yaml` | Tier 2 + baseline `b1` (Bengio 2009) |
| `tier1_distil.yaml` | RoBERTa com batch menor (GPUs <16 GB) |
| `tier3_large.yaml` | Datasets grandes |
| `spcl_soft.yaml` / `spcl_loss.yaml` | Variantes SPCL |
| `large_datasets_roberta_base_5cv.yaml` | Reuters90 e similares (5-fold) |

O design experimental completo está em [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Modos de execução (matriz IS × CL)

A flag `--mode` seleciona a combinação de instance selection (IS) e curriculum learning (CL). Use `--model lr` para trocar RoBERTa por Regressão Logística (mais rápido para testes).

Métodos de curriculum (`--curriculum-method`):

| Método | Descrição |
|---|---|
| `biois_discrete` | 3 fases discretas Clean → Diverse → Hard (default) |
| `spcl_soft` | Soft-pacing contínuo sobre sinais BIOIS (entropia/redundância) |
| `spcl_loss` | SPCL canônico (Jiang et al. AAAI 2015): região Ψ derivada do BIOIS + scheme em `{binary, linear, log, mixture}` |

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

BIOIS é executado apenas para gerar os sinais (beta=0, theta=0); curriculum organiza o treino em fases sobre o conjunto completo.

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

### is_continuos_cl — IS + CL contínuo (SPCL soft)

Alias para IS+CL com `--curriculum-method spcl_soft` por default.

```sh
uv run python main.py webkb --data_dir datasets --fold 0 \
    --mode is_continuos_cl --epochs-per-phase 2 --beta 0.3 --theta 0.2
```

### Exemplo: SPCL canônico (com região Ψ derivada do BIOIS)

```sh
uv run python main.py webkb --data_dir datasets --fold 0 \
    --mode is_cl --curriculum-method spcl_loss \
    --curriculum-loss-scheme linear \
    --curriculum-n-steps 6 --epochs-per-phase 2
```

`--curriculum-loss-scheme` aceita `binary | linear | log | mixture`
(Eqs. 4–7 do paper SPCL). Use `--no-curriculum-loss-prior-reliability`
para usar apenas entropia BIOIS no prior `a`.

## Baselines da literatura

Baselines são indexados por `--baseline N` (ou token `bN` em runners multi-fold). Resultados ficam em `b{N}_fold<k>/`.

| Índice | Token | Método | Referência |
|---|---|---|---|
| 1 | `b1` | Confidence-paced CL | Bengio et al., ICML 2009 |

```sh
# Execução individual
uv run python main.py webkb --fold 0 --baseline 1 --epochs-per-phase 2

# Via run.py (tier2_base_baselines.yaml inclui b1)
uv run python run.py experiments/tier2_base_baselines.yaml

# Via run_experiment.py
uv run python run_experiment.py webkb --modes raw is cl is_cl b1 --n-splits 10
```

O baseline `b1` reutiliza `_probaEveryone` do BIOIS (classificador fraco em TF-IDF), com fases cumulativas por confiança no rótulo — sem máscara de ruído nem peso de redundância.

## Experimentos multi-fold

### `run.py` (YAML)

Preferido para experimentos reproduzíveis. Ao final, gera `summary.csv` com média ± IC 95% por modo (macro-F1, efficiency_score, data_efficiency, etc.).

### `run_experiment.py` (CLI)

Alternativa sem YAML; repassa flags extras ao `main.py`:

```sh
uv run python run_experiment.py webkb --n-splits 10 --model roberta \
    --modes raw is cl is_cl b1 \
    --folds 0 1 2 \
    --beta 0.3 --theta 0.2 --epochs-per-phase 2
```

### Docker (batch multi-dataset)

Scripts em `scripts/` para rodar CV completo em container:

```sh
IMAGE=bio-is-curriculum:latest ./scripts/run_docker_full_cv_multi.sh 0 webkb reuters90
```

Variáveis úteis: `MODES`, `CURRICULUM_METHODS`, `SPCL_LOSS_SCHEMES`, `DATASET_SPLITS` (ex.: `reuters90:5`).

## Agregação e comparação de resultados

`summary.py` consolida `summary.csv` de vários experimentos em Excel:

```sh
# Uma métrica, vários experimentos
uv run python summary.py --metric macro_f1 \
    webkb-10cv-20260605-011430-0815f0 \
    webkb-10cv-20260607-191540-731d44

# Comparação multi-dataset a partir de um batch Docker
uv run python summary.py --compare --run-prefix 20260711-022935 \
    --datasets webkb reuters90 \
    --output summary-compare-20260711-022935.xlsx
```

Notebook de análise exploratória: `analysis/analysis.ipynb`.

## Organização do código

```
├── main.py                  # entry point CLI (delega para src/cli.py)
├── run.py                   # entry point YAML (recomendado)
├── run_experiment.py        # runner multi-fold via CLI
├── summary.py               # agregação cross-experimento → Excel
├── download_datasets.py     # download Zenodo
├── experiments/             # configs YAML prontas
├── scripts/                 # runners Docker
├── analysis/                # notebooks de análise
└── src/
    ├── cli.py               # argumentos e orquestração
    ├── curriculum/
    │   ├── core.py          # orquestrador compartilhado
    │   ├── class_balance.py
    │   ├── methods/         # biois_discrete, spcl_soft, spcl_loss
    │   └── models.py
    ├── baselines/           # baselines da literatura (--baseline N / bN)
    ├── iSel/                # instance selection (BIOIS)
    ├── data/                # loader, upsampling de classes raras
    └── results/             # gravação de métricas e timings
```

## Resultados

### Estrutura por execução isolada

Cada run gera `results/<mode>-<timestamp>-<hex6>/` com os artefatos abaixo.

### Estrutura agrupada (multi-fold)

Com `--experiment-id` (usado por `run.py` e `run_experiment.py`):

```
results/<experiment_id>/
    raw_fold0/
    is_fold0/
    cl_fold0/
    is_cl_fold0/
    b1_fold0/
    ...
    summary.csv          # agregado com IC 95%
```

| Arquivo | Conteúdo |
|---|---|
| `config.json` | Todos os hiperparâmetros, dataset, fold e commit git |
| `timings.csv` | `name, seconds` — data_load, preprocess, model_train, total |
| `phase_metrics.csv` | Métricas por fase: F1, accuracy, hard_slice_macro_f1, avg_seq_len, compute_proxy, … |
| `train_history.csv` | `phase, epoch, step, loss, lr` — uma linha por step de treino |
| `predictions_test.csv` | `idx, y_true, y_pred, pred_entropy` — predições finais no teste |
| `instance_selection.json` | Métricas de IS: redução, n_before/after, remoção por classe |
| `summary.csv` | (nível experimento) média ± IC 95% por modo, incluindo `efficiency_score` e `data_efficiency` |

Para comparar modos dentro de um experimento, use `summary.csv`. Para comparar entre experimentos, use `summary.py`.

## Opções principais

```
--mode {raw,is,cl,is_cl,is_continuos_cl}  Modo de execução (default: is_cl)
--baseline N                              Baseline da literatura (sobrescreve --mode)
--curriculum-method {biois_discrete,spcl_soft,spcl_loss}
--model {lr,roberta}                      Modelo (default: roberta)
--hf-model                                Checkpoint HuggingFace (default: roberta-base)
--n-splits                                Folds no split file (default: 10)
--epochs                                  Épocas para treino único / raw / is (default: 6)
--epochs-per-phase                        Épocas por fase do curriculum (default: 1)
--batch-size                              Batch de treino (default: 32)
--eval-batch-size                         Batch de avaliação (default: 64)
--max-length                              Comprimento máximo de tokenização (default: 256)
--lr / --weight-decay / --warmup-ratio    Hiperparâmetros de fine-tune RoBERTa
--class-balanced-loss                     Peso por frequência de classe na CE (default: True)
--beta / --theta                          Taxas de redução do BIOIS (default: 0.3 / 0.2)
--hard-slice-quantile                     Quantil para hard-slice macro-F1 (default: 0.8)
--curriculum-beta                         Peso de redundância na Fase Hard (default: 0.5)
--curriculum-q                            Quantis das fases discretas (default: 0.3 0.6 0.95)
--curriculum-n-steps                      Passos para spcl_soft / spcl_loss (default: 6)
--curriculum-alpha-decay                  Suavidade do soft-pacing (default: 10.0)
--curriculum-soft-*                       Parâmetros do SPCL soft (lambda, saturação, etc.)
--curriculum-loss-scheme                  Scheme do SPCL canônico: binary|linear|log|mixture
--curriculum-lambda-init/step/mult/max    Controle de λ no SPCL canônico
--curriculum-lambda2                      λ₂ do scheme mixture (default: λ_init/2)
--curriculum-loss-prior-reliability       Usa reliability BIOIS no prior a (default: True)
--curriculum-loss-recompute-every         Recomputa losses a cada K steps no SPCL (default: 2)
--experiment-id                           Agrupa runs multi-fold sob results/<id>/
--results-dir                             Diretório base de resultados (default: results/)
```
