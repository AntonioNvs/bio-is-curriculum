# biO-IS-Curriculum

Treinamento curricular guiado por redundância e ruído para classificação de texto com Transformers.

## Instalação

```sh
uv sync
```

### Datasets

Os datasets vêm do [Zenodo](https://zenodo.org/) (mesma suíte do [bio-is](https://github.com/waashk/bio-is) / [atcBench](https://github.com/waashk/atcBench)) e são organizados em `datasets/<nome>/` com o layout:

```
datasets/<nome>/
    texts.txt          # documentos (um por linha)
    score.txt          # rótulos de classe
    splits/            # split_5.pkl, split_10.pkl (partições CV)
    tfidf/             # matrizes TF-IDF em CSR (.gz) por fold
```

A representação TF-IDF usada pelo BIOIS segue o pré-processamento do bio-is: remoção de stopwords (scikit-learn) e retenção apenas de termos que aparecem em pelo menos dois documentos (`min_df=2`).

```sh
uv run python download_datasets.py              # todos
uv run python download_datasets.py webkb reuters90 agnews yelp_2013 medline  # subset
```

| Dataset | Tamanho | Dim. | # Classes | Densidade | Desbalanceamento | CV | Link |
|---------|---------|------|-----------|-----------|------------------|----|------|
| `webkb` | 8,199 | 23,047 | 7 | 209 | Desbalanceado | 10-fold | [Zenodo](https://doi.org/10.5281/zenodo.7555368) |
| `reuters90` | 13,327 | 27,302 | 90 | 171 | Extremamente desbalanceado | 5-fold | [Zenodo](https://doi.org/10.5281/zenodo.7555298) |
| `mpqa` | 10,606 | 2,643 | 2 | 3 | Desbalanceado | 10-fold | [Zenodo](https://doi.org/10.5281/zenodo.7555268) |
| `twitter` | 6,997 | 8,135 | 6 | 28 | Desbalanceado | 10-fold | [Zenodo](https://doi.org/10.5281/zenodo.7554707) |
| `sst1` | 11,855 | 9,015 | 5 | 19 | Balanceado | 10-fold | [Zenodo](https://doi.org/10.5281/zenodo.7555319) |
| `yelp_reviews` | 5,000 | 23,631 | 2 | 132 | Balanceado | 10-fold | [Zenodo](https://doi.org/10.5281/zenodo.7555396) |
| `20ng` | 18,846 | 97,401 | 20 | 96 | Balanceado | 10-fold | [Zenodo](https://doi.org/10.5281/zenodo.7555237) |
| `agnews` | 127,600 | 39,837 | 4 | 37 | Balanceado | 5-fold | [Zenodo](https://doi.org/10.5281/zenodo.7555424) |
| `yelp_2013` | 335,018 | 62,964 | 6 | 152 | Desbalanceado | 5-fold | [Zenodo](https://doi.org/10.5281/zenodo.7555898) |
| `medline` | 860,424 | 125,981 | 7 | 77 | Desbalanceado | 5-fold | [Zenodo](https://doi.org/10.5281/zenodo.7555820) |

**Experimental use:** `webkb` and `reuters90` are for fast ablations. Multi-dataset batches use [`experiments/campaigns/full_cv_multi.yaml`](experiments/campaigns/full_cv_multi.yaml). Large-scale efficiency claims use `agnews`, `yelp_2013`, and `medline` via [`experiments/campaigns/large_datasets_5cv.yaml`](experiments/campaigns/large_datasets_5cv.yaml).

## Quick start (recommended)

YAML-first experiments via `bio-experiment` (Docker + GPU 7 configured in campaign YAML):

```sh
# Smoke test (single fold)
uv run bio-experiment experiments/campaigns/smoke_docker.yaml --folds 0

# Curriculum signal ablations (4 datasets × 4 methods)
uv run bio-experiment experiments/campaigns/curriculum_ablations_multi.yaml --folds 0

# Full multi-dataset CV matrix
uv run bio-experiment experiments/campaigns/full_cv_multi.yaml
```

Legacy shims still work: `uv run python run.py experiments/smoke.yaml` → same as `bio-experiment`.

Campaign configs live in [`experiments/campaigns/`](experiments/campaigns/). Single-dataset YAMLs remain in [`experiments/`](experiments/).

The experiment design doc is [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

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
| 1 | `b1` | Margin-paced CL | Bengio et al., ICML 2009 |

```sh
# Execução individual
uv run python main.py webkb --fold 0 --baseline 1 --epochs-per-phase 2

# Via run.py (tier2_base_baselines.yaml inclui b1)
uv run python run.py experiments/tier2_base_baselines.yaml

# Via run_experiment.py
uv run python run_experiment.py webkb --modes raw is cl is_cl b1 --n-splits 10
```

O baseline `b1` usa margem multiclasse OOF de LR em TF-IDF (`signals/oracle_margin.py`) — currículo em 2 fases (easy → target), sem BIOIS, máscara de ruído ou peso de redundância.

## Multi-fold experiments

`bio-experiment` runs all modes × folds, aggregates per experiment folder, and writes a manifest JSON.

```sh
uv run bio-experiment experiments/webkb.yaml --folds 0 1 2
```

Each job produces `results/<experiment_id>/summary.csv` (mean ± 95% CI per mode).

### Manifest and Excel export

After a campaign completes:

```sh
uv run python summary.py results/experiments/curriculum_ablations_multi_<timestamp>/
```

This writes `.xlsx` and `.csv` summaries next to the manifest. Configure sheet layout in the YAML:

```yaml
campaign:
  name: my_experiment
  summary:
    layout: compare_by_dataset
    metrics: [macro_f1, hard_slice_macro_f1, train_time_s, total_time]
```

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for details.

## Result aggregation (legacy)

For runs without a manifest, deprecated folder discovery still works:

```sh
uv run python summary.py --compare --run-prefix 20260711-022935 --datasets webkb reuters90
```

Notebook de análise exploratória: `analysis/analysis.ipynb`.

## Code layout

```
├── main.py                  # shim → bio-run
├── run.py                   # shim → bio-experiment
├── run_experiment.py        # shim → bio-experiment
├── summary.py               # manifest → Excel/CSV export
├── download_datasets.py     # Zenodo download
├── experiments/             # YAML configs
│   └── campaigns/           # multi-dataset campaign YAMLs
├── scripts/                 # utility scripts (e.g. export_imbalance_comparison.py)
├── analysis/                # analysis notebooks
└── src/bio_is_curriculum/
    ├── cli/                 # bio-run, bio-experiment, bio-summary
    ├── config/              # schema, campaign expansion, defaults
    ├── curriculum/          # curriculum methods and orchestrator
    ├── selection/           # BIOIS instance selection
    ├── data/                # dataset loader
    └── results/             # metrics, aggregator, manifest, summary export
```

## Resultados

### Estrutura por execução isolada

Cada run gera `results/<mode>-<timestamp>-<hex6>/` com os artefatos abaixo.

### Grouped structure (multi-fold)

With `--experiment-id` (used by `bio-experiment`):

```
results/<experiment_id>/
    raw_fold0/
    is_fold0/
    ...
    summary.csv

results/experiments/
    <event>_<timestamp>/
        manifest.json
        summary.xlsx
        summary.csv
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

Compare modes within one experiment via `summary.csv`. Compare across experiments via the manifest + `summary.py`.

## Opções principais

```
--mode {raw,is,cl,is_cl,is_continuos_cl}  Modo de execução (default: is_cl)
--baseline N                              Baseline da literatura (sobrescreve --mode)
--curriculum-method {biois_discrete,spcl_soft,spcl_loss}
--model {lr,modernbert}                   Modelo (default: modernbert)
--hf-model                                Checkpoint HuggingFace (default: answerdotai/ModernBERT-base)
--train-fraction                          Fração do train split (default: 1.0)
--n-splits                                Folds no split file (default: 10)
--epochs                                  Épocas para treino único / raw / is (default: 6)
--epochs-per-phase                        Épocas por fase do curriculum (default: 1)
--batch-size                              Batch de treino (default: 32)
--eval-batch-size                         Batch de avaliação (default: 64)
--max-length                              Comprimento máximo de tokenização (default: 256)
--lr / --weight-decay / --warmup-ratio    Hiperparâmetros de fine-tune ModernBERT
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
