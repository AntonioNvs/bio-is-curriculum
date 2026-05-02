# biO-IS-Curriculum

Treinamento curricular guiado por redundância e ruído para classificação de texto com Transformers.

## Instalação

```sh
uv sync
```

## 4 modos de execução (matriz IS × CL)

A flag `--mode` seleciona uma das 4 combinações de teste. Use `--model lr` para trocar RoBERTa por Regressão Logística (mais rápido para testes rápidos).

### baseline — sem IS, sem CL

Fine-tuning padrão no conjunto de treino completo.

```sh
uv run python main.py webkb --data_dir datasets --fold 0 \
    --mode baseline --epochs 6
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
--mode {baseline,is,cl,is_cl}   Modo de execução (default: is_cl)
--model {lr,roberta}             Modelo (default: roberta)
--hf-model                       Checkpoint HuggingFace (default: roberta-base)
--epochs                         Épocas para treino único / baseline / is (default: 6)
--epochs-per-phase               Épocas por fase do curriculum (default: 2)
--batch-size                     Batch de treino (default: 16)
--max-length                     Comprimento máximo de tokenização (default: 256)
--beta / --theta                 Taxas de redução do BIOIS (default: 0.3 / 0.2)
--curriculum-beta                Peso de redundância na Fase Hard: w=1-beta*r (default: 0.5)
--results-dir                    Diretório base de resultados (default: results/)
```
