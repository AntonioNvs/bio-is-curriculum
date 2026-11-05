# Runbook — experimentos para a nova máquina

**Documento operacional.** Consolida o que deve ser executado, com o `.yaml` de cada bloco, e um plano de fatiamento que minimiza o wall-clock.

Deriva de `paper/report_consolidado.md` §11. Custos são os **declarados pelas próprias campanhas** (ver §7, caveat de calibração).

---

## 1. Pré-requisitos de código

Nada roda de forma confiável antes destes patches. P1–P3 são correções; P4 é a única implementação nova obrigatória.

> **P1–P3 aplicados** (2026-11-05). Falta só `docker build`. Ver §1.1 para o efeito colateral que precisa ser conhecido antes de rodar `bio-equivalence`.

| ID | Patch | Arquivo | O que fazer |
|---|---|---|---|
| **P1** | `compute_proxy` | `src/bio_is_curriculum/models/modernbert.py:536-540` | ✅ `return float(self._token_count_total)`. Hoje é `global_step × avg_seq_len`, que omite `batch_size` e infla o ganho em bs=32 |
| **P2** | `_matches_method` | `src/bio_is_curriculum/results/discovery.py:102-115` | ✅ Compara também `hf_model` e `batch_size` contra `REQUIRED_CONFIG`, para que célula de config mista nunca seja resolvida em silêncio |
| **P3** | `FACTORIAL_DATASETS` | `src/bio_is_curriculum/results/paper_tables.py:20-29` | ✅ Removido `"webkb"` das duas listas — o `.tex` usa 6 datasets no fatorial e 5 nas ablações |
| **P4** | Método `random_discrete` | novo `src/bio_is_curriculum/curriculum/methods/random_discrete.py` + `registry.py` | Atribuição **estratificada aleatória** às fases. Implementação mínima: subclasse de `BIOISDiscreteCurriculum` que substitui o escore de dificuldade por `np.random.RandomState(seed).rand(n)`. Como o código já faz `rank_{y_i}(·)` por classe, isso produz exatamente fases aleatórias com cotas por classe preservadas |
| **P5** | Flags de ablação (opcional) | `curriculum/methods/biois_discrete.py` | `risk_floor: bool` (liga/desliga `max(d, γ)`) e `noise_weight: float` (hoje `λn = 0.5` está hardcoded). Necessários só para a Frente D.2 |
| **P6** | Avaliação por época (opcional) | `models/modernbert.py`, bloco `epoch_end` (~`:440`) | Avaliar no **teste** a cada `epoch_end`, além da validação que já roda. Um único run Raw de 6 épocas passa a render a curva inteira, em vez de treinar 2 e 4 épocas do zero |

**Ordem:** P1–P3 antes de qualquer coisa. P4 antes do Y3. P5/P6 são opcionais e não bloqueiam.

### 1.1 Efeito colateral do P2 (importante)

O P2 é **ligado por padrão**. Enquanto o Y1 não for re-rodado, `bio-equivalence` e `bio-significance` vão reportar como **`unresolved`** toda célula cujo run mais recente tenha `roberta-base` ou `batch_size` 16 — o que hoje inclui boa parte das células de `reuters90`, `medline` e `dblp`.

Isso é o comportamento pretendido (essas células *são* inválidas), mas é uma mudança visível. Para inspecionar os resultados legados sem o filtro:

```sh
uv run python -m bio_is_curriculum.results.discovery --results-dir results --allow-config-mismatch
```

Contagem medida em `results/` (2026-11-05): **104 células resolvidas sem filtro → 83 com filtro**; 25 avisos nomeiam os run dirs excluídos e o campo divergente.


### Imagem Docker

```sh
docker build -t bio-is-curriculum:latest .
```

Reconstruir **após** qualquer mudança de scheduler ou de método de currículo.

### Datasets

```sh
uv run python download_datasets.py
```

Necessários: `reuters90`, `books`, `dblp`, `agnews`, `yelp_2013`. `medline` fica para a fase deferida (§6).

---

## 2. Calibração obrigatória antes do plano

Os tempos do artigo foram medidos em **A100 80GB**. Se a nova máquina for diferente, todo o orçamento abaixo muda. Rodar um smoke e calibrar:

```sh
uv run bio-experiment experiments/campaigns/factorial_matched_2x2.yaml \
  --dataset reuters90 --folds 0
```

Isso executa 4 modos × 1 fold. Multiplicar o tempo observado por 5 (folds) e comparar com o orçamento de **2,7 GPU-h** para reuters90. Se der mais de ~2× de diferença, escalar toda a tabela do §5 pelo fator medido antes de decidir o fatiamento.

**Caveat registrado:** o orçamento declarado em `factorial_matched_2x2.yaml` (yelp_2013 = 65,1 GPU-h) é ~4× maior que o implícito nos tempos medidos do artigo (yelp_2013 ≈ 13,8 h somando os 4 modos). A discrepância não foi resolvida — por isso a calibração acima é obrigatória, e os números do §5 devem ser tratados como **conservadores**.

---

## 3. Bloco 1 — Fatorial 2² pareado (bloqueante)

**Responde:** a tabela principal é válida? Gera o substrato para o bootstrap pareado, o corte por classe e a curva Raw.

**Por que é bloqueante:** as quatro células do fatorial atual não foram rodadas sob config pareada (`_matches_method` não filtrava encoder nem batch size). Medline compara Raw `roberta-base` contra IS/CL/IS+CL `ModernBERT-base`; yelp/agnews/reuters90 misturam bs=16 e bs=32. Além disso, **0 de 259 run dirs têm `predictions_test.csv`** — sem re-run não existe bootstrap por instância.

### Y1 — `experiments/campaigns/factorial_matched_2x2.yaml`

**Já existe.** Duas alterações:

1. **Comentar `medline`** no bloco `datasets:`. Corta 151,1 dos 245,8 GPU-h (61%) e derruba o caminho crítico de ~9-10 h para ~1,1 h (o run mais longo passa a ser um fold de yelp Raw).
2. **Remover o job `raw_ep{epochs}`** deste arquivo (o `matrix: training.epochs: [2, 4]`). Ele treina do zero, não reaproveita trajetória, e custa +1× Raw em todos os datasets. Vai para o Y2, restrito aos datasets diagnósticos.

Conteúdo efetivo após a edição:

```yaml
docker:
  image: bio-is-curriculum:latest
  gpu_id: 0            # trocar por shard
  cpus: 16
  memory: 32g

campaign:
  name: factorial_matched_2x2
  timestamp: auto
  summary:
    layout: compare_by_dataset
    metrics:
      - macro_f1
      - micro_f1
      - f1_weighted
      - accuracy
      - hard_slice_macro_f1
      - n_train_instances
      - n_removed
      - is_reduction
      - train_time_s
      - total_time

  datasets:
    reuters90: { n_splits: 5 }
    books:     { n_splits: 5 }
    dblp:      { n_splits: 5 }
    agnews:    { n_splits: 5 }
    yelp_2013: { n_splits: 5 }
    # medline: { n_splits: 5 }   # DEFERIDO — ver §6

  defaults:
    model: modernbert
    hf_model: answerdotai/ModernBERT-base
    folds: [0, 1, 2, 3, 4]
    random_state: 42
    instance_selection:
      beta: 0.3
      theta: 0.2
    curriculum:
      method: biois_discrete
      q_low: 0.3
      q_mid: 0.6
      q_high: 0.95
      beta: 0.5
      margin_weight: 0.6
      entropy_weight: 0.4
      length_weight: 0.25
      noise_weight_phases: [hard]
      phase_max_lengths: [96, 160, 256]
    training:
      batch_size: 32          # UM valor para as 4 células
      eval_batch_size: 64
      max_length: 256
      epochs: 6
      epochs_per_phase: 2
      lr: 2.0e-5
      weight_decay: 0.001
      warmup_ratio: 0.06
      imbalance_method: inverse_freq_ce

  jobs:
    - modes: [raw, is, cl, is_cl]
      experiment_id: "{dataset}-{n_splits}cv-{timestamp}_factorial_matched"
```

**Invariante:** `batch_size: 32` e `hf_model: answerdotai/ModernBERT-base` valem para as quatro células. É isso que torna o bloco comparável — e é o que P2 passa a garantir.

### Y2 — `experiments/campaigns/raw_curve_diagnostic.yaml` **(novo)**

**Responde:** "parar o treino completo mais cedo já atinge a mesma qualidade?"

Restrito aos três datasets diagnósticos, para não pagar +1× Raw em todos. Custo ≈ **7,6 GPU-h**.

```yaml
# Curva Raw@2ep / Raw@4ep / Raw@6ep (o 6ep vem do Y1).
# Só nos datasets onde a perda existe (books, reuters90) e no contraponto (agnews).
# Treina do zero em cada orçamento — ver P6 para a versão que reaproveita trajetória.
docker:
  image: bio-is-curriculum:latest
  gpu_id: 0
  cpus: 16
  memory: 32g

campaign:
  name: raw_curve_diagnostic
  timestamp: auto
  summary:
    layout: compare_by_dataset
    metrics:
      - macro_f1
      - hard_slice_macro_f1
      - train_time_s
      - total_time

  datasets:
    books:     { n_splits: 5 }
    reuters90: { n_splits: 5 }
    agnews:    { n_splits: 5 }

  defaults:
    model: modernbert
    hf_model: answerdotai/ModernBERT-base
    folds: [0, 1, 2, 3, 4]
    random_state: 42
    training:
      batch_size: 32
      eval_batch_size: 64
      max_length: 256
      epochs_per_phase: 2
      lr: 2.0e-5
      weight_decay: 0.001
      warmup_ratio: 0.06
      imbalance_method: inverse_freq_ce

  jobs:
    - modes: [raw]
      matrix:
        training.epochs: [2, 4]
      experiment_id: "{dataset}-{n_splits}cv-{timestamp}_raw_ep{epochs}"
```

**Se P6 for aplicado**, este arquivo deixa de ser necessário: um Raw de 6 épocas do Y1 rende a curva inteira, e o Y2 vira só um job de verificação em `reuters90`. Economia de ~7,6 GPU-h.

---

## 4. Bloco 2 — Controles da pergunta decisiva

**Responde:** com o mesmo orçamento, BIOIS-CL bate alternativas simples? Roda em **Books, Reuters-90 e AG News** — os dois casos de perda e o contraponto com perda pequena. Repetir nos demais só os contrastes que sustentarem a conclusão.

### Y3 — `experiments/campaigns/controls_signal_same_selection.yaml` **(novo)**

**Isola o escore de dificuldade mantendo a seleção congelada.** Todos os jobs são `is_cl` com o mesmo β, θ e `random_state` → **os mesmos sobreviventes BIOIS**. Só o escore muda. Custo ≈ **11,6 GPU-h**.

```yaml
# Mesmos sobreviventes BIOIS, mesmo orçamento, só o escore de dificuldade muda.
# margem pura (1.0/0.0) e entropia pura (0.0/1.0) não exigem código novo:
# margin_weight e entropy_weight já são chaves de config.
# random_discrete exige o patch P4.
docker:
  image: bio-is-curriculum:latest
  gpu_id: 0
  cpus: 16
  memory: 32g

campaign:
  name: controls_signal_same_selection
  timestamp: auto
  summary:
    layout: compare_by_dataset
    metrics:
      - macro_f1
      - hard_slice_macro_f1
      - n_train_instances
      - n_removed
      - is_reduction
      - train_time_s
      - total_time

  datasets:
    books:     { n_splits: 5 }
    reuters90: { n_splits: 5 }
    agnews:    { n_splits: 5 }

  defaults:
    model: modernbert
    hf_model: answerdotai/ModernBERT-base
    folds: [0, 1, 2, 3, 4]
    random_state: 42
    instance_selection:
      beta: 0.3
      theta: 0.2
    curriculum:
      method: biois_discrete
      q_low: 0.3
      q_mid: 0.6
      q_high: 0.95
      beta: 0.5
      margin_weight: 0.6
      entropy_weight: 0.4
      length_weight: 0.25
      noise_weight_phases: [hard]
      phase_max_lengths: [96, 160, 256]
    training:
      batch_size: 32
      eval_batch_size: 64
      max_length: 256
      epochs: 6
      epochs_per_phase: 2
      lr: 2.0e-5
      weight_decay: 0.001
      warmup_ratio: 0.06
      imbalance_method: inverse_freq_ce

  jobs:
    # Referência: escore composto. Repetir aqui para o mesmo timestamp/shard.
    - modes: [is_cl]
      experiment_id: "{dataset}-{n_splits}cv-{timestamp}_sel_composite"

    # Margem pura — matriz de 1 elemento = override fixo.
    - modes: [is_cl]
      matrix:
        curriculum.margin_weight: [1.0]
        curriculum.entropy_weight: [0.0]
      experiment_id: "{dataset}-{n_splits}cv-{timestamp}_sel_margin_only"

    # Entropia pura.
    - modes: [is_cl]
      matrix:
        curriculum.margin_weight: [0.0]
        curriculum.entropy_weight: [1.0]
      experiment_id: "{dataset}-{n_splits}cv-{timestamp}_sel_entropy_only"

    # Fases aleatórias estratificadas — exige P4.
    - modes: [is_cl]
      matrix:
        curriculum.method: [random_discrete]
      experiment_id: "{dataset}-{n_splits}cv-{timestamp}_sel_random_phases"
```

**Nota sobre `matrix` com um elemento:** o schema só permite override por job através de `matrix` (chave pontuada). Uma matriz de um único valor por chave produz exatamente um combo — funciona como override fixo. Confirmado em `config/campaign.py::_job_yaml_for_combo`.

**Nota sobre elegibilidade final.** Com `q_high: 0.95`, o cronograma natural muda também **quais IDs nunca são treinados**. Para separar isso do efeito de escore, rodar uma sensibilidade com `curriculum.q_high: [1.0]` no job de referência.

### Y4 — `experiments/campaigns/controls_random_stratified.yaml` **(novo)**

**Responde:** "selecionar aleatoriamente, com a mesma retenção, já basta?"

Depende do Y1: o `train_fraction` deve igualar a retenção medida, lida de `is_reduction` no summary do Y1. Preencher `{frac}` por dataset **depois** que o Y1 terminar. Custo ≈ **5,1 GPU-h**.

```yaml
# Controle: treino completo com subsample ESTRATIFICADO da mesma retenção que o IS.
# train_fraction usa StratifiedShuffleSplit sobre o split de treino (data/preprocessing.py:105).
# O valor deve ser lido de is_reduction no summary do Y1 — não chutar 0.66.
docker:
  image: bio-is-curriculum:latest
  gpu_id: 0
  cpus: 16
  memory: 32g

campaign:
  name: controls_random_stratified
  timestamp: auto
  summary:
    layout: compare_by_dataset
    metrics:
      - macro_f1
      - hard_slice_macro_f1
      - n_train_instances
      - train_time_s
      - total_time

  datasets:
    books:     { n_splits: 5 }
    reuters90: { n_splits: 5 }
    agnews:    { n_splits: 5 }

  defaults:
    model: modernbert
    hf_model: answerdotai/ModernBERT-base
    folds: [0, 1, 2, 3, 4]
    random_state: 42
    training:
      batch_size: 32
      eval_batch_size: 64
      max_length: 256
      epochs: 6
      epochs_per_phase: 2
      lr: 2.0e-5
      weight_decay: 0.001
      warmup_ratio: 0.06
      imbalance_method: inverse_freq_ce

  jobs:
    - modes: [raw]
      matrix:
        training.train_fraction: [0.66]   # SUBSTITUIR por 1 - is_reduction medido
      experiment_id: "{dataset}-{n_splits}cv-{timestamp}_raw_rand{frac}"
```

### Y5 — `experiments/campaigns/beta_theta_sweep.yaml`

**Já existe.** Sensibilidade a β/θ — a questão aberta mais barata do artigo. Duas reduções para tempo:

- Remover `dblp` de `datasets:` (o caso "plano"; o ponto qualitativo está em books). Fica `reuters90` + `books`.
- Manter `folds: [0, 1, 2]`.

Custo: ~**16 GPU-h** (contra 24 no escopo completo).

---

## 5. Orçamento e fatiamento

### Total

| Bloco | Arquivo | GPU-h |
|---|---|---|
| Y1 — fatorial pareado (5 datasets, sem medline) | `factorial_matched_2x2.yaml` | **94,7** |
| Y2 — curva Raw | `raw_curve_diagnostic.yaml` | 7,6 |
| Y3 — controles de escore | `controls_signal_same_selection.yaml` | 11,6 |
| Y4 — aleatório estratificado | `controls_random_stratified.yaml` | 5,1 |
| Y5 — sweep β/θ (2 datasets) | `beta_theta_sweep.yaml` | 16,0 |
| | **Total** | **≈ 135** |

Sem medline. Com medline: **+151 GPU-h** — motivo do deferimento.

### Custo por dataset no Y1

| Dataset | GPU-h | Shard por fold |
|---|---|---|
| yelp_2013 | 65,1 | **13,0** |
| agnews | 12,3 | 2,5 |
| dblp | 7,7 | 1,5 |
| books | 6,9 | 1,4 |
| reuters90 | 2,7 | 0,5 |

**yelp_2013 é 69% do bloco 1.** O fatiamento começa por ele, sempre.

### Wall-clock por número de GPUs

Regra: fatiar `yelp_2013` por fold primeiro (1 shard = `--dataset yelp_2013 --folds N`), depois distribuir os datasets pequenos como preenchimento.

| GPUs | Wall-clock | Como |
|---|---|---|
| 1 | ~95 h | sequencial, barato-primeiro |
| 2 | ~48 h | yelp f0-f4 dividido em 2 |
| 4 | ~24 h | yelp f0-f4 em 3 GPUs + 1 GPU para o resto |
| 6 | ~17 h | yelp 1 fold/GPU (5 GPUs) + 1 GPU para o resto |
| 8 | ~13 h | **limitado pelo shard de yelp** (13,0 h) |
| 12 | ~9 h | exige fatiar yelp também **por modo** (20 shards de ~3,3 h) |

Acima de 8 GPUs o shard de yelp vira gargalo. Para descer, gerar jobs separados com `modes: [raw]`, `[is]`, `[cl]`, `[is_cl]` em vez de um job com os quatro.

### Gerando shards

`gpu_id` está no YAML e não há flag de CLI para sobrescrever. Gerar uma cópia por shard:

```sh
mkdir -p experiments/campaigns/_shards
for i in 0 1 2 3 4 5; do
  sed "s/gpu_id: 0/gpu_id: $i/" experiments/campaigns/factorial_matched_2x2.yaml \
    > experiments/campaigns/_shards/factorial_gpu$i.yaml
done
```

Lançar (um por GPU, ajustando o recorte de dataset/folds):

```sh
mkdir -p logs
nohup uv run bio-experiment experiments/campaigns/_shards/factorial_gpu0.yaml \
  --dataset yelp_2013 --folds 0 > logs/y1_yelp_f0.log 2>&1 &

nohup uv run bio-experiment experiments/campaigns/_shards/factorial_gpu5.yaml \
  --dataset agnews --folds 0 1 2 3 4 > logs/y1_agnews.log 2>&1 &
```

`--folds` aceita lista (`nargs="*"`); `--dataset` aceita um único dataset por invocação.

---

## 6. Fase deferida

Não submeter junto. Cada item só entra depois que os controles centrais estiverem lidos.

| Item | Arquivo | GPU-h | Gatilho |
|---|---|---|---|
| **medline** no fatorial | `factorial_matched_2x2.yaml` (descomentar) | 151 | Só se a máquina tiver GPUs ociosas após o bloco 1. Alternativa: rodar com `n_splits: 3` → ~91 GPU-h, e declarar no texto |
| **Ruído injetado** | `label_noise_injection.yaml` | 66 | **Bloqueado.** Exige patch de 7 arquivos + go/no-go de ~2 GPU-h em reuters90 ρ=0,4 fold 0. Se o recall do θ ficar perto do acaso, **descartar a campanha** |
| **EL2N dinâmico** (concorrente externo) | novo `competitor_el2n.yaml` | a medir | Exige implementação nova. Só depois dos controles |
| **Confiança simples (E2SC-like)** | novo | a medir | Exige método novo. Nomear como ablação **inspirada** em E2SC |
| **Seeds pareadas** | matriz `random_state` no Y1 | 2× Y1 | Só para as comparações que sobreviverem aos controles |
| **Ablações de componente** (piso de risco, pesos) | exige P5 | a medir | Só onde os logs apontarem hipótese |

---

## 7. Ordem de submissão

1. **P1–P3** e `docker build`.
2. **Smoke de calibração** (§2) — obrigatório, define a escala de tudo.
3. **Y1** — barato-primeiro: reuters90 → books → dblp → agnews → yelp_2013. Fatiar yelp desde o início.
4. **Y2** e **Y5** em paralelo com o Y1 (não dependem dele).
5. **Y3** em paralelo (não depende do Y1; exige P4).
6. **Y4** — **só depois** do Y1, para ler `is_reduction`.
7. Ler `predictions_test.csv` → `bio-equivalence` → `bio-significance` → `bio-summary`.

---

## 8. Checklist de sanidade

- [ ] `--dry-run` de cada campanha **antes** de lançar
- [ ] As linhas de comando emitidas carregam as flags esperadas, e o **valor muda** entre células da matriz. Um dry-run que sai 0 **não** prova que a chave está ligada — o loader não valida chaves desconhecidas
- [ ] `config.json` de cada célula do Y1 tem o **mesmo** `hf_model` e `batch_size`. É a verificação que o P2 passa a automatizar
- [ ] Existe `predictions_test.csv` em cada run dir do Y1
- [ ] `total_run_time_s` presente em todas as execuções
- [ ] `git_commit` nos `config.json` — se continuar `None`, a imagem está desatualizada
