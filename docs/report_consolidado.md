# BIOIS-CL — Plano consolidado para submissão ACL

**Documento de trabalho para o autor e o orientador**
Consolida o parecer do orientador, a revisão crítica interna e a auditoria do repositório.

Fontes: `report.md` (parecer), `review-acl.md` (revisão crítica), `acl_latex.tex` (manuscrito, 684 linhas), `references.bib` (28 entradas), `experiments/campaigns/*.yaml`, `src/`.

---

## 0. Veredito

**Não submeter a versão atual.** O artigo mostra reduções de tempo relevantes, mas ainda não demonstra que as regras propostas oferecem vantagem sobre alternativas simples com custo comparável. Três problemas são estruturais:

1. **Novidade não sustentada.** Reusar o mesmo vetor de escores para selecionar e ordenar dados já tem antecedente direto (Dai et al. 2026). O argumento "não é um pipeline" está no manuscrito (`acl_latex.tex:47`, `:304`) e é falso quando o baseline conserva os posteriors.
2. **Comparação inválida na tabela principal.** As quatro células do fatorial 2² não foram rodadas sob configuração pareada — misturam encoder e batch size entre células do mesmo dataset. Isso contamina a manchete de eficiência e a claim de não inferioridade em Medline.
3. **Claim de mecanismo não falsificável.** Nada no artigo testa que a regra θ remove ruído. Em dados limpos, ela é indistinguível de "remover exemplos difíceis".

A pergunta decisiva: **com o mesmo orçamento, BIOIS-CL produz resultados melhores do que parar o treino completo mais cedo, selecionar aleatoriamente, ou usar um currículo simples com os mesmos posteriors?**

---

## 1. Estado atual — o que já está feito

`review-acl.md` está desatualizado em relação ao `.tex`. Não refazer:

| Item apontado em `review-acl.md` | Estado real |
|---|---|
| A1 — Abstract ausente | **Feito** (`acl_latex.tex:41`) |
| A2 — Falta resumo na introdução | **Feito** (`:53`) |
| A3 — Remover WebKB | **Feito** — o `.tex` usa 6 datasets. *Mas `paper_tables.py:20` ainda lista `webkb`* |
| A4 — Inverter ordem das tabelas | **Feito** (`:348` documenta a ordenação por margem) |
| A5 — Typos | **Feito** — não há `pottentially` nem `statitics` |
| E1 — Seção IS Parameters vazia | **Feito** (`:500` preenchida) |
| PF5 — Related work insuficiente | **Parcialmente feito** — `Sorscher2022`, `Marion2023`, `Swayamdipta2020`, `EinDor2020`, `Hu2022`, `Cui2019` já estão no `.bib` e citados |
| PF8 — Sem seção ética | **Pendente** |
| E3 — Page limit | **Pendente** — o `.tex` nunca foi compilado para conferência |

O que o `.tex` **não** tem: nenhuma citação a Dai et al. 2026, Sun et al. 2023, Attendu & Corbeil 2023, Bengio & Grandvalet 2004 ou Dror et al. 2018.

---

## 2. Bloqueadores técnicos

Executar **antes de qualquer re-run**. Sem isso, os experimentos reproduzem os erros que o parecer aponta.

| ID | Ação | Onde | Como |
|---|---|---|---|
| **B1** | Corrigir `compute_proxy` | `src/bio_is_curriculum/models/modernbert.py:536-540` | Hoje `global_step × avg_seq_len`, que omite `batch_size` → exatamente 2× menor em bs=32 do que em bs=16. Trocar por `return float(self._token_count_total)`, já rastreado. O speedup honesto em Reuters-90 é ~2,9×, não 5,8× |
| **B2** | `_matches_method` filtrar `hf_model` e `batch_size` | `src/bio_is_curriculum/results/discovery.py:102-115` | Hoje filtra só `mode`, `curriculum_method`, `n_splits`, `train_fraction` e `method.match`. Adicionar comparação em `cfg` para que uma célula mista nunca seja resolvida em silêncio |
| **B3** | Remover `webkb` de `FACTORIAL_DATASETS` | `src/bio_is_curriculum/results/paper_tables.py:20-29` | O `.tex` usa 6 datasets; o código ainda lista 7 e reintroduziria WebKB nas tabelas regeneradas |
| **B4** | Validar chaves de config desconhecidas no loader | `src/bio_is_curriculum/config/loader.py` | O caminho YAML→config tem **dois saltos** (`loader.py` e `cli/experiment.py:~120`). Uma chave não implementada some sem aviso. Ver Frente E |

### Evidência do problema B2

Lido dos `config.json` das células resolvidas:

- **medline**: Raw = `roberta-base` (125M); IS, CL, IS+CL = `answerdotai/ModernBERT-base` (149M). A claim de não inferioridade em Medline compara o método contra um encoder **mais fraco**.
- **yelp_2013, agnews, reuters90**: Raw/IS em `batch_size=16`, CL/IS+CL em `batch_size=32`, de lotes de execução diferentes (2026-09-04 vs. 2026-09-14/15).
- **dblp, books**: os únicos blocos pareados de verdade.

O texto afirma "batch size 32" e "ModernBERT-base" como fixos (`:316`), o que contradiz as execuções de Raw/IS.

---

## 3. Frente A — Texto e posicionamento

| ID | Ação | Onde | Como |
|---|---|---|---|
| A1 | Reescrever a contribuição #1 | `acl_latex.tex:58` | Hoje: "reused as schedule controls rather than discarded once pruning ends". É exatamente o que Dai et al. desmonta. Ancorar no que a formulação específica acrescenta: posteriors OOF de um proxy leve controlando seleção, exposição e ponderação |
| A2 | Remover "não é um pipeline" | `:47` | Descrever as operações e o estado compartilhado, não nomear a arquitetura |
| A3 | Remover "prune-then-schedule exige outro probe" | `:304` | Falso quando o baseline conserva os posteriors. A vantagem de custo vale frente aos estimadores **efetivamente** usados, não frente a toda implementação sequencial |
| A4 | Corrigir "entropy is our signal" | `:41`, `:497`, `:668` | O método é `d_i = (1−λℓ)(λm·m_i + λe·e_i) + λℓ·ℓ_i`, depois `d̃_i = max(d_i, γ_i)` e `w_i = clip[(1−λr·min(r_i,r_max)·I[i∈H])(1−λn·γ_i), ε, 1]`. Descrição e ablação devem corresponder à fórmula composta |
| A5 | Trocar linguagem de detecção | `:66`, `:134-149` | "risco potencial de ruído" e "sinal relacionado à redundância" enquanto não houver validação independente. Erro confiante do proxy não prova rótulo errado; confiança alta em acerto não prova redundância |
| A6 | Explicar as perdas no corpo | `:433` | Hoje só "localized to small imbalanced datasets", sem evidência. Ligar ao diagnóstico da Frente E; não afirmar causa (classes minoritárias) sem teste |
| A7 | Limitar a conclusão | `:668` | "best overall accuracy–efficiency profile" precisa sobreviver aos controles |
| A8 | Uniformizar terminologia | global | `não inferioridade` ≠ `equivalência`; `épocas nominais` ≠ visitas por exemplo; `erros confiantes` vs. `baixa entropia` |

**Nota sobre A4.** Os pesos `λr = λn = 0.5` e `r_max = 0.5` estão fixos no texto (`:288`) e no código, sem flag de CLI. As ablações de ponderação exigem expô-los.

---

## 4. Frente B — Related work

Entradas ausentes do `references.bib` (28 chaves hoje, nenhuma destas):

| Chave | Papel | Onde citar |
|---|---|---|
| **Dai et al. 2026** | **Ameaça de novidade #1.** Uso explícito do mesmo vetor de escores nas duas etapas, em pré-treinamento e SFT de LLMs | Novo parágrafo em Related Work. Diferenciar tarefa (classificação), proxy (TF-IDF LR) e regras específicas |
| **Sun et al. 2023** | Seleção + currículo já articulados para sumarização | Mesmo parágrafo; delimitar a diferença para classificação |
| **Attendu & Corbeil 2023** (NLU on Data Diets) | Poda dinâmica curricular para classificação NLP — **também é baseline**, não só citação | Related Work + Frente C |
| **Bengio & Grandvalet 2004** | Estimador de variância de K-fold | Seção de não inferioridade |
| **Dror et al. 2018** | Teste de significância estatística em NLP | Seção de estatística |

**Corrigir:** `Feng2025` já está no `.bib` — conferir que o venue no texto diz **ACL Student Research Workshop**.

**Remover como argumento:** a frase em `:66` que apresenta BIOIS vs. E2SC apenas como posicionamento. O parecer pede que E2SC vire **ablação rodada** (Frente C).

---

## 5. Frente C — Baselines e orçamento

Cinco famílias de comparação. Nenhuma existe hoje no código, exceto a primeira.

| ID | Comparação | Estado | Como implementar |
|---|---|---|---|
| C1 | **Raw com checkpoints** | ✅ **Resolvido** | `experiments/campaigns/factorial_matched_2x2.yaml` já tem o job `modes: [raw]` com `matrix: training.epochs: [2, 4]`. Dá a curva Raw@2ep / 4ep / 6ep reaproveitando trajetórias. **Falta só extrair e plotar** |
| C2 | **Aleatório estratificado** | ❌ | `train_fraction` já faz subsample estratificado (`data/preprocessing.py:105-140`). Adicionar job `modes: [raw]` com `training.train_fraction: 0.66` na campanha pareada — 66% ≈ retenção observada do IS. Cotas definidas com rótulos de treino |
| C3 | **Confiança simples (E2SC-like)** | ❌ | Novo método em `curriculum/methods/` (ex.: `confidence_discrete.py`) que remove por `p_i(y_i)` com a mesma retenção total. Registrar em `registry.py`. **Nomear como ablação inspirada em E2SC**, não como reprodução do método original |
| C4 | **Currículo simples, mesmos posteriors** | ⚠️ Parcial | `curriculum.margin_weight: 1.0 / entropy_weight: 0.0` (e o inverso) cobrem margem e entropia puros — já suportado. Fases aleatórias estratificadas exigem método novo |
| C5 | **Concorrente externo: EL2N dinâmico** | ❌ | Implementar EL2N dinâmico de NLU on Data Diets em `curriculum/methods/`. Incluir warm-up e scoring no custo total |

### Orçamento

| ID | Ação | Como |
|---|---|---|
| C6 | Comparar no **mesmo tempo total** | O `.tex` reporta `total time (s)`, que é wall-clock e não é comparável entre bs=16 e bs=32 — é a raiz do problema B2 |
| C7 | Registrar medidas adicionais de mecanismo | Adicionar a `summary.metrics` das campanhas: `n_updates`, `visits_per_id`, `tokens_processed`. Os dois últimos derivam de `train_history.csv` + `_token_count_total`. **Não são equivalentes entre si** |
| C8 | Custo completo | Contabilizar TF-IDF, ajustes OOF, calibração, seleção, ordenação, construção das fases, fine-tuning e checkpoints. Distinguir execução desde o início de cenário amortizado com reuso declarado e possível para todos os métodos |
| C9 | Produto esperado | Curvas **medidas** de macro-F1 × tempo acumulado por base, com checkpoints e incerteza. Selecionar configurações e checkpoints por validação. **Não interpolar** os quatro pontos finais atuais |

**Contas do cronograma (a confirmar nos logs).** Se cada época percorre o subconjunto elegível da fase, para M exemplos na entrada do currículo: `2 × (0,30M + 0,60M + 0,95M) = 3,70M` visitas, contra `6M` no treino sem fases. Com retenção de 66% de N, ≈ `2,44N` visitas. Os caps 96/160/256 também alteram custo e informação disponível.

---

## 6. Frente D — Ablações

Ordem diagnóstica: **Books e Reuters-90** (onde há perdas) e **AG News** (contraponto com perda pequena). Repetir nos demais só os contrastes que sustentarem a conclusão — não escolher a confirmação pelo resultado favorável.

### D.1 — Primeiro: comparar escores sobre a mesma seleção

| ID | Ação | Como |
|---|---|---|
| D1 | Guardar os mesmos sobreviventes BIOIS e a mesma matriz OOF | Requer novo parâmetro `curriculum.signal_override` (ou método `frozen_discrete`) que consome uma matriz OOF persistida em disco |
| D2 | Comparar score composto vs. margem vs. entropia vs. atribuição aleatória estratificada | Pesos uniformes inicialmente. Manter quantidades por classe/fase, duração, caps e otimização |
| D3 | Elegibilidade final fixada | Contraste com IDs elegíveis da fase final congelados pela referência + sensibilidade com `curriculum.q_high: 1.0` (já suportado). Separa efeitos de exclusão final e exposição |
| D4 | Teste de ordem pura | Mesmos minibatches, ocorrências por ID, pesos e caps, mudando só a sequência. **Item mais caro** — requer seed de shuffle fixa e reordenação de batch, que não existem hoje. Só fazer se o texto atribuir o ganho especificamente à ordem |
| D5 | Congelar indicadores e pesos da referência | Recalculá-los pela nova ordem muda ranking e ponderação ao mesmo tempo. Identificar os controles congelados como **diagnósticos**, não como a receita operacional |

### D.2 — Depois: remover componentes um a um

| Componente | Contraste focal em IS+CL | Como |
|---|---|---|
| **Piso de risco** | Com e sem `max(d, γ)`, elegibilidade final comum, pesos por ID congelados | Flag nova em `biois_discrete.py` |
| **Ponderação** | Sem peso de redundância; sem peso de ruído; todos unitários. Manter as atribuições de fase | `curriculum.beta: 0.0` desliga redundância (já suportado). O peso de ruído (`λn`) exige flag — hoje é hardcoded |
| **Comprimento** | Sem prior de comprimento; separadamente caps 96/160/256 vs. 256/256/256, mantendo IDs e fases | `curriculum.length_weight: 0.0` e `curriculum.phase_max_lengths: [256,256,256]` — **ambos já suportados** |

### D.3 — O que **não** precisa virar grade completa

Ranks globais vs. por classe, ordem inversa, diversas taxas de poda e substituições em cada etapa são testes direcionados. Priorizar os necessários para a alegação mantida ou para uma hipótese indicada pelos logs. A seleção BIOIS herdada não precisa ser reinventada neste artigo.

### Já pronto para rodar sem código novo

| Campanha | Escopo | Custo |
|---|---|---|
| `experiments/campaigns/beta_theta_sweep.yaml` | 3×3 grid de β/θ, 3 datasets, modo `is` | ~24 GPU-h |
| `experiments/campaigns/cl_params_ablation_multi.yaml` | schedule / signal / compute, 4 datasets, modo `cl` | conforme config |

---

## 7. Frente E — Diagnóstico das perdas

Em Reuters-90: IS-only tem F1 0,418 em 390 s; IS+CL tem 0,344 em 375 s. **O ganho adicional de 15 s acompanha uma perda de 7,4 pontos percentuais.** Caso prioritário, sem concluir antecipadamente que a causa são as classes minoritárias.

| ID | Ação | Como |
|---|---|---|
| E1 | Contagens e cobertura por fold e classe | original → removido por redundância → removido por ruído → sobrevivente → elegível em cada fase. `n_removed` e `is_reduction` já são métricas; falta o corte por classe. Script novo em `scripts/` lendo `phase_metrics.csv` |
| E2 | Identificar classes com poucos ou nenhum exemplo | e a regra aplicada nesses casos |
| E3 | Qualidade e exposição | Precisão, recall e F1 por classe nos 4 braços; confusões do probe; número de visitas, tokens truncados e pesos efetivos |
| E4 | Inspecionar a loss | Combinação entre cross-entropy por classe e pesos por exemplo, inclusive a normalização (`curriculum/imbalance_losses.py`) |
| E5 | Intervenção direcionada | Só quando os logs apontarem hipótese. Ao restaurar exemplos, controlar tamanho e classe quando possível; caso contrário, declarar que aumentou a quantidade de dados. Decidir com treino/validação |

### Teste de mecanismo: ruído injetado

`experiments/campaigns/label_noise_injection.yaml` existe e **não roda** — está bloqueada num patch de 7 arquivos.

| ID | Ação |
|---|---|
| E6 | Aplicar o patch: `config/defaults.py`, `config/schema.py`, `config/loader.py`, `cli/args.py`, `cli/experiment.py:~120`, `pipeline/runner.py` (flip após o split do fold, antes do classificador fraco), `results/recorder.py` (persistir índices invertidos) |
| E7 | **Sanity check obrigatório:** re-rodar `--dry-run` e confirmar que as linhas de comando carregam flag de ruído **cujo valor muda** entre r0.0 / r0.2 / r0.4. Um dry-run que sai 0 **não** é evidência de que a chave está ligada |
| E8 | **Go/no-go:** reuters90 a ρ=0,4, modos `[raw, is]`, fold 0. ~2 GPU-h, menos de 1 h. Se o recall de recuperação do θ estiver próximo do acaso (≈1/n_classes), a claim de mecanismo não sobrevive e a campanha deve ser **descartada**, não rodada a custo cheio (~66 GPU-h) |
| E9 | Manter "risco potencial de ruído" sem o teste. Corrupção/duplicação artificiais são extensões condicionadas à alegação de detecção; sucesso em ruído injetado **não** valida ruído natural |

---

## 8. Frente F — Auditoria dos baselines implementados

| ID | Baseline | O que falta | Onde |
|---|---|---|---|
| F1 | **Loss** | Documentar a cabeça de classificação, seu treinamento e os dados usados. O encoder sozinho não define os logits das classes; uma cabeça aleatória produz loss calculável, mas sua utilidade como dificuldade precisa de validação | `curriculum/methods/heuristic_discrete.py` |
| F2 | **PMD** | Publicar prompt e verbalizers por classe/fold; verificar tokens únicos e sua qualidade na validação. A escolha automática por TF-IDF é uma **adaptação**. O artigo de referência é do ACL Student Research Workshop | `curriculum/methods/self_adaptive_pmd.py`; artefatos `self_adaptive_verbalizers.json` e `self_adaptive_scores.csv` já são salvos |
| F3 | **Bengio, SPDCL, TD** | Distinguir inspiração, adaptação controlada e reprodução fiel. Especificar scoring, estados/camadas, agregação temporal e custos reais | `baselines/b1_bengio2009.py`, `baselines/b2_spdcl.py`, `heuristic_discrete.py` |
| F4 | Todos | Corrigir e repetir **apenas** os resultados afetados por um problema comprovado | — |

---

## 9. Frente G — Estatística e protocolo

| ID | Ação | Como |
|---|---|---|
| G1 | Parear folds e seeds | Começar com 3 execuções por fold, aproveitando a original se recuperável. Registrar seeds de seleção, inicialização e amostragem. Hoje tudo usa `random_state: 42` |
| G2 | Publicar diferenças pareadas | Folds têm treinos sobrepostos: 5 folds × 3 seeds **não** equivalem a 15 amostras independentes |
| G3 | Substituir o t unilateral por bootstrap pareado | **Já implementado** em `results/equivalence.py` (`bio-equivalence`, δ=0,01, CI 90%, 10k reamostragens) — mas cai no fallback `bootstrap_level="fold"` porque não existe nenhum `predictions_test.csv` |
| G4 | Rejustificar δ e confiança | O limite unilateral de 90% corresponde a α=0,10; `t_{0,90;4}=1,53` é coerente, mas exige hipóteses não garantidas pelos 5 folds. Justificar a margem de 0,01 **antes** de nova confirmação. Sem suporte inferencial adequado, relatar o trade-off e **retirar a conclusão formal forte** |
| G5 | Distinguir não inferioridade de equivalência | Ausência de significância não estabelece equivalência |
| G6 | Calcular a interação | Por fold/seed: `I = F(IS+CL) − F(IS) − F(CL) + F(Raw)`, com incerteza pareada. O contraste avalia as **receitas**; não identifica o componente responsável. Menor tempo, sozinho, não demonstra sinergia |
| G7 | Família de testes | Para múltiplas conclusões confirmatórias, definir a família; ajustes não consertam dependência mal modelada. `results/significance.py` já faz Holm-Bonferroni + Wilcoxon |
| G8 | Separar os dados | Documentar teste externo, validação, folds internos e calibração. Para scores inteiramente OOF, ajustar TF-IDF em cada treino interno. Se o IDF usa todos os textos do treino externo, **declarar a escolha**, distinguindo acesso a textos internos de vazamento de rótulos |
| G9 | Recuperar o histórico | Todos os `config.json` têm `git_commit: None` — as execuções vieram de imagem Docker obsoleta. Relatar versões, conversões single-label, IDs, grupos/duplicatas, tokenizer e estado do otimizador/scheduler entre fases. Warm-start dos parâmetros **não** esclarece esses estados |
| G10 | Robustez ≠ confirmação | Novas seeds nas seis bases avaliam robustez; não tornam testes já examinados uma confirmação independente |

---

## 10. Frente H — Formato e submissão

| ID | Ação | Detalhe |
|---|---|---|
| H1 | Reconciliar AG News | Tabela 2 (IS+CL) = `0,945` vs. Tabela 6 (BIOIS-CL) = `0,944` — `acl_latex.tex:373` vs. `:611` |
| H2 | Yelp: intervalos repetidos | `0,659 ± 0,002` aparece nas Tabelas 4 e 5 (`:482`, `:556`) |
| H3 | Diferenças derivadas de médias arredondadas | Recalcular a partir dos dados brutos |
| H4 | Regenerar tabelas dos dados brutos | `bio-summary` + `results/paper_tables.py`, com identificação de runs. **Depende de B3** |
| H5 | Cortar de ~11 para 8 páginas | O `.tex` nunca foi compilado. Compilar e contar; condensar redundâncias na introdução, revisão e método; levar detalhes suplementares ao apêndice identificado |
| H6 | Tabela 6 com localização e referência claras | — |
| H7 | Adicionar Ethical Considerations | **Ausente.** Hoje há só `Limitations` (`:670`). Obrigatória na ACL |
| H8 | Atualizar Limitations | Já existe; incluir escopo de tarefa (single-label) e diversidade de encoder |
| H9 | Remover agradecimento de infraestrutura da versão anônima | `:673-674`. Preservar os detalhes de hardware no método (`:313`) |
| H10 | Conferir requisitos do ciclo | Anonimização de repositórios, checklist Responsible NLP, uso de IA, requisitos de serviço do ARR |

---

## 11. Sequência de execução

> **Runbook operacional:** [`experimentos_maquina.md`](experimentos_maquina.md) — lista os `.yaml` de cada bloco (com o conteúdo completo dos novos), o plano de fatiamento por GPU e a ordem de submissão.

| Etapa | Executar e entregar | Prioridade |
|---|---|---|
| **1** | Bloqueadores **B1–B4** | Imediata |
| **2** | **factorial_matched_2x2** — gera `predictions_test.csv`, destrava o bootstrap pareado, o corte por classe e o controle Raw@2ep/4ep (C1) | Central |
| **3** | **beta_theta_sweep** + **go/no-go de ruído (E8)** em paralelo | Central |
| **4** | Frentes A e B — texto e related work. **Não depende de GPU** | Central |
| **5** | Aleatório estratificado (C2), confiança simples (C3), EL2N (C5) | Central |
| **6** | Ablações focais D1–D5 e diagnóstico E1–E5 | Central |
| **7** | Ampliar contrastes decisivos; código anônimo e versão de 8 páginas | Fechamento |

### Custo estimado

| Campanha | Custo | Nota |
|---|---|---|
| `factorial_matched_2x2` completo (6 datasets) | ~246 GPU-h | medline = 151,1; yelp_2013 = 65,1; agnews = 12,3; dblp = 7,7; books = 6,9; reuters90 = 2,7 |
| **idem, sem medline** | **~95 GPU-h** | corta 61% do custo; caminho crítico cai de ~9-10 h para ~1,1 h |
| `beta_theta_sweep` | ~24 GPU-h | ~16 se restrito a reuters90 + books |
| `label_noise_injection` | ~66 GPU-h | **só depois do go/no-go** de ~2 GPU-h |

**Núcleo recomendado (blocos Y1–Y5 do runbook): ≈ 135 GPU-h.**

Ordem barato-primeiro: reuters90 → books → dblp → agnews → yelp_2013 → medline.

**Não executar todos os fatores em todas as bases de início.** Reaproveitar trajetórias, posteriors e previsões. Decisões técnicas fixadas em treino/validação; resultados de teste já utilizados reconhecidos no histórico. Se surgir vazamento ou erro real, corrigir antes de ampliar.

---

## 12. Critérios de decisão

| Se… | Então… |
|---|---|
| Raw mais curto igualar o resultado | A economia frente a seis épocas não estabelece o valor da seleção. Verificar outra região de orçamento e limitar a conclusão ao que as curvas mostrarem |
| Um currículo simples empatar com o composto | Remover os termos sem benefício demonstrado, ou reformular o artigo em torno da versão simples. **Não** usar a ausência de significância como prova de igualdade |
| Houver vantagem só em parte das bases | Delimitar as condições favoráveis, quantificar as perdas, evitar conclusão geral de preservação de qualidade. Explicações sobre tamanho ou desbalanceamento precisam de evidência própria |
| O recall do θ estiver no acaso | A claim de mecanismo não sobrevive; remover a narrativa de remoção de ruído |
| A vantagem resistir aos controles | A contribuição passa a ter suporte claro: ganho operacional, comparação pertinente e explicação do que os componentes acrescentam |

### Ampliação recomendada

Priorizar **um** novo dataset de classificação que acrescente condição relevante e não tenha orientado o desenvolvimento. Um segundo encoder é importante para afirmar transferência entre encoders; pode ficar como extensão se o escopo permanecer em ModernBERT-base. Não exigir automaticamente geração, grandes decoders ou vários idiomas. **Nenhuma ampliação substitui os controles centrais.**

---

## 13. Entregáveis da próxima rodada

- [ ] Quadro das comparações executadas e pendentes
- [ ] Tabelas geradas dos dados brutos, com identificação de runs
- [ ] Curvas qualidade × tempo acumulado
- [ ] Ablações centrais em IS+CL
- [ ] Diagnóstico por classe
- [ ] Descrição verificável dos splits e baselines
- [ ] Resumo de quais conclusões sobreviveram aos controles

A decisão de submeter deve se apoiar nesse conjunto.

---

## 14. Rastreabilidade

| Ref | Fonte |
|---|---|
| [1] | `paper/acl_latex.tex` — manuscrito, 684 linhas, 6 datasets, Tabelas 1–6 |
| [2] | Dai et al. 2026 — *Demystifying Data Organization for Enhanced LLM Training*, ACL main, seção 2 |
| [3] | Sun et al. 2023 — *Data Selection Curriculum for Abstractive Text Summarization*, Findings of EMNLP |
| [4] | Cunha et al. 2025 — *A Noise-Oriented and Redundancy-Aware Instance Selection Framework*, TOIS (`Cunha2025`) |
| [5] | Attendu & Corbeil 2023 — *NLU on Data Diets*, SustaiNLP |
| [6] | Bengio & Grandvalet 2004 — *No Unbiased Estimator of the Variance of K-Fold Cross-Validation*, JMLR |
| [7] | Feng et al. 2025 — *Your Pretrained Model Tells the Difficulty Itself*, ACL SRW (`Feng2025`) |
| [8] | ARR Call for Papers — requisitos oficiais |
| [9] | ARR Common Submission Problems — checklist oficial |
| [10] | Quatro relatórios do Elicit — originalidade; baselines e custo; ablação e diagnóstico; validade estatística |

**Limite de escopo:** o parecer combina a leitura do manuscrito e os relatórios do Elicit. Código, logs e previsões individuais **não** foram auditados no parecer original; a auditoria de repositório deste documento é a verificação adicional. Pontos de implementação não esclarecidos permanecem como verificações pendentes.
