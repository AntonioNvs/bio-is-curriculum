# Baselines da literatura

Catálogo dos baselines reproduzidos para comparação com o `is_cl` (BIOIS-Curriculum). Cada baseline tem um índice estável, ativado via `--baseline N` no `main.py`. O backbone e os hiperparâmetros de fine-tuning (otimizador, lr, batch size, max_length) são mantidos idênticos aos demais modos para isolar o efeito do método.

**Nota:** baselines da literatura (`--baseline N`) são distintos dos métodos de curriculum (`--curriculum-method`), que incluem `biois_discrete`, `spcl_soft` e `spcl_loss`. O modo `is_continuos_cl` é um alias IS+CL com `spcl_soft` por default.

| Índice | Nome curto | Tipo | Referência |
|---|---|---|---|
| 1 | Confidence-paced CL (Bengio 2009) | Currículo ingênuo | Bengio et al. (2009) |

---

## `--baseline 1` — Confidence-paced Curriculum Learning (Bengio et al. 2009)

**O que é.** Currículo de exemplos do mais fácil para o mais difícil, onde "dificuldade" é definida exogenamente por um classificador fraco. É a formulação canônica de CL e o controle-padrão pra qualquer método de CL mais sofisticado.

**Implementação aqui.** Reaproveita o `_probaEveryone` que o `BIOIS.fitting_alpha` já calcula (LR multinomial em 5-fold cross-val sobre o TF-IDF), evitando treinar um classificador adicional:

1. Para cada instância `i` com rótulo `y_i`, calcula a confiança no rótulo verdadeiro: `conf_i = probaEveryone[i, y_i]`.
2. Ordena as instâncias por `conf_i` decrescente (mais fáceis primeiro).
3. Particiona em 3 fases pelos quantis (`q_low`, `q_mid`, 1.0) — defaults `0.3 / 0.6 / 1.0`, idênticos aos do `is_cl` para comparação justa.
4. Treina o RoBERTa em 3 estágios com pool crescente (fase 1 = 30% mais fáceis; fase 2 = 60%; fase 3 = 100%), `--epochs-per-phase` épocas por estágio, **sem `sample_weight`** e **sem remover ruído**.

**O que ele NÃO faz** (e o `is_cl` faz):
- Não remove instâncias ruidosas (alta confiança em predição errada → entropia baixa em classe errada).
- Não pondera redundância na fase hard (`w = 1 - β·r`).
- Não reduz o conjunto de treino — o pool final é 100%.

**Por que é o controle certo.** Ao manter o backbone, o classificador fraco, o esquema de fases e os quantis idênticos ao `is_cl`, qualquer ganho de `is_cl` sobre este baseline é atribuível especificamente aos sinais de **ruído + redundância** do BIOIS no curriculum — não a "currículo qualquer ajuda".

**Referência.**
Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). *Curriculum Learning*. In *Proceedings of the 26th International Conference on Machine Learning (ICML)*, 41–48. <https://doi.org/10.1145/1553374.1553380>
