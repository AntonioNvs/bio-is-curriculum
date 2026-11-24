# S3 independent verification report

Blind verifier `analysis/paper/verify_independent.py`: own resolver, own
metric (confusion-count macro-F1, bit-identical to `sklearn.metrics.f1_score`), own paired bootstrap with a different seed.

- point-estimate tolerance: 1e-09
- CI tolerance: 0.002
- bootstrap replicas: 5000; verifier seed 20261124 (generator seed 42)
- CI level: 90% percentile; rows: 64

## 1. Point estimates (all comparisons)

Recomputed delta_hat, f1_a, f1_b, n_instances and paired identity for all 64 rows: **64/64 exact to 1e-09**.

| comparison | verifier delta | generator delta | |diff| | n | ok |
|---|---|---|---|---|---|
| SPURS-Raw/agnews | -0.003630055421 | -0.003630055421 | 4.49e-14 | 127600 | yes |
| IS-Raw/agnews | -0.003089179850 | -0.003089179850 | 4.97e-13 | 127600 | yes |
| CL-Raw/agnews | -0.001164423453 | -0.001164423453 | 4.84e-13 | 127600 | yes |
| I/agnews | 0.000623547883 | 0.000623547883 | 2.61e-14 | 127600 | yes |
| SPURS-Raw@2/agnews | -0.005236992992 | -0.005236992992 | 7.49e-14 | 127600 | yes |
| SPURS-Raw@4/agnews | -0.002695463741 | -0.002695463741 | 1.59e-14 | 127600 | yes |
| IS-Random@ret/agnews | 0.000779638647 | 0.000779638647 | 2.89e-14 | 127600 | yes |
| RandomPhases-Composite/agnews | -0.000924229712 | -0.000924229712 | 1.28e-14 | 127600 | yes |
| *(all 64 rows)* | | | **max 4.34e-12** | | |

## 2. Reconstructed CIs (representative set)

| comparison | role | d_hat | CI low gen/ver/diff | CI high gen/ver/diff | ok |
|---|---|---|---|---|---|
| SPURS-Raw/agnews | Y1 SPURS-Raw | -0.00363006 | -0.004451 / -0.004441 / 9.97e-06 | -0.002821 / -0.002839 / 1.75e-05 | yes |
| I/agnews | four-arm interaction I (agnews) | 0.00062355 | -0.000327 / -0.000371 / 4.38e-05 | 0.001571 / 0.001547 / 2.38e-05 | yes |
| IS-Random@ret/agnews | IS vs Random@ret (Y1 x Y4) | 0.00077964 | -0.000006 / -0.000018 / 1.12e-05 | 0.001564 / 0.001564 / 2.89e-11 | yes |
| SPURS-Raw@2/books | Y2 comparison SPURS-Raw@2 (books) | -0.00459851 | -0.007050 / -0.007017 / 3.36e-05 | -0.002135 / -0.002199 / 6.43e-05 | yes |
| RandomPhases-Composite/agnews | Y3 control RandomPhases vs Composite | -0.00092423 | -0.001697 / -0.001705 / 7.50e-06 | -0.000130 / -0.000128 / 2.20e-06 | yes |
| theta0.2-theta0.0/beta0.1/books | Y5 theta0.2 vs theta0.0 (books beta0.1) | -0.00243317 | -0.005608 / -0.005600 / 8.25e-06 | 0.000694 / 0.000872 / 1.78e-04 | yes |

6/6 representative CIs within 0.002.

## 3. Paired idx / y_true identity

Per representative comparison, all arms over the same fold set; pooled
`idx` and `y_true` must be byte-identical across arms (paired bootstrap
precondition).

- `SPURS-Raw/agnews` (Y1 SPURS-Raw): folds [0, 1, 2, 3, 4], pooled idx/y_true identical across arms = True
- `I/agnews` (four-arm interaction I (agnews)): folds [0, 1, 2, 3, 4], pooled idx/y_true identical across arms = True
- `IS-Random@ret/agnews` (IS vs Random@ret (Y1 x Y4)): folds [0, 1, 2, 3, 4], pooled idx/y_true identical across arms = True
- `SPURS-Raw@2/books` (Y2 comparison SPURS-Raw@2 (books)): folds [0, 1, 2, 3, 4], pooled idx/y_true identical across arms = True
- `RandomPhases-Composite/agnews` (Y3 control RandomPhases vs Composite): folds [0, 1, 2, 3, 4], pooled idx/y_true identical across arms = True
- `theta0.2-theta0.0/beta0.1/books` (Y5 theta0.2 vs theta0.0 (books beta0.1)): folds [0, 1, 2], pooled idx/y_true identical across arms = True

All representative fold sets paired-identical: **True**.

## 4. Hard-F1 is descriptive only (no paired delta)

- hard columns: ['hard_f1_a_foldmean', 'hard_f1_b_foldmean']
- hard columns carrying a delta: none
- hard_f1_a_foldmean reproduces the runs_index fold mean on every row: **True**

## 5. Import hygiene

- torch/transformers in sys.modules: none
- bio_is_curriculum.* imported: none
- imports used: numpy, pandas (+ stdlib)

## Verdict

**PASS** - every point estimate exact to 1e-09; every representative CI within 0.002; paired identity holds; no hard-F1 delta; no torch/transformers.

