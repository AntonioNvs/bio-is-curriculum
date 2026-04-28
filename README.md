## Exemplo de execução

```sh
uv run python main.py webkb --data_dir datasets --fold 0 --beta 0.3 --theta 0.2
```

### biO-IS-Curriculum (primeiro passo)

Executa o BIOIS apenas para gerar os escores de redundância e entropia
(`--beta 0.0 --theta 0.0` mantém todas as instâncias) e em seguida treina
um `LogisticRegression` em três fases (Clean → Diverse → Hard):

```sh
uv run python main.py webkb --data_dir datasets --fold 0 --beta 0.3 --theta 0.2 \
    --curriculum --curriculum-beta 0.5
```
