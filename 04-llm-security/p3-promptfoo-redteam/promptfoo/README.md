# promptfoo/ — the optional eval-as-code red-team path

`promptfoo` is a **Node** CLI (there is no pip package), so this path needs
`node >= 18` and `npx`. The default `make redteam` path does NOT need any of
this — it's the pure-python harness in `../scripts/run_redteam.py`.

## What's here

- `promptfooconfig.yaml` — the eval config. Uses the **`owasp:llm`** red-team
  preset plus `jailbreak` / `prompt-injection` strategies. The target is the
  local mock RAG (`../../p4-vulnerable-rag`) reached through an `exec:` provider,
  so even this path makes **no real LLM API calls**.
- `target_provider.py` — the bridge promptfoo calls: it reads the rendered
  prompt and returns `load_target().query(prompt)` (the same target the python
  harness attacks).

## Run it (needs node/npx)

```bash
cd promptfoo
npx promptfoo@latest redteam run -o results.json   # generates + runs adversarial tests
python ../scripts/parse_promptfoo.py results.json  # -> ../results/metrics.json + chart
npx promptfoo@latest view                          # optional interactive report UI
```

`parse_promptfoo.py` maps each promptfoo plugin to its OWASP-LLM category and
emits the **same metrics.json shape** as the offline harness, so either path
feeds the portfolio dashboard identically.

> Authorized use only — see [../../../ETHICS.md](../../../ETHICS.md). The target
> is a self-built local lab app with synthetic data.
