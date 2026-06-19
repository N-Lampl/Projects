# data/ (git-ignored)

The **default offline path needs no dataset** -- residual activations are
*synthesised* in code (`src/refusal_interp/synthetic.py`) with a planted refusal
direction. Nothing is downloaded, nothing is committed here.

## Optional enhanced path (real model)

If you run the optional real-transformer extraction (see the README "Real-model
path"), you provide two small prompt sets yourself:

- **Harmful set:** AdvBench-style harmful-instruction prompts.
  - Source: *Universal and Transferable Adversarial Attacks on Aligned Language
    Models* (Zou et al., 2023), `harmful_behaviors.csv` (520 rows, MIT license).
  - Download: `git clone https://github.com/llm-attacks/llm-attacks` then use
    `data/advbench/harmful_behaviors.csv`.
- **Harmless set:** Alpaca-style instructions.
  - Source: Stanford Alpaca (`alpaca_data.json`, CC BY-NC 4.0 - research only).
  - Download: `https://github.com/tatsu-lab/stanford_alpaca`.

Use a small subset (e.g. 128 of each) for CPU. These are git-ignored; never
commit them.

> Authorized-use note: harmful prompts are used here **only** to compute and
> analyse internal activations of a model you loaded yourself. See ../../ETHICS.md.
