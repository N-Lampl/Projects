# Models

No model weights are produced or committed here (`models/` is git-ignored). The
"model" is a tiny 2-layer decoder-only transformer trained **in memory** from
scratch at run time (a few seconds on CPU) - there is nothing to persist. The
optional `@slow` test downloads **distilgpt2** on demand via `transformers`; its
weights are cached by the Hugging Face hub outside this repo. This folder exists
only to keep the project skeleton identical to the rest of the monorepo.
