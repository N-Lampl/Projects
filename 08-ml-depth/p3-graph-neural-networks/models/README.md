# Models

No model weights are produced or committed here (`models/` is git-ignored). The
"model" is a 2-layer GCN (and its graph-blind MLP twin) trained **in memory** at run
time in a few seconds on CPU — there is nothing to persist. This folder exists only
to keep the project skeleton identical to the rest of the monorepo.
