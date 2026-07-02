# Models

No model weights are produced or committed here (`models/` is git-ignored). The
"models" are a small policy MLP and a small reward MLP, both trained **in memory**
by torch at run time — there is nothing to persist. This folder exists only to
keep the project skeleton identical to the rest of the monorepo.
