# Models

No model weights are produced or committed here (`models/` is git-ignored). The
teacher, its pruned / quantized variants, and the distilled student are all
trained **in memory** by `scripts/run_analysis.py` at run time and benchmarked on
the spot — there is nothing to persist. This folder exists only to keep the
project skeleton identical to the rest of the monorepo.
