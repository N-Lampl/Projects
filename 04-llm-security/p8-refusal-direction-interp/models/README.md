# models/ (git-ignored)

The default offline path uses an in-code **synthetic** frozen model
(`src/refusal_interp/synthetic.py`); nothing is saved here.

The optional real-model path downloads an open-weight instruct model via
`transformers` (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) into the HuggingFace cache,
not this folder. Downloaded weights are never committed.

**Ethics:** the committed artifact of this project is **analysis** -- a direction
vector, metrics, and figures. We never write out or redistribute a modified
("abliterated") model. See ../../ETHICS.md.
