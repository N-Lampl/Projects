# models/ (git-ignored)

The default path uses **no trained model** -- the LLM is a deterministic offline
mock (`src/vulnerable_rag/providers.py::MockProvider`) and the retriever is a
TF-IDF vectorizer fit at runtime.

If you enable the optional dense retriever (`sentence-transformers`), its
downloaded weights (e.g. `all-MiniLM-L6-v2`) are cached by that library outside
this folder. Nothing here is committed.
