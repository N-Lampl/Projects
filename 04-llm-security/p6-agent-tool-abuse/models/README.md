# models/ (git-ignored)

No trained model artifacts. The default agent "brain" is a deterministic,
rule-based **mock LLM** (`src/agent_tool_abuse/llm.py: MockLLM`) — no weights,
no checkpoint. The optional real-LLM path calls a hosted API and likewise
stores nothing here.
