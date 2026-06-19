# data/ (git-ignored)

This project uses **no external dataset**. The "world" the agent acts on is a
tiny in-memory mock filesystem, outbox and query log defined in
`src/agent_tool_abuse/tools.py` (`ToolWorld`), and the attack/benign scenarios
are defined in `src/agent_tool_abuse/agent.py` (`default_scenarios`).

- **Dataset:** none — synthetic scenarios + a mock LLM provider.
- **Download:** nothing to download; `make attack` runs fully offline.

Nothing in this folder is committed.
