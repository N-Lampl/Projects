"""Agent tool-abuse: confused-deputy attacks on a tool-using LLM agent, and a
tool-call guardrail that stops them. Default path is a deterministic MOCK LLM.

Public API:
    set_seed, get_device          -- reproducibility helpers
    MockLLM, AnthropicLLM         -- the agent "brain" (mock default / optional real)
    ToolWorld, ToolCall           -- the mock tools + in-memory world
    tool_schemas, classify_call   -- tool definitions + ground-truth safety labels
    ToolGuardrail                 -- the defense (allow-list policy enforcement)
    Scenario, run_episode         -- the attack harness
    default_scenarios             -- the bundled attack/benign suite
    evaluate                      -- unsafe-rate before vs after the guardrail
"""

from .agent import EpisodeResult, Scenario, default_scenarios, run_episode
from .evaluate import evaluate
from .guardrail import GuardrailDecision, ToolGuardrail
from .llm import AnthropicLLM, MockLLM
from .tools import ToolCall, ToolWorld, classify_call, tool_schemas
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "MockLLM",
    "AnthropicLLM",
    "ToolWorld",
    "ToolCall",
    "tool_schemas",
    "classify_call",
    "ToolGuardrail",
    "GuardrailDecision",
    "Scenario",
    "EpisodeResult",
    "run_episode",
    "default_scenarios",
    "evaluate",
]
