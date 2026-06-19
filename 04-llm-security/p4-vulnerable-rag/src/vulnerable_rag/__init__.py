"""Vulnerable RAG lab target -- the deliberately-insecure app attacked by p5/p6/p7.

Public API:
    set_seed, get_device          -- reproducibility helpers
    build_corpus, corpus_stats    -- the planted knowledge base (PII/secret/bait)
    TfidfRetriever, DenseRetriever -- retrievers (TF-IDF default; dense optional)
    get_provider, MockProvider    -- pluggable LLM client (offline mock default)
    VulnerableRAG, QueryResult    -- the RAG app; query() and query_detailed()
"""

from .app import SYSTEM_PROMPT, QueryResult, VulnerableRAG
from .corpus import PLANTED_API_KEY, Document, build_corpus, corpus_stats
from .providers import MockProvider, get_provider
from .retriever import DenseRetriever, RetrievalResult, TfidfRetriever
from .tools import available_tools, lookup_account
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "build_corpus",
    "corpus_stats",
    "Document",
    "PLANTED_API_KEY",
    "TfidfRetriever",
    "DenseRetriever",
    "RetrievalResult",
    "get_provider",
    "MockProvider",
    "VulnerableRAG",
    "QueryResult",
    "SYSTEM_PROMPT",
    "lookup_account",
    "available_tools",
]
