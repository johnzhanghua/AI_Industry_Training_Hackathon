"""Optional LangSmith tracing.

Tracing is opt-in through `LANGSMITH_TRACING=true` and is never load-bearing: if
langsmith is missing or unconfigured, `traceable` degrades to a transparent
pass-through decorator. A tracing problem must never fail a graded /query call,
so nothing here is allowed to raise at import time.
"""
import logging
import os

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def _tracing_requested() -> bool:
    """LANGSMITH_TRACING is the current name; LANGCHAIN_TRACING_V2 is the legacy one."""
    return (
        os.getenv("LANGSMITH_TRACING", "").strip().lower() in _TRUTHY
        or os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() in _TRUTHY
    )


def _resolve() -> bool:
    if not _tracing_requested():
        return False

    if not os.getenv("LANGSMITH_API_KEY"):
        logger.warning(
            "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is unset -- "
            "tracing disabled."
        )
        return False

    try:
        import langsmith  # noqa: F401
    except ImportError:
        logger.warning("langsmith is not installed -- tracing disabled.")
        return False

    # LangGraph/langchain-core check the legacy name on some code paths, so mirror
    # it. This is what makes the graph's nodes auto-trace under our root run.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    return True


TRACING_ENABLED = _resolve()
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT") or "financial-market-agent"

if TRACING_ENABLED:
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
    from langsmith import traceable

    logger.info("LangSmith tracing enabled (project=%s)", LANGSMITH_PROJECT)
else:
    def traceable(*args, **kwargs):
        """No-op stand-in supporting both @traceable and @traceable(run_type=...)."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator
