"""Streaming execution: real-time book updates -> instant edge check -> execute.

The design splits slow from fast:

  - **Slow** (every few minutes): the existing scan + embed-match + LLM-confirm +
    date-gate pipeline produces the set of *confirmed same-event pairs*. Cached.
  - **Fast** (this module): subscribe both venues' WebSockets to just those pairs'
    markets; on every book update, recompute the edge from live top-of-book and fire
    the executor immediately when it clears ``min_edge`` + risk.

This is what turns "detect on a 60s tick" into "act in milliseconds."
"""
