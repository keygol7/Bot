"""Self-hosted trading bot for Kalshi, Polymarket US, and opt-in Polymarket.com.

The package is split so the deterministic, latency-critical core (models, fees,
arbitrage, risk, in-memory book, persistence) imports only the standard library.
Venue adapters and the local-LLM analyst import their heavier dependencies lazily,
so the test suite for the core runs with nothing installed but pytest.
"""

__version__ = "0.1.0"
