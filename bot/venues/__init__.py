"""Venue adapters. Each normalizes raw payloads into ``bot.models`` types.

The pure ``normalize_*`` functions are import-light and unit-tested. The networked
clients import ``httpx`` / ``websockets`` / ``cryptography`` lazily so importing a
``normalize_*`` helper never requires those packages.
"""
