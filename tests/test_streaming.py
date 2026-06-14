"""Streaming engine core: live book -> per-tick edge check -> execute (fake executor)."""

import asyncio

from bot.fees import ZeroFeeModel
from bot.models import MarketQuote
from bot.streaming.engine import ConfirmedPair, LiveBook, StreamingEngine


class FakeExec:
    def __init__(self):
        self.calls = []

    async def execute(self, opp):
        self.calls.append(opp)
        return f"executed {opp.event_key}"


def q(venue, mid, yes_ask=None, ya=0.0, no_ask=None, na=0.0):
    return MarketQuote(venue=venue, market_id=mid, title="", yes_ask=yes_ask,
                       yes_ask_size=ya, no_ask=no_ask, no_ask_size=na)


def make_engine(executor, cooldown=100.0, now=0.0):
    clock = lambda: now
    eng = StreamingEngine(
        executor=executor,
        fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=cooldown, clock=clock,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    return eng


def test_index_built_from_pairs():
    eng = make_engine(FakeExec())
    assert ("kalshi", "K1") in eng._index and ("poly", "P1") in eng._index
    assert eng.market_ids["kalshi"] == ["K1"] and eng.market_ids["poly"] == ["P1"]


def test_executes_when_edge_appears():
    fe = FakeExec()
    eng = make_engine(fe)
    # First leg quote alone -> no counterpart yet -> nothing.
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)))
    assert fe.calls == []
    # Second leg arrives -> YES@kalshi 0.40 + NO@poly 0.55 = 0.95 -> edge 0.05 -> execute.
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    assert len(fe.calls) == 1
    opp = fe.calls[0]
    assert opp.buy_yes_venue == "kalshi" and opp.buy_no_venue == "poly"
    assert opp.max_contracts == 60 and round(opp.edge_per_contract, 4) == 0.05


def test_no_execute_without_edge():
    fe = FakeExec()
    eng = make_engine(fe)
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.55, ya=100, no_ask=0.55, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.55, ya=100, no_ask=0.55, na=100)))
    assert fe.calls == []   # 0.55+0.55 > 1, no arb either direction


def test_cooldown_prevents_refire():
    fe = FakeExec()
    eng = make_engine(fe, cooldown=100.0)
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    assert len(fe.calls) == 1   # second update within cooldown -> not re-fired


def test_picks_better_direction():
    fe = FakeExec()
    eng = make_engine(fe)
    # Cheaper to buy YES on poly + NO on kalshi.
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.70, ya=100, no_ask=0.42, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.50, ya=80, no_ask=0.70, na=100)))
    assert len(fe.calls) == 1
    assert fe.calls[0].buy_yes_venue == "poly" and fe.calls[0].buy_no_venue == "kalshi"


def test_run_consumes_streams_and_stops(monkeypatch):
    fe = FakeExec()
    eng = make_engine(fe)

    class StreamVenue:
        def __init__(self, name, quotes):
            self.name = name
            self._quotes = quotes

        async def stream_order_book(self, mids):
            for x in self._quotes:
                yield x
            # keep the generator open briefly so both venues' quotes arrive
            await asyncio.sleep(0.01)

    venues = [
        StreamVenue("kalshi", [q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)]),
        StreamVenue("poly", [q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)]),
    ]

    async def refresh():
        return [ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")]

    async def driver():
        task = asyncio.create_task(eng.run(venues, refresh, refresh_interval=0.05))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(driver())
    assert len(fe.calls) >= 1
