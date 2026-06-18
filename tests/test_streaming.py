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


def test_edge_snapshot_only_includes_two_sided_pairs(caplog):
    import logging
    # The snapshot proves WS prices are matched to events: a pair appears only when
    # BOTH legs have a live quote in the book.
    eng = make_engine(FakeExec())
    eng.set_pairs([
        ConfirmedPair("E1", "kalshi", "K1", "poly", "P1"),
        ConfirmedPair("E2", "kalshi", "K2", "poly", "P2"),
    ])
    # Only E1 gets both legs; E2 gets one leg -> excluded from the snapshot.
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    asyncio.run(eng.on_quote(q("kalshi", "K2", yes_ask=0.50, ya=100, no_ask=0.55, na=100)))

    snap = eng.edge_snapshot()
    assert [r[1].event_key for r in snap] == ["E1"]      # only the two-sided pair
    edge, p, yq, nq, size = snap[0]
    assert round(yq.yes_ask + nq.no_ask, 2) == 0.95 and round(edge, 2) == 0.05
    with caplog.at_level(logging.INFO, logger="bot.streaming"):
        asyncio.run(eng.log_edge_snapshot())   # no depth_fetch -> uses live-book values
    assert "1/2 pairs two-sided" in caplog.text


def test_sizeless_ws_quote_triggers_depth_fetch_then_executes():
    # Kalshi-style: WS quotes have a price edge but size 0. The engine must depth-fetch
    # real sizes before firing, then execute.
    fe = FakeExec()
    deep = {
        ("kalshi", "K1"): q("kalshi", "K1", yes_ask=0.40, ya=50, no_ask=0.65, na=50),
        ("poly", "P1"): q("poly", "P1", yes_ask=0.62, ya=50, no_ask=0.55, na=50),
    }

    async def depth_fetch(venue, mid):
        return deep.get((venue, mid))

    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, depth_fetch=depth_fetch,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    # Sizeless WS ticks (size 0) — would be skipped without the depth fetch.
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=0, no_ask=0.65, na=0)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=0, no_ask=0.55, na=0)))
    assert len(fe.calls) == 1
    assert fe.calls[0].max_contracts == 50          # real size came from the depth fetch


def test_depth_fetch_says_edge_gone_no_execute():
    # Price edge on the WS book, but the depth fetch shows the edge has evaporated.
    fe = FakeExec()

    async def depth_fetch(venue, mid):
        # Both legs now priced so the pair sums > 1 (no edge).
        return q(venue, mid, yes_ask=0.60, ya=50, no_ask=0.60, na=50)

    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, depth_fetch=depth_fetch,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=0, no_ask=0.65, na=0)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=0, no_ask=0.55, na=0)))
    assert fe.calls == []                            # depth fetch vetoed the stale edge


def test_prime_and_sweep_seeds_book_and_executes():
    # Neither leg has ticked over WS, but a REST snapshot prime should populate both
    # legs and fire the edge — closing the quiet-leg gap.
    fe = FakeExec()
    deep = {
        ("kalshi", "K1"): q("kalshi", "K1", yes_ask=0.40, ya=50, no_ask=0.65, na=50),
        ("poly", "P1"): q("poly", "P1", yes_ask=0.62, ya=50, no_ask=0.55, na=50),
    }

    async def depth_fetch(venue, mid):
        return deep.get((venue, mid))

    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, depth_fetch=depth_fetch,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    asyncio.run(eng.prime_and_sweep())               # no WS quotes were fed at all
    assert len(fe.calls) == 1                          # edge found purely from the prime


def test_inflight_execution_survives_consumer_cancellation():
    # The real bug: a trade fires in the last instant of an interval; run() then
    # cancels the consumer task that launched it. The shielded execute() must still
    # run to completion (not be aborted mid-order), and run() must wait for it.
    started = asyncio.Event()
    release = asyncio.Event()
    finished = []

    class SlowExec:
        async def execute(self, opp):
            started.set()
            await release.wait()        # simulate an in-flight leg placement
            finished.append(opp.event_key)
            return f"done {opp.event_key}"

    eng = make_engine(SlowExec())

    async def driver():
        # First leg seeds the book; the second leg (in a consumer-like task) is what
        # fires the trade — and that task is the one cancelled mid-flight.
        await eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100))
        consumer = asyncio.create_task(
            eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60))
        )
        await started.wait()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        assert finished == []                 # still in flight, NOT aborted by cancel
        assert eng._inflight                  # run() would wait on this
        release.set()
        await asyncio.gather(*list(eng._inflight), return_exceptions=True)
        assert finished == ["E1"]             # completed to a definitive outcome

    asyncio.run(driver())


def test_consume_counts_ws_quotes_for_health():
    # The WS-health heartbeat: _consume must count each tick per venue so the run
    # loop can report whether a venue's WebSocket is actually delivering data.
    eng = make_engine(FakeExec())

    class V:
        name = "kalshi"

        async def stream_order_book(self, mids):
            yield q("kalshi", "K1", yes_ask=0.4, ya=10, no_ask=0.6, na=10)
            yield q("kalshi", "K1", yes_ask=0.4, ya=10, no_ask=0.6, na=10)

    asyncio.run(eng._consume(V()))
    assert eng._ws_counts["kalshi"] == 2
