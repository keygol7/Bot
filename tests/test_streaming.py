"""Streaming engine core: live book -> per-tick edge check -> execute (fake executor)."""

import asyncio

from bot.fees import ZeroFeeModel
from bot.models import MarketQuote
from bot.streaming.engine import ConfirmedPair, StreamingEngine


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


def test_fresh_ws_book_skips_rest_depth_fetch():
    # When both legs have a recent, sized WS quote, fire off the live book WITHOUT the
    # REST depth re-fetch (the latency win once both venues stream sized depth).
    import time as _time

    fe = FakeExec()
    fetched = []

    async def depth_fetch(venue, mid):
        fetched.append((venue, mid))                # must NOT be called on a fresh book
        return q(venue, mid, yes_ask=0.99, ya=1, no_ask=0.99, na=1)  # would kill the edge

    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, depth_fetch=depth_fetch,
        max_ws_quote_age=2.0,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    now = _time.time()
    ka = q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100); ka.timestamp = now
    pa = q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60); pa.timestamp = now
    asyncio.run(eng.on_quote(ka))
    asyncio.run(eng.on_quote(pa))
    assert len(fe.calls) == 1 and fetched == []      # executed off WS, no REST fetch


def test_maker_mode_confirms_depth_even_on_fresh_ws_book():
    # In MAKER mode a fresh WS edge must still be REST-confirmed before resting a maker:
    # resting on a phantom WS top (real book has no edge) just gets cancelled and backs the
    # pair off. Here the real book kills the edge -> nothing is armed, despite a fresh quote.
    import time as _time

    fe = FakeExec()
    fetched = []

    async def depth_fetch(venue, mid):
        fetched.append((venue, mid))
        return q(venue, mid, yes_ask=0.99, ya=1, no_ask=0.99, na=1)   # real book: no edge

    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, depth_fetch=depth_fetch,
        max_ws_quote_age=2.0, maker_mode=True,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    now = _time.time()
    ka = q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100); ka.timestamp = now
    pa = q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60); pa.timestamp = now
    asyncio.run(eng.on_quote(ka))
    asyncio.run(eng.on_quote(pa))
    assert fetched != []                  # confirmed real depth despite the fresh WS book
    assert fe.calls == []                 # phantom edge -> nothing armed


def test_confirm_depth_fetches_both_legs_concurrently():
    # The two confirm fetches must be in flight at once (asyncio.gather) so the edge is
    # sampled on a near-simultaneous snapshot. Each fetch here blocks until BOTH have
    # started: if they ran sequentially the second would never start, the first would time
    # out, and only one leg would be recorded. Concurrency -> both start -> the gate opens.
    started = []
    both = asyncio.Event()

    async def depth_fetch(venue, mid):
        started.append(venue)
        if len(started) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 1.0)
        return q(venue, mid, yes_ask=0.40, ya=50, no_ask=0.65, na=50)

    eng = StreamingEngine(
        executor=FakeExec(), fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, depth_fetch=depth_fetch,
    )
    ev = asyncio.run(eng._confirm_depth(ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")))
    assert started == ["kalshi", "poly"] and ev is not None    # both in flight together


def test_stale_ws_quote_still_uses_rest_depth_fetch():
    # An old WS quote (beyond max_ws_quote_age) must fall back to the REST confirm.
    fe = FakeExec()
    deep = {("kalshi", "K1"): q("kalshi", "K1", yes_ask=0.40, ya=50, no_ask=0.65, na=50),
            ("poly", "P1"): q("poly", "P1", yes_ask=0.62, ya=50, no_ask=0.55, na=50)}
    calls = []

    async def depth_fetch(venue, mid):
        calls.append((venue, mid))
        return deep.get((venue, mid))

    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, depth_fetch=depth_fetch,
        max_ws_quote_age=2.0,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    # timestamps left at 0.0 (ancient vs time.time()) -> not fresh -> REST fallback.
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    assert len(fe.calls) == 1 and calls != []        # REST confirm ran
    assert fe.calls[0].max_contracts == 50           # size from the depth fetch


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


def test_rejected_pair_backs_off_then_retries():
    # A pair whose order keeps failing must not be hammered every tick: after a failed
    # attempt the engine backs it off, and only retries once the backoff elapses.
    from bot.execution.executor import ExecStatus, ExecutionReport
    from bot.execution.orders import OrderResult, OrderStatus

    now = {"t": 1000.0}

    class RejectExec:
        def __init__(self):
            self.calls = 0

        async def execute(self, opp):
            self.calls += 1
            leg = OrderResult("kalshi", "K1", None, "buy", 1, status=OrderStatus.REJECTED)
            return ExecutionReport(ExecStatus.SKIPPED, "leg1 not filled (REJECTED)", [leg])

    fe = RejectExec()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=5.0, clock=lambda: now["t"],
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])

    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    assert fe.calls == 1                          # fired once, then rejected -> backoff

    now["t"] += 10                                # past the 5s cooldown, but inside backoff (60s)
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    assert fe.calls == 1                          # still backed off, not retried

    now["t"] += 60                                # backoff elapsed -> retries
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    assert fe.calls == 2


def test_skips_when_a_leg_is_not_open():
    # The Polymarket leg reports a non-OPEN state on its quote -> don't fire.
    fe = FakeExec()
    eng = make_engine(fe)
    pa = q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)
    pa.state = "MARKET_STATE_HALTED"
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)))
    asyncio.run(eng.on_quote(pa))
    assert fe.calls == []                          # halted leg -> skipped


def test_skips_when_lifecycle_marks_leg_not_open():
    # Kalshi's ticker carries no state; the lifecycle channel feeds it via
    # set_market_state. A terminated market must block the fire.
    fe = FakeExec()
    eng = make_engine(fe)
    eng.set_market_state("kalshi", "K1", "MARKET_STATE_TERMINATED")
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    assert fe.calls == []


def test_open_state_still_trades():
    fe = FakeExec()
    eng = make_engine(fe)
    pa = q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)
    pa.state = "MARKET_STATE_OPEN"
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)))
    asyncio.run(eng.on_quote(pa))
    assert len(fe.calls) == 1                       # OPEN (poly) + unknown (kalshi) -> trades


def test_skips_settling_market_at_price_extreme():
    # A leg at ~$0.01 is a resolved/settling market (phantom depth) -> skip the fire.
    fe = FakeExec()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, min_leg_price=0.02,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    # poly YES @ 0.01 (extreme) + kalshi NO @ 0.90 -> 9c "edge" but it's settling.
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.90, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.01, ya=100, no_ask=0.60, na=100)))
    assert fe.calls == []                          # skipped: leg at price extreme


def test_normal_prices_not_blocked_by_extreme_guard():
    fe = FakeExec()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, min_leg_price=0.02,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    assert len(fe.calls) == 1                       # 0.40/0.55 are not extreme -> trades


class RecordingStore:
    def __init__(self):
        self.edges = []

    def record_edge(self, event_key, yes_venue, no_venue, yes_price, no_price, edge, size, outcome):
        self.edges.append((event_key, edge, size, outcome))


def test_edge_observations_logged_for_fire_and_settling_skip():
    fe = FakeExec()
    st = RecordingStore()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=0.0, clock=lambda: 0.0, min_leg_price=0.02, store=st,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    # A genuine, mid-priced edge -> executes -> logged with the exec outcome.
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    # A settling-market phantom edge (poly YES @ 0.01) -> skipped -> logged as settling.
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.01, ya=100, no_ask=0.60, na=100)))
    outcomes = [e[3] for e in st.edges]
    assert "skip_settling" in outcomes
    assert any(o not in ("skip_settling", "edge_gone_after_depth") for o in outcomes)


def test_maker_mode_dispatches_off_the_quote_loop():
    from bot.execution.executor import ExecStatus, ExecutionReport

    class MakerExec:
        def __init__(self):
            self.calls = []

        async def execute_maker(self, opp):
            self.calls.append(opp)
            return ExecutionReport(ExecStatus.SUCCESS, "locked")

    fe = MakerExec()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, maker_mode=True)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])

    async def driver():
        await eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100))
        r = await eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60))
        assert r is None                          # dispatched in background, not awaited inline
        await asyncio.gather(*list(eng._inflight))

    asyncio.run(driver())
    assert len(fe.calls) == 1
    assert eng._maker_inflight == set()           # cleared after completion


def test_maker_mode_one_resting_maker_per_pair():
    from bot.execution.executor import ExecStatus, ExecutionReport

    release = asyncio.Event()

    class SlowMaker:
        def __init__(self):
            self.calls = 0

        async def execute_maker(self, opp):
            self.calls += 1
            await release.wait()
            return ExecutionReport(ExecStatus.SUCCESS, "x")

    fe = SlowMaker()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=0.0, clock=lambda: 0.0, maker_mode=True)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])

    async def driver():
        await eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100))
        await eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60))  # dispatch
        await asyncio.sleep(0)                     # let the maker task start (then it blocks)
        await eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60))  # in-flight -> skip
        assert fe.calls == 1                       # second did NOT post a duplicate maker
        release.set()
        await asyncio.gather(*list(eng._inflight))

    asyncio.run(driver())
    assert fe.calls == 1


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
