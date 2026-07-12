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


class _Risk:
    def __init__(self): self.is_killed = False
    def trip_kill_switch(self, reason): self.is_killed = True


class _ExecR(FakeExec):
    def __init__(self): super().__init__(); self.risk = _Risk()


def _snap(venue, positions):
    from types import SimpleNamespace
    return SimpleNamespace(
        venue=venue,
        positions=[SimpleNamespace(market_id=m, quantity=q, is_open=True) for m, q in positions])


def _open_quote(state):
    from types import SimpleNamespace
    return SimpleNamespace(state=state)


def test_reconcile_skips_actively_trading_pair_then_halts_when_quiesced():
    # A pair traded within the grace window shows a TRANSIENT burst imbalance (Poly fills
    # land instantly; Kalshi /positions lags) — it must NOT false-halt while active. Only
    # once trading quiesces past the grace does a persistent imbalance trip the kill switch.
    t = [1000.0]
    fe = _ExecR()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: t[0], reconcile_halt=True,
        depth_fetch=lambda v, m: _open_quote("MARKET_STATE_OPEN"))   # both legs OPEN
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    key = next(iter(eng._pairs))
    snaps = [_snap("kalshi", [("K1", 34)]), _snap("poly", [("P1", 46)])]  # Δ12 mid-burst

    eng._last_acted[key] = 1000.0                       # pair just traded
    asyncio.run(eng.reconcile_positions(snaps))
    asyncio.run(eng.reconcile_positions(snaps))         # two checks INSIDE the grace window
    assert not fe.risk.is_killed                        # no false halt while actively trading

    t[0] = 1000.0 + 30.0                                # trading quiesces (> 25s grace)
    asyncio.run(eng.reconcile_positions(snaps))         # first sighting after quiesce -> warn
    assert not fe.risk.is_killed
    asyncio.run(eng.reconcile_positions(snaps))         # still imbalanced -> persistent -> halt
    assert fe.risk.is_killed


def _settled_engine(open_states, depth_states):
    fe = _ExecR()
    async def oc(v, m): return open_states.get(v)          # True/False/None per venue
    async def fetch(v, m): return _open_quote(depth_states.get(v))
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 5000.0, reconcile_halt=True,
        depth_fetch=fetch, open_check=oc)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    return eng, fe


def test_reconcile_skips_poly_settled_leg_leftover():
    # Poly leg EXPIRED (settled -> 0) while Kalshi remains: realized arb, not stranded.
    eng, fe = _settled_engine(
        open_states={"kalshi": True, "poly": False},
        depth_states={"kalshi": "MARKET_STATE_OPEN", "poly": "MARKET_STATE_EXPIRED"})
    snaps = [_snap("kalshi", [("K1", 2)]), _snap("poly", [])]
    asyncio.run(eng.reconcile_positions(snaps))
    asyncio.run(eng.reconcile_positions(snaps))
    assert not fe.risk.is_killed


def test_reconcile_skips_kalshi_finalized_leg_leftover():
    # Kalshi market FINALIZED (settled -> 0) while Poly remains open: Kalshi's quote state is
    # None, so this relies on open_check (status field) reporting it settled -> no halt.
    eng, fe = _settled_engine(
        open_states={"kalshi": False, "poly": True},
        depth_states={"kalshi": None, "poly": "MARKET_STATE_OPEN"})
    snaps = [_snap("kalshi", []), _snap("poly", [("P1", 2)])]   # kalshi=0 vs poly=2
    asyncio.run(eng.reconcile_positions(snaps))
    asyncio.run(eng.reconcile_positions(snaps))
    assert not fe.risk.is_killed


def test_reconcile_halts_when_both_legs_open_real_naked():
    # Both markets OPEN but imbalanced -> a genuine stranded hedge -> still halts.
    eng, fe = _settled_engine(
        open_states={"kalshi": True, "poly": True},
        depth_states={"kalshi": "MARKET_STATE_OPEN", "poly": "MARKET_STATE_OPEN"})
    snaps = [_snap("kalshi", []), _snap("poly", [("P1", 2)])]
    asyncio.run(eng.reconcile_positions(snaps))
    asyncio.run(eng.reconcile_positions(snaps))
    assert fe.risk.is_killed


def test_reconcile_treats_empty_venue_read_as_down_not_naked():
    # Poly's API returns ZERO positions (an outage) while Kalshi holds hedges on 2+ pairs, so
    # every pair looks naked at once. That's a venue-DOWN/stale read, not simultaneous hedge
    # failures — it must NOT halt even when it persists across checks (the outage spans them).
    fe = _ExecR()
    async def depth(v, m): return _open_quote("MARKET_STATE_OPEN")   # markets OPEN (not settled)
    async def opencheck(v, m): return True
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 5000.0, reconcile_halt=True,
        depth_fetch=depth, open_check=opencheck)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1"),
                   ConfirmedPair("E2", "kalshi", "K2", "poly", "P2")])
    down = [_snap("kalshi", [("K1", 40), ("K2", 30)]), _snap("poly", [])]   # poly empty (down)
    asyncio.run(eng.reconcile_positions(down))
    asyncio.run(eng.reconcile_positions(down))          # persists across checks, but it's a down-read
    assert not fe.risk.is_killed                         # NOT halted (venue down, not naked)

    # a SINGLE pair naked on an empty venue is below the threshold -> still treated as real
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    one = [_snap("kalshi", [("K1", 40)]), _snap("poly", [])]
    asyncio.run(eng.reconcile_positions(one))
    asyncio.run(eng.reconcile_positions(one))
    assert fe.risk.is_killed                             # isolated naked still halts


def test_reconcile_skips_settled_and_removed_leg_404():
    # Poly settled AND was pruned -> /book 404 -> is_open None AND quote state None. The
    # 0-position leg whose market is now unreadable is a settled leftover (not a stranded
    # leg, whose market would still read OPEN). Must NOT halt.
    eng, fe = _settled_engine(
        open_states={"kalshi": True, "poly": None},     # poly 404 -> unknown
        depth_states={"kalshi": "MARKET_STATE_OPEN", "poly": None})
    snaps = [_snap("kalshi", [("K1", 66)]), _snap("poly", [])]  # kalshi=66 vs poly=0 (pruned)
    asyncio.run(eng.reconcile_positions(snaps))
    asyncio.run(eng.reconcile_positions(snaps))
    assert not fe.risk.is_killed


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


def test_run_streams_during_a_slow_refresh(monkeypatch):
    # THE duty-cycle fix: the discovery pass used to CANCEL the consumers for its whole
    # duration (~minutes), leaving the fast path dark. Consumers must now keep consuming
    # (and trading) WHILE a slow refresh is in progress.
    fe = FakeExec()
    eng = make_engine(fe)

    class SlowStreamVenue:
        def __init__(self, name, quotes):
            self.name = name
            self._quotes = quotes

        async def stream_order_book(self, mids):
            await asyncio.sleep(0.05)          # quotes arrive DURING the slow refresh below
            for x in self._quotes:
                yield x
            await asyncio.sleep(10)            # stay open

    venues = [
        SlowStreamVenue("kalshi", [q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)]),
        SlowStreamVenue("poly", [q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)]),
    ]
    refreshes = [0]

    async def refresh():
        refreshes[0] += 1
        if refreshes[0] == 1:
            return [ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")]   # boot: fast
        await asyncio.sleep(10)                # second refresh is SLOW (a discovery pass)
        return [ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")]

    async def driver():
        task = asyncio.create_task(eng.run(venues, refresh, refresh_interval=0.01))
        await asyncio.sleep(0.3)               # well into the slow second refresh
        assert len(fe.calls) >= 1              # traded WHILE the refresh was running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(driver())


def test_run_resubscribes_only_when_market_set_changes():
    # Unchanged watchlist -> consumers keep their WS sessions (no churn); a changed
    # market set -> resubscribe.
    fe = FakeExec()
    eng = make_engine(fe)
    subscribes = []

    class CountingVenue:
        def __init__(self, name):
            self.name = name

        async def stream_order_book(self, mids):
            subscribes.append((self.name, tuple(sorted(mids))))
            await asyncio.sleep(10)            # stay open, never yields
            yield None                          # pragma: no cover (makes it a generator)

    venues = [CountingVenue("kalshi"), CountingVenue("poly")]
    sets = [
        [ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")],
        [ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")],   # unchanged -> no resubscribe
        [ConfirmedPair("E2", "kalshi", "K2", "poly", "P2")],   # changed -> resubscribe
    ]
    idx = [0]

    async def refresh():
        i = min(idx[0], len(sets) - 1)
        idx[0] += 1
        return sets[i]

    async def driver():
        task = asyncio.create_task(eng.run(venues, refresh, refresh_interval=0.02))
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(driver())
    per_venue = [s for s in subscribes if s[0] == "kalshi"]
    assert len(per_venue) == 2                                  # boot + the ONE change
    assert per_venue[0][1] == ("K1",) and per_venue[1][1] == ("K2",)


def test_hybrid_take_does_not_block_the_quote_loop():
    # A slow execution must not stall on_quote: the take is spawned, on_quote returns
    # immediately, and a second pair's edge on the same venue stream is still evaluated
    # while the first take is mid-flight.
    class SlowExec(FakeExec):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def execute(self, opp):
            self.calls.append(opp)
            self.started.set()
            await self.release.wait()          # simulate a slow two-leg execution
            return f"executed {opp.event_key}"

    fe = SlowExec()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, maker_mode=True,
        hybrid_take_depth=10)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1"),
                   ConfirmedPair("E2", "kalshi", "K2", "poly", "P2")])

    async def driver():
        await eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100))
        await eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60))
        await fe.started.wait()                # first take is now mid-flight (blocked)
        # the quote loop must still process OTHER pairs while it runs:
        await eng.on_quote(q("kalshi", "K2", yes_ask=0.40, ya=100, no_ask=0.65, na=100))
        await eng.on_quote(q("poly", "P2", yes_ask=0.62, ya=100, no_ask=0.55, na=60))
        fe.release.set()
        await eng.drain()

    asyncio.run(driver())
    assert len(fe.calls) == 2                  # both pairs fired despite the slow first take


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


def test_log_edge_snapshot_ranks_by_confirmed_depth_excludes_one_sided(caplog):
    import logging
    from bot.streaming.engine import StreamingEngine
    # Two pairs both quote a "good" WS edge, but E_empty's real book is one-sided (no NO
    # offer on the Kalshi leg) while E_real has true two-sided depth. The snapshot must
    # rank by the CONFIRMED book and drop the empty/sentinel pair entirely.
    async def depth_fetch(venue, mid):
        if mid in ("K_empty", "P_empty"):
            # One-sided: only a YES ask exists, no NO offer -> not a tradeable arb.
            return q(venue, mid, yes_ask=0.01, ya=100, no_ask=None, na=0.0)
        if venue == "kalshi":
            return q("kalshi", "K_real", yes_ask=0.40, ya=500, no_ask=0.62, na=500)
        return q("poly", "P_real", yes_ask=0.61, ya=800, no_ask=0.55, na=800)

    eng = StreamingEngine(
        executor=FakeExec(),
        fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, depth_fetch=depth_fetch,
    )
    eng.set_pairs([
        ConfirmedPair("E_empty", "kalshi", "K_empty", "poly", "P_empty"),
        ConfirmedPair("E_real", "kalshi", "K_real", "poly", "P_real"),
    ])
    # Both pairs get a two-sided WS quote so both are CANDIDATES; depth confirm decides.
    asyncio.run(eng.on_quote(q("kalshi", "K_empty", yes_ask=0.01, ya=100, no_ask=0.50, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P_empty", yes_ask=0.40, ya=100, no_ask=0.50, na=100)))
    asyncio.run(eng.on_quote(q("kalshi", "K_real", yes_ask=0.40, ya=100, no_ask=0.62, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P_real", yes_ask=0.61, ya=100, no_ask=0.55, na=100)))

    with caplog.at_level(logging.INFO, logger="bot.streaming"):
        asyncio.run(eng.log_edge_snapshot())
    text = caplog.text
    # The confirm RESEEDS the live book with REST truth: the phantom pair's one-sided
    # real book replaces its lying WS quote, so the denominator honestly reads 1/2
    # (pre-reseed it read 2/2 and the stale book kept re-triggering confirms).
    assert "1/2 pairs two-sided on WS; depth-sampled top 1: 1 verified two-sided" in text
    assert "sz=500" in text          # min(K_real 500, P_real 800) from the confirmed book
    assert "sz=0" not in text        # the empty/sentinel pair is gone, not shown at size 0


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


def test_maker_mode_deep_fresh_ws_fast_takes_without_confirm():
    # LATENCY: in MAKER mode, a fresh + sized WS book deep enough to TAKE (size >=
    # hybrid_take_depth) fires the take WITHOUT the REST depth-confirm round-trip — beating
    # slower actors to the edge instead of losing it in the ~35ms confirm window. (Thin
    # books and maker rests still confirm; covered by the test above with depth 0.)
    import time as _time

    fe = FakeExec()
    fetched = []

    async def depth_fetch(venue, mid):
        fetched.append((venue, mid))                 # must NOT be called on the fast take
        return q(venue, mid, yes_ask=0.99, ya=1, no_ask=0.99, na=1)

    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, depth_fetch=depth_fetch,
        max_ws_quote_age=2.0, maker_mode=True, hybrid_take_depth=10,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    now = _time.time()
    ka = q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100); ka.timestamp = now
    pa = q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60); pa.timestamp = now
    asyncio.run(eng.on_quote(ka))
    asyncio.run(eng.on_quote(pa))
    assert len(fe.calls) == 1 and fetched == []      # took off the WS book, no REST confirm
    # a SHALLOW fresh book (below hybrid_take_depth) must still confirm, not fast-take
    fe2 = FakeExec(); fetched.clear()
    eng2 = StreamingEngine(
        executor=fe2, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, depth_fetch=depth_fetch,
        max_ws_quote_age=2.0, maker_mode=True, hybrid_take_depth=10,
    )
    eng2.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    ka2 = q("kalshi", "K1", yes_ask=0.40, ya=5, no_ask=0.65, na=5); ka2.timestamp = now
    pa2 = q("poly", "P1", yes_ask=0.62, ya=5, no_ask=0.55, na=5); pa2.timestamp = now
    asyncio.run(eng2.on_quote(ka2))
    asyncio.run(eng2.on_quote(pa2))
    assert fetched != []                              # size 5 < 10 -> confirmed, not fast-taken


def test_sync_window_fires_sub_100ms_bypassing_persist():
    # With a sync window set, a deep edge whose BOTH legs ticked within the window fires
    # IMMEDIATELY (first sighting) — bypassing the 0.75s persist wait — and with NO REST
    # confirm. A non-synced edge (legs ticked far apart) falls back to the persist guard.
    import time as _time

    def make(persist, sync):
        fe = FakeExec(); fetched = []

        async def depth_fetch(venue, mid):
            fetched.append((venue, mid))
            return q(venue, mid, yes_ask=0.99, ya=1, no_ask=0.99, na=1)

        eng = StreamingEngine(
            executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
            min_edge=0.01, cooldown=100.0, clock=lambda: 1000.0, depth_fetch=depth_fetch,
            max_ws_quote_age=2.0, maker_mode=True, hybrid_take_depth=10,
            edge_persist_secs=persist, sync_window_secs=sync,
        )
        eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
        return eng, fe, fetched

    now = _time.time()
    # SYNCED: both legs stamped ~now -> within the 50ms window of each other and of now.
    eng, fe, fetched = make(persist=0.75, sync=0.05)
    ka = q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100); ka.timestamp = now
    pa = q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60); pa.timestamp = now
    asyncio.run(eng.on_quote(ka))
    asyncio.run(eng.on_quote(pa))
    assert len(fe.calls) == 1 and fetched == []      # fired first sighting, no persist, no REST

    # NOT SYNCED: one leg ticked 200ms ago -> outside the window -> persist guard applies ->
    # first sighting just waits (nothing fired).
    eng2, fe2, _ = make(persist=0.75, sync=0.05)
    ka2 = q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100); ka2.timestamp = now - 0.2
    pa2 = q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60); pa2.timestamp = now
    asyncio.run(eng2.on_quote(ka2))
    asyncio.run(eng2.on_quote(pa2))
    assert fe2.calls == []                            # not synced -> persist held it back


def test_implausible_edge_skipped_as_false_match():
    # An edge too large to be a real arb (the R6/CoD signature: YES+NO ≈ 0.59 -> ~41% "edge")
    # is the fingerprint of a FALSE same-event match -> never trade it.
    fe = FakeExec()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, max_plausible_edge=0.06, cooldown=100.0, clock=lambda: 0.0)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.50, ya=100, no_ask=0.50, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.50, ya=100, no_ask=0.09, na=100)))
    assert fe.calls == []                            # 0.50+0.09=0.59 -> edge ~0.41 > 0.06 -> skip


def test_fat_edge_fires_on_proven_complement():
    # NO hard edge ceiling: a pair whose observed YES+NO history proves complementarity
    # (>= ~30 samples, mean >= 0.97) fires on ANY edge — a fat edge on a proven pair is
    # a genuine dislocation, not a false match.
    fe = FakeExec()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, max_plausible_edge=0.06, empirical_min_obs=4,
        empirical_sum_floor=0.93, cooldown=0.0, clock=lambda: 0.0)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])

    async def driver():
        await eng.on_quote(q("poly", "P1", yes_ask=0.52, ya=100, no_ask=0.51, na=100))
        for _ in range(31):                       # build a ~$1.01-sum history (no edge)
            await eng.on_quote(q("kalshi", "K1", yes_ask=0.50, ya=100, no_ask=0.50, na=100))
        assert fe.calls == []                     # nothing fired while sum ~1.01
        # genuine dislocation: poly NO collapses -> sum 0.90 -> edge 0.10 > the 0.06 bar
        await eng.on_quote(q("poly", "P1", yes_ask=0.52, ya=100, no_ask=0.40, na=100))

    asyncio.run(driver())
    assert len(fe.calls) == 1                     # proven complement -> fat edge FIRES


def test_fat_edge_unproven_observes_then_fires_once_proven():
    # An unproven pair showing a fat edge must NOT fire yet — but it also must NOT be
    # permanently parked (the old hard-cap behavior): once its history proves the
    # complement, the same fat edge fires.
    fe = FakeExec()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, max_plausible_edge=0.06, empirical_min_obs=4,
        empirical_sum_floor=0.93, cooldown=0.0, clock=lambda: 0.0)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])

    async def driver():
        await eng.on_quote(q("poly", "P1", yes_ask=0.52, ya=100, no_ask=0.40, na=100))
        await eng.on_quote(q("kalshi", "K1", yes_ask=0.50, ya=100, no_ask=0.50, na=100))
        assert fe.calls == []                     # fat edge, 2 samples -> observe only
        for _ in range(35):                       # now prove the complement (sum ~1.02)
            await eng.on_quote(q("poly", "P1", yes_ask=0.52, ya=100, no_ask=0.52, na=100))
        await eng.on_quote(q("poly", "P1", yes_ask=0.52, ya=100, no_ask=0.40, na=100))

    asyncio.run(driver())
    assert len(fe.calls) == 1                     # not parked: fires once proven


def test_fat_edge_persistent_false_match_still_blacklists():
    # A pair whose sum history sits far from $1 (a spread-vs-moneyline false match) never
    # fires AND still gets evidence-blacklisted, exactly as before.
    from bot.data.store import Store
    store = Store(":memory:")
    fe = FakeExec()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, max_plausible_edge=0.06, empirical_min_obs=4,
        empirical_sum_floor=0.93, cooldown=0.0, clock=lambda: 0.0, store=store)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])

    async def driver():
        await eng.on_quote(q("poly", "P1", yes_ask=0.50, ya=100, no_ask=0.09, na=100))
        for _ in range(10):                       # sum ~0.59 every tick -> false match
            await eng.on_quote(q("kalshi", "K1", yes_ask=0.50, ya=100, no_ask=0.50, na=100))

    asyncio.run(driver())
    assert fe.calls == []                         # never traded
    assert store.blacklisted_keys()               # and evidence-blacklisted


def test_empirical_gate_blocks_false_match_and_confirms_real_pair():
    # The empirical same-event gate: a pair only trades once its observed YES+NO sum confirms
    # the legs are complements (mean >= floor over >= min_obs samples).
    def run(no_ask_poly):
        fe = FakeExec(); t = [0.0]
        eng = StreamingEngine(
            executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
            min_edge=0.01, empirical_min_obs=4, empirical_sum_floor=0.93,
            cooldown=0.0, clock=lambda: t[0])
        eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
        for i in range(12):
            t[0] = float(i)
            asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)))
            asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=no_ask_poly, na=100)))
        return fe

    # FALSE match: YES 0.40 + NO 0.50 = 0.90 < floor 0.93 -> never trades, however many ticks.
    assert run(0.50).calls == []
    # REAL pair: YES 0.40 + NO 0.55 = 0.95 >= floor -> confirms after observing, then trades.
    assert len(run(0.55).calls) >= 1


def test_maker_volume_gate_blocks_maker_keeps_take():
    # A thin (maker-ineligible) Polymarket market: a shallow edge does NOT rest a maker (its
    # hedge would 500 and go naked), but a DEEP edge still TAKES it — a 500 on a taker leg is
    # a clean skip via the leg-order fix. So thin markets stay tradeable, just never rested.
    def fire(depth_size):
        fe = FakeExec()
        eng = StreamingEngine(
            executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
            min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, maker_mode=True,
            hybrid_take_depth=10, maker_eligible=lambda v, m: False)   # poly thin -> never maker
        eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
        asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=depth_size, no_ask=0.65, na=depth_size)))
        asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=depth_size, no_ask=0.55, na=depth_size)))
        return fe

    assert fire(5).calls == []          # shallow (5 < take_depth 10) -> maker path -> blocked
    assert len(fire(50).calls) == 1     # deep (50 >= 10) -> hybrid TAKE fires despite thin market


def test_prime_and_sweep_fetches_concurrently():
    # The watchlist prime must fetch legs in parallel (peak in-flight > 1), not one-by-one,
    # so a ~140-leg refresh takes seconds rather than a minute.
    inflight = [0]
    peak = [0]

    async def depth_fetch(venue, mid):
        inflight[0] += 1
        peak[0] = max(peak[0], inflight[0])
        await asyncio.sleep(0.02)
        inflight[0] -= 1
        return q(venue, mid, yes_ask=0.40, ya=50, no_ask=0.65, na=50)

    eng = StreamingEngine(
        executor=FakeExec(), fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, clock=lambda: 0.0, depth_fetch=depth_fetch, prime_concurrency=8,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    asyncio.run(eng.prime_and_sweep())
    assert peak[0] == 2          # both legs in flight at once (sequential would peak at 1)


def test_persistence_filter_waits_for_edge_to_hold():
    # With edge_persist_secs set, a freshly-appeared edge must hold continuously for the
    # window before the engine acts — separating a real venue-lag from a flicker artifact.
    fe = FakeExec()
    t = [0.0]
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=0.0, clock=lambda: t[0], edge_persist_secs=1.0,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    eng.livebook.update(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100))
    tick = lambda: asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    tick()                       # t=0: first sighting -> wait
    assert fe.calls == []
    t[0] = 0.5; tick()           # still within the window -> wait
    assert fe.calls == []
    t[0] = 1.5; tick()           # held > 1s -> act
    assert len(fe.calls) == 1


def test_persistence_filter_resets_when_edge_vanishes():
    # If the edge drops below threshold the persistence timer resets, so a later flicker
    # has to hold the full window again (it can't accumulate across gaps).
    fe = FakeExec()
    t = [0.0]
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=0.0, clock=lambda: t[0], edge_persist_secs=1.0,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    eng.livebook.update(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))  # edge, t=0
    t[0] = 0.5  # both directions now > 1 (0.40+0.65 and 0.50+0.65) -> edge gone -> reset
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.50, ya=100, no_ask=0.65, na=60)))
    t[0] = 1.2                   # > 1s since first sighting, but the timer was reset
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    assert fe.calls == []        # re-armed at t=1.2; not held a full second yet


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


def test_unfilled_maker_does_not_back_off():
    # A maker that RESTED and expired uncrossed (leg status RESTING) is a benign no-trade:
    # it must NOT back the pair off (so it can re-rest while the edge persists). A genuine
    # venue REJECTION still does.
    from bot.execution.executor import ExecStatus, ExecutionReport
    from bot.execution.orders import OrderResult, OrderStatus

    eng = make_engine(FakeExec())
    p = next(iter(eng._pairs.values()))
    resting = OrderResult("kalshi", "K1", None, "buy", 0, status=OrderStatus.RESTING)
    eng._note_outcome(p.key, p, ExecutionReport(ExecStatus.SKIPPED, "maker unfilled — expired", [resting]))
    assert p.key not in eng._backoff_until            # benign -> no backoff, re-rests

    rejected = OrderResult("kalshi", "K1", None, "buy", 0, status=OrderStatus.REJECTED)
    eng._note_outcome(p.key, p, ExecutionReport(ExecStatus.SKIPPED, "leg REJECTED", [rejected]))
    assert p.key in eng._backoff_until                # hostile -> backs off


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


class HybridExec:
    """Records taker (execute) and maker (execute_maker) calls separately so a test
    can assert which path the hybrid router chose."""

    def __init__(self):
        from bot.execution.executor import ExecStatus, ExecutionReport
        self.taker = []
        self.maker = []
        self._report = ExecutionReport(ExecStatus.SUCCESS, "locked")

    async def execute(self, opp):
        self.taker.append(opp)
        return self._report

    async def execute_maker(self, opp):
        self.maker.append(opp)
        return self._report


def _hybrid_engine(fe, *, take_depth, take_bar):
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, maker_mode=True,
        hybrid_take_depth=take_depth, hybrid_take_bar=take_bar)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    return eng


def test_hybrid_takes_deep_edge_that_clears_taker_bar():
    # edge 0.05 on 60 contracts: clears the 0.04 taker bar AND has >= 50 depth -> TAKE now.
    fe = HybridExec()
    eng = _hybrid_engine(fe, take_depth=50, take_bar=0.04)

    async def driver():
        await eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100))
        # The take is SPAWNED off the quote loop (a blocking inline take stalled the
        # venue's whole WS stream during execution), so on_quote returns None and the
        # execution is settled via drain().
        await eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60))
        await eng.drain()

    asyncio.run(driver())
    assert len(fe.taker) == 1 and fe.maker == []    # took it, did not rest a maker
    assert fe.taker[0].max_contracts == 60


def test_hybrid_rests_maker_when_depth_too_thin():
    # Same 0.05 edge but only 20 contracts of size (< take_depth 50) -> rest a maker.
    fe = HybridExec()
    eng = _hybrid_engine(fe, take_depth=50, take_bar=0.04)

    async def driver():
        await eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100))
        await eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=20))
        await asyncio.gather(*list(eng._inflight))

    asyncio.run(driver())
    assert fe.maker and fe.taker == []              # rested a maker, did not take


def test_hybrid_rests_maker_when_edge_below_taker_bar():
    # Deep (60) but a thin 0.05 edge under a high 0.06 taker bar -> rest a maker, don't take.
    fe = HybridExec()
    eng = _hybrid_engine(fe, take_depth=50, take_bar=0.06)

    async def driver():
        await eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100))
        await eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60))
        await asyncio.gather(*list(eng._inflight))

    asyncio.run(driver())
    assert fe.maker and fe.taker == []


def test_hybrid_disabled_always_rests_maker():
    # take_depth 0 = pure maker mode: never auto-takes even a deep, bar-clearing edge.
    fe = HybridExec()
    eng = _hybrid_engine(fe, take_depth=0, take_bar=0.04)

    async def driver():
        await eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100))
        await eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60))
        await asyncio.gather(*list(eng._inflight))

    asyncio.run(driver())
    assert fe.maker and fe.taker == []


def test_kill_switch_skips_all_pair_work():
    # When the executor's kill switch is tripped, a quote must NOT trigger any edge work
    # (no execute, no depth-confirm REST) — the pair is dead until a restart clears it.
    class KilledExec:
        class risk:
            is_killed = True
        def __init__(self):
            self.calls = []
        async def execute(self, opp):
            self.calls.append(opp)
            return "executed"

    fe = KilledExec()
    depth_calls = []

    async def depth_fetch(venue, mid):
        depth_calls.append((venue, mid))
        return q(venue, mid, yes_ask=0.40, ya=100, no_ask=0.55, na=100)

    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, depth_fetch=depth_fetch)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])

    asyncio.run(eng.on_quote(q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)))
    asyncio.run(eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)))
    assert fe.calls == [] and depth_calls == []     # no execute, no REST depth-confirm


def _snap(venue, positions):
    from bot.execution.account import AccountSnapshot, VenuePosition
    return AccountSnapshot(venue, 1000.0, [VenuePosition(m, q, 0) for m, q in positions])


def test_reconcile_flags_and_halts_on_persistent_naked():
    class RiskStub:
        def __init__(self):
            self.is_killed = False
            self.reason = None
        def trip_kill_switch(self, reason):
            self.is_killed = True
            self.reason = reason

    class Exc:
        def __init__(self):
            self.risk = RiskStub()

    exc = Exc()
    eng = StreamingEngine(
        executor=exc, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, clock=lambda: 0.0, reconcile_halt=True)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])

    # Kalshi holds 140, Polymarket holds nothing -> naked. First check warns, no halt.
    # No depth_fetch -> can't verify a settled leg -> fails toward halt (real naked).
    snaps = [_snap("kalshi", [("K1", 140)]), _snap("poly", [])]
    out = asyncio.run(eng.reconcile_positions(snaps))
    assert len(out) == 1 and not exc.risk.is_killed     # warned, not yet halted
    # Still naked on the next check -> trip the kill switch.
    asyncio.run(eng.reconcile_positions(snaps))
    assert exc.risk.is_killed


def test_reconcile_balanced_pair_is_clean():
    class Exc:
        risk = None
    eng = StreamingEngine(
        executor=Exc(), fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, clock=lambda: 0.0)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    # Equal contracts on both legs = a locked arb, not naked.
    snaps = [_snap("kalshi", [("K1", 140)]), _snap("poly", [("P1", 140)])]
    assert asyncio.run(eng.reconcile_positions(snaps)) == []


def test_preview_no_fill_backs_off_pair():
    # A "hedge unfillable ..." skip should back the pair off briefly so it stops
    # re-confirming an unfillable edge every cooldown (no legs were placed -> not a failure).
    from bot.execution.executor import ExecStatus, ExecutionReport
    eng = make_engine(FakeExec(), now=100.0)
    p = next(iter(eng._pairs.values()))
    eng._note_outcome(p.key, p, ExecutionReport(
        ExecStatus.SKIPPED, "hedge unfillable: book depth <1 contract at limit"))
    assert eng._backoff_until.get(p.key, 0) > 100.0       # backed off
    # A benign maker-expired skip (no 'hedge unfillable' reason) must NOT back off.
    eng2 = make_engine(FakeExec(), now=100.0)
    p2 = next(iter(eng2._pairs.values()))
    eng2._note_outcome(p2.key, p2, ExecutionReport(
        ExecStatus.SKIPPED, "maker unfilled — expired/cancelled, no trade"))
    assert p2.key not in eng2._backoff_until


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


def test_fill_tracker_is_bounded():
    # One _OrderState per order forever = a slow leak; the map must stay bounded.
    from bot.streaming.fills import FillEvent, FillTracker

    async def main():
        t = FillTracker()
        t._max = 50
        for i in range(200):
            await t.apply(FillEvent("kalshi", f"o{i}", "FILL", 1.0, 0.5))
        assert len(t._orders) <= 50
        # the most recent order's state survived
        status, filled, _ = await t.confirm("kalshi", "o199", 1.0, timeout=0.01)
        assert filled == 1.0

    asyncio.run(main())


def _acted_store(pairs):
    """Store with acted opportunities for (kalshi_mkt, poly_mkt) pairs."""
    from types import SimpleNamespace

    from bot.data.store import Store
    store = Store(":memory:")
    for km, pm in pairs:
        store.record_opportunity(SimpleNamespace(
            event_key=f"{km}|{pm}", buy_yes_venue="kalshi", buy_yes_market=km,
            buy_no_venue="poly", buy_no_market=pm, yes_price=0.4, no_price=0.55,
            edge_per_contract=0.05, max_contracts=10, total_profit=0.5), acted=True)
    return store


def _hist_engine(store):
    fe = _ExecR()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 5000.0, reconcile_halt=True,
        store=store)
    eng.set_pairs([ConfirmedPair("LIVE", "kalshi", "KL", "poly", "PL")])  # unrelated live pair
    return eng, fe


def test_reconcile_halts_unpaired_naked_via_history():
    # THE TPZRL regression: a pair leaves the watchlist but its Kalshi leg is still held
    # and its counterpart is FLAT — a genuinely naked leg that used to be only an INFO
    # line. The store's acted history identifies the counterpart; persistent -> halt.
    store = _acted_store([("K_old", "p-old")])
    eng, fe = _hist_engine(store)
    snaps = [_snap("kalshi", [("K_old", 3)]), _snap("poly", [("P_other", 1)])]
    asyncio.run(eng.reconcile_positions(snaps))
    assert not fe.risk.is_killed                       # first sighting: warn only
    asyncio.run(eng.reconcile_positions(snaps))
    assert fe.risk.is_killed                           # persisted -> halt


def test_reconcile_unpaired_but_hedged_on_counterpart_is_quiet():
    # Both legs of the forgotten pair still hold matching size -> hedged, no alarm.
    store = _acted_store([("K_old", "p-old")])
    eng, fe = _hist_engine(store)
    snaps = [_snap("kalshi", [("K_old", 3)]), _snap("poly", [("p-old", 3)])]
    asyncio.run(eng.reconcile_positions(snaps))
    asyncio.run(eng.reconcile_positions(snaps))
    assert not fe.risk.is_killed


def test_reconcile_unpaired_blacklisted_pair_stays_quarantined():
    # A quarantined (blacklisted) pair's stranded leg was deliberately recorded and left —
    # the whole point of quarantine is NOT freezing the bot on it. Must not halt.
    store = _acted_store([("K_q", "p-q")])
    store.blacklist_pair("kalshi", "K_q", "poly", "p-q", reason="quarantine test")
    eng, fe = _hist_engine(store)
    snaps = [_snap("kalshi", [("K_q", 5)]), _snap("poly", [])]
    asyncio.run(eng.reconcile_positions(snaps))
    asyncio.run(eng.reconcile_positions(snaps))
    assert not fe.risk.is_killed


def test_reconcile_unpaired_without_history_stays_info():
    # No acted history for the market at all -> can't verify -> surface for a human,
    # never halt on a guess.
    store = _acted_store([])
    eng, fe = _hist_engine(store)
    snaps = [_snap("kalshi", [("K_manual", 4)]), _snap("poly", [])]
    asyncio.run(eng.reconcile_positions(snaps))
    asyncio.run(eng.reconcile_positions(snaps))
    assert not fe.risk.is_killed


def test_verified_pair_fires_fat_edge_with_zero_history():
    # A rules/settlement-verified pair needs NO price history: first-ever tick with a
    # 41% edge fires (definitional truth outranks price statistics). An unverified pair
    # in the identical situation observes instead.
    def make(verified):
        fe = FakeExec()
        eng = StreamingEngine(
            executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
            min_edge=0.01, max_plausible_edge=0.06, empirical_min_obs=4,
            empirical_sum_floor=0.93, cooldown=0.0, clock=lambda: 0.0)
        pair = ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")
        eng.set_pairs([pair])
        if verified:
            eng.verified_pairs = {pair.key}
        return eng, fe

    async def driver(eng):
        await eng.on_quote(q("kalshi", "K1", yes_ask=0.50, ya=100, no_ask=0.50, na=100))
        await eng.on_quote(q("poly", "P1", yes_ask=0.50, ya=100, no_ask=0.09, na=100))

    eng, fe = make(verified=True)
    asyncio.run(driver(eng))
    assert len(fe.calls) == 1                        # fired on the first sighting

    eng, fe = make(verified=False)
    asyncio.run(driver(eng))
    assert fe.calls == []                            # unverified -> observes first


def test_fat_edge_rejects_wide_book_high_sum_history():
    # A mean ASK-sum far ABOVE $1 = chronically wide/illiquid books (live incident: a
    # pair averaging 1.281 "fat-fired" when one book collapsed — a quote pull, not a
    # dislocation). The proven-complement band must reject it, floor AND ceiling.
    fe = FakeExec()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, max_plausible_edge=0.06, empirical_min_obs=4,
        empirical_sum_floor=0.93, cooldown=0.0, clock=lambda: 0.0)
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])

    async def driver():
        await eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.78, na=100))
        for _ in range(35):                        # wide books: sum history ~1.28
            await eng.on_quote(q("kalshi", "K1", yes_ask=0.50, ya=100, no_ask=0.50, na=100))
        # one book collapses -> apparent 26% "edge" on a 1.28-mean pair -> must NOT fire
        await eng.on_quote(q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.24, na=100))

    asyncio.run(driver())
    assert fe.calls == []


def test_reconcile_ignores_recycle_remnant():
    # A capital-recycled pair holds only the cheap OTM leg (upset-hedge remnant) with
    # both markets still OPEN — the reconcile must read it from the remnant registry,
    # not as naked exposure. Larger-than-remnant imbalance still halts.
    eng, fe = _settled_engine(
        open_states={"kalshi": True, "poly": True},
        depth_states={"kalshi": "MARKET_STATE_OPEN", "poly": "MARKET_STATE_OPEN"})
    fe.recycled_remnants = {("poly", "P1"): 2.0}
    snaps = [_snap("kalshi", []), _snap("poly", [("P1", 2)])]
    asyncio.run(eng.reconcile_positions(snaps))
    asyncio.run(eng.reconcile_positions(snaps))
    assert not fe.risk.is_killed
    # imbalance beyond the registered remnant -> still a genuine naked -> halts
    fe.recycled_remnants = {("poly", "P1"): 2.0}
    snaps = [_snap("kalshi", []), _snap("poly", [("P1", 8)])]
    asyncio.run(eng.reconcile_positions(snaps))
    asyncio.run(eng.reconcile_positions(snaps))
    assert fe.risk.is_killed


def test_stream_build_opp_carries_settle_ts_from_id():
    # The CA-gov leak: _build_opp omitted settle_ts -> 0 -> horizon gate silently OFF
    # on the fast path (where all live fires happen). The id-parsed date must flow in.
    from datetime import datetime, timezone
    from bot.models import MarketQuote
    eng = make_engine(FakeExec())
    p = ConfirmedPair("E1", "kalshi", "KXGOVCA-26-SHIL",
                      "poly", "ewc-usgub-ca-2026-11-03-stehil")
    yq = MarketQuote(venue="kalshi", market_id="KXGOVCA-26-SHIL", title="",
                     yes_ask=0.08, yes_ask_size=100, no_ask=0.93, no_ask_size=100)
    nq = MarketQuote(venue="poly", market_id="ewc-usgub-ca-2026-11-03-stehil", title="",
                     yes_ask=0.11, yes_ask_size=100, no_ask=0.90, no_ask_size=100)
    opp = eng._build_opp(p, 0.01, yq, nq, 10)
    assert opp.settle_ts > 0
    got = datetime.fromtimestamp(opp.settle_ts, timezone.utc).strftime("%Y-%m-%d")
    assert got == "2026-11-03"


def test_eval_direction_depth_sweep_reprices_quotes():
    from bot.fees import ZeroFeeModel
    from bot.streaming.engine import StreamingEngine
    eng = StreamingEngine.__new__(StreamingEngine)
    eng.min_edge = 0.0
    eng._fee = lambda v: ZeroFeeModel()
    a = MarketQuote(venue="kalshi", market_id="K", title="",
                    yes_ask=0.44, yes_ask_size=5, no_ask=0.60, no_ask_size=5,
                    yes_ask_levels=((0.44, 5), (0.46, 200)),
                    no_ask_levels=((0.60, 5),))
    b = MarketQuote(venue="poly", market_id="P", title="",
                    yes_ask=0.60, yes_ask_size=5, no_ask=0.50, no_ask_size=300,
                    yes_ask_levels=((0.60, 5),),
                    no_ask_levels=((0.50, 300),))
    edge, yq, nq, size = eng._eval_direction(a, b)
    assert yq.yes_ask == 0.46 and size == 205      # swept to level 2, cumulative size
    assert abs(edge - 0.04) < 1e-9
    # original book quotes untouched (copies were re-priced, not the shared book)
    assert a.yes_ask == 0.44 and a.yes_ask_size == 5


def test_confirm_reject_escalates_backoff_and_reseeds():
    import asyncio
    from bot.models import MarketQuote
    from bot.streaming.engine import StreamingEngine, ConfirmedPair, LiveBook
    from bot.fees import ZeroFeeModel

    eng = StreamingEngine.__new__(StreamingEngine)
    eng.min_edge = 0.005
    eng._fee = lambda v: ZeroFeeModel()
    eng.livebook = LiveBook()
    eng._backoff_base = 30.0
    eng._backoff_cap = 1800.0
    eng._confirm_fails = {}
    eng._backoff_until = {}
    eng.clock = lambda: 1000.0
    p = ConfirmedPair("ek", "kalshi", "K1", "poly", "P1")
    # REST truth: NO edge (sum 1.02) — the WS book was lying
    rest_k = MarketQuote(venue="kalshi", market_id="K1", title="",
                         yes_ask=0.52, yes_ask_size=50, no_ask=0.50, no_ask_size=50)
    rest_p = MarketQuote(venue="poly", market_id="P1", title="",
                         yes_ask=0.52, yes_ask_size=50, no_ask=0.50, no_ask_size=50)
    async def fake_fetch(venue, market):
        return rest_k if venue == "kalshi" else rest_p
    eng.depth_fetch = fake_fetch
    ev = asyncio.run(eng._confirm_depth(p, quiet=True))
    # reseed: livebook now carries the REST truth
    assert eng.livebook.get("kalshi", "K1").yes_ask == 0.52
    assert eng.livebook.get("poly", "P1").no_ask == 0.50
    # and REST-seeded quotes are timestampless -> can never enable the fast path
    assert (eng.livebook.get("kalshi", "K1").timestamp or 0) == 0


def test_ws_trust_ladder_earns_skip_and_resets_on_lie():
    # Three straight confirms where REST agrees with the WS claim earn the pair a
    # trusted fast-fire (no REST round-trip); a lying book cannot climb the ladder.
    import time as _time

    class MakerFake(FakeExec):
        async def execute_maker(self, opp):
            self.calls.append(opp)
            return f"executed {opp.event_key}"

    fe = MakerFake()
    fetched = []
    honest = {"v": True}
    t = [0.0]

    async def depth_fetch(venue, mid):
        fetched.append((venue, mid))
        if honest["v"]:   # same books the WS shows -> agreement
            return (q(venue, mid, yes_ask=0.40, ya=100, no_ask=0.65, na=100)
                    if venue == "kalshi" else
                    q(venue, mid, yes_ask=0.62, ya=100, no_ask=0.55, na=60))
        return q(venue, mid, yes_ask=0.99, ya=1, no_ask=0.99, na=1)  # liar: no edge

    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: t[0], depth_fetch=depth_fetch,
        max_ws_quote_age=5.0, maker_mode=True, ws_trust_min=3,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])

    def tick():
        t[0] += 200.0                                 # clear the cooldown each tick
        ka = q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100)
        ka.timestamp = _time.time()
        pa = q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60)
        pa.timestamp = _time.time()
        asyncio.run(eng.on_quote(ka))
        asyncio.run(eng.on_quote(pa))

    for _ in range(3):
        tick()
    assert max(eng._ws_trust.values()) >= 3           # ladder climbed on honest confirms
    n_confirmed = len(fetched)
    assert n_confirmed > 0
    tick()                                            # trusted -> fires with NO REST call
    assert len(fetched) == n_confirmed
    assert len(fe.calls) == 4
    # a lying book can never climb: reset and confirm against the liar
    honest["v"] = False
    eng._ws_trust = {k: 0 for k in eng._ws_trust}
    tick()
    assert max(eng._ws_trust.values()) == 0


def test_ws_synced_uses_exchange_time_over_arrival():
    import time as _time
    from bot.streaming.engine import StreamingEngine
    eng = StreamingEngine.__new__(StreamingEngine)
    eng.sync_window_secs = 0.1
    now = _time.time()
    # arrival times 80ms apart (inside window) but EXCHANGE times 5s apart:
    # one venue repriced, the other book is old news delivered late -> NOT synced
    a = q("kalshi", "K1", yes_ask=0.40, ya=50, no_ask=0.62, na=50)
    b = q("poly", "P1", yes_ask=0.62, ya=50, no_ask=0.55, na=50)
    a.timestamp = now; b.timestamp = now - 0.08
    a.exchange_ts = now - 0.01; b.exchange_ts = now - 5.0
    assert not eng._ws_synced(a, b)
    # exchange times 50ms apart -> genuinely simultaneous repricing -> synced,
    # even with arrival skew near the window edge (transport jitter)
    b.exchange_ts = now - 0.06
    assert eng._ws_synced(a, b)
    # missing exchange ts on one side falls back to arrival comparison
    b.exchange_ts = None
    assert eng._ws_synced(a, b)


def test_kalshi_parse_ticker_carries_exchange_ts():
    from bot.venues.kalshi import parse_ticker
    msg = {"type": "ticker", "msg": {
        "market_ticker": "KXT-1", "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.42",
        "yes_bid_size_fp": "10", "yes_ask_size_fp": "12", "ts_ms": 1783440160211}}
    quote = parse_ticker(msg)
    assert abs(quote.exchange_ts - 1783440160.211) < 1e-6


def test_rules_gate_blocks_unchecked_pairs():
    # A pair with NO rules verdict may not trade (the FTTS incident traded 4 minutes
    # after appearing, before the rules loop reached it); a checked pair fires.
    import time as _time
    fe = FakeExec()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0,
        max_ws_quote_age=5.0, require_rules_verify=True,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    now = _time.time()
    ka = q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100); ka.timestamp = now
    pa = q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60); pa.timestamp = now
    asyncio.run(eng.on_quote(ka)); asyncio.run(eng.on_quote(pa))
    assert fe.calls == []                          # unchecked -> blocked
    eng.rules_checked = {eng._pairs[next(iter(eng._pairs))].key} if hasattr(eng, "_pairs") else set()
    # simpler: mark via the pair's own key
    for p in [ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")]:
        eng.rules_checked = {p.key}
    eng._last_acted.clear()
    asyncio.run(eng.on_quote(ka)); asyncio.run(eng.on_quote(pa))
    assert len(fe.calls) == 1                      # checked -> trades


def test_reconcile_dust_remnant_does_not_halt():
    # a worthless leftover (bid ~1c on 5 contracts = $0.05) must not trip the kill
    # switch; the fail-closed path (unreadable book -> still naked) is covered by
    # test_reconcile_ignores_recycle_remnant's genuine-naked case.
    eng, fe = _settled_engine(
        open_states={"kalshi": True, "poly": True},
        depth_states={"kalshi": "MARKET_STATE_OPEN", "poly": "MARKET_STATE_OPEN"})
    async def penny_depth(venue, mid):
        return q(venue, mid, yes_ask=0.99, ya=1, no_ask=0.99, na=1)   # bids = 0.01
    eng.depth_fetch = penny_depth
    snaps = [_snap("kalshi", []), _snap("poly", [("P1", 5)])]
    asyncio.run(eng.reconcile_positions(snaps))
    asyncio.run(eng.reconcile_positions(snaps))
    assert not fe.risk.is_killed                     # $0.05 of dust: no halt


def test_one_way_pair_fires_only_safe_direction():
    # timing-scope pair: YES must sit on the wider-window venue (kalshi). The edge
    # here favors YES on POLY -> refused; flipping the books so kalshi is the YES
    # side -> fires.
    import time as _time
    fe = FakeExec()
    eng = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, max_ws_quote_age=5.0,
    )
    eng.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    pair = ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")
    eng.one_way_yes = {pair.key: "kalshi"}
    now = _time.time()
    # direction: YES poly (0.40) + NO kalshi (0.55) -> UNSAFE (yes on poly)
    ka = q("kalshi", "K1", yes_ask=0.62, ya=100, no_ask=0.55, na=100); ka.timestamp = now
    pa = q("poly", "P1", yes_ask=0.40, ya=100, no_ask=0.65, na=60); pa.timestamp = now
    asyncio.run(eng.on_quote(ka)); asyncio.run(eng.on_quote(pa))
    assert fe.calls == []
    # flip the books: YES kalshi (0.40) + NO poly (0.55) -> SAFE -> fires
    eng2 = StreamingEngine(
        executor=fe, fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
        min_edge=0.01, cooldown=100.0, clock=lambda: 0.0, max_ws_quote_age=5.0,
    )
    eng2.set_pairs([ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")])
    eng2.one_way_yes = {pair.key: "kalshi"}
    ka2 = q("kalshi", "K1", yes_ask=0.40, ya=100, no_ask=0.65, na=100); ka2.timestamp = now
    pa2 = q("poly", "P1", yes_ask=0.62, ya=100, no_ask=0.55, na=60); pa2.timestamp = now
    asyncio.run(eng2.on_quote(ka2)); asyncio.run(eng2.on_quote(pa2))
    assert len(fe.calls) == 1


def test_timing_scope_keys_from_legacy_rationale_and_new_column():
    from bot.data.store import Store
    st = Store(":memory:")
    # legacy row: material=0, rationale mentions extra time
    st.record_rules_verdict("kalshi", "KA", "polymarket_us", "PA",
                            identical=False, confidence=1.0,
                            rationale="Market A includes extra time while B does not")
    # new row: explicit category — but the belt+suspenders guard also requires
    # window language in the rationale (the category was briefly misused for
    # listing-time skew, which must not restrict anything)
    st.record_rules_verdict("kalshi", "KB", "polymarket_us", "PB",
                            identical=False, confidence=1.0,
                            rationale="A counts extra time while B settles on regulation",
                            divergence="timing_scope")
    # timing_scope WITHOUT window language (listing-skew misuse) -> not restricted
    st.record_rules_verdict("kalshi", "KD", "polymarket_us", "PD",
                            identical=False, confidence=1.0,
                            rationale="different resolution times", divergence="timing_scope")
    # different_event stays a hard drop, NOT one-way
    st.record_rules_verdict("kalshi", "KC", "polymarket_us", "PC",
                            identical=False, confidence=1.0,
                            rationale="different teams", material=True,
                            divergence="different_event")
    keys = st.timing_scope_keys()
    assert len(keys) == 2
    assert st._pair_key("kalshi", "KC", "polymarket_us", "PC") in st.rules_divergent_keys()
    assert st._pair_key("kalshi", "KA", "polymarket_us", "PA") not in st.rules_divergent_keys()
    st.close()


def test_livebook_asof_rewinds_event_time():
    import dataclasses
    from bot.streaming.engine import LiveBook
    lb = LiveBook()
    base = q("poly", "P1", yes_ask=0.40, ya=10)
    for ts, ask in ((100.0, 0.40), (100.2, 0.45), (100.4, 0.50)):
        quote = dataclasses.replace(base, yes_ask=ask)
        quote.exchange_ts = ts
        lb.update(quote)
    assert lb.asof("poly", "P1", 100.25).yes_ask == 0.45   # rewound between ticks
    assert lb.asof("poly", "P1", 100.5).yes_ask == 0.50    # latest
    assert lb.asof("poly", "P1", 99.9) is None             # before history


def test_aligned_edge_separates_standing_from_skew_phantom(monkeypatch):
    import dataclasses
    import time as _time
    ex = FakeExec()
    eng = make_engine(ex)
    now = _time.time()
    # SKEW PHANTOM: poly just moved (yes 0.40) creating an apparent edge against
    # kalshi's 300ms-old book (no 0.55); at kalshi's event time poly was 0.47 (no
    # edge). Rewinding poly must kill the fast-take.
    kq = q("kalshi", "K1", no_ask=0.55, na=50); kq.exchange_ts = now - 0.30
    kq.timestamp = now - 0.05
    p_old = q("poly", "P1", yes_ask=0.47, ya=50); p_old.exchange_ts = now - 0.35
    p_new = q("poly", "P1", yes_ask=0.40, ya=50); p_new.exchange_ts = now - 0.02
    p_new.timestamp = now - 0.01
    eng.livebook.update(p_old); eng.livebook.update(p_new)
    assert not eng._aligned_edge_ok(p_new, kq)
    # STANDING dislocation: poly was ALREADY 0.40 at kalshi's event time.
    eng2 = make_engine(FakeExec())
    p_old2 = q("poly", "P1", yes_ask=0.40, ya=50); p_old2.exchange_ts = now - 0.35
    eng2.livebook.update(p_old2); eng2.livebook.update(p_new)
    assert eng2._aligned_edge_ok(p_new, kq)


def test_resubscribe_streams_before_prime_completes():
    # THE dark-window regression test: a prime that never finishes must NOT block
    # quote consumption — consumers start first, prime runs in the background.
    # (Observed live: ~6min REST primes with consumers torn down, 17x in 3h.)
    import asyncio as aio

    ex = FakeExec()
    eng = make_engine(ex)
    blocked = aio.Event()

    async def never_fetch(venue, market):
        blocked.set()
        await aio.Event().wait()               # prime hangs forever

    eng.depth_fetch = never_fetch
    eng.prime_concurrency = 2
    consumed = []

    class _V:
        name = "kalshi"
        async def stream_order_book(self, mids):
            consumed.append(list(mids))
            q1 = q("kalshi", "K1", no_ask=0.55, na=50)
            yield q1
            await aio.Event().wait()

    async def refresh():
        return [ConfirmedPair("E1", "kalshi", "K1", "poly", "P1")]

    async def main():
        task = aio.create_task(eng.run([_V()], refresh, refresh_interval=9999))
        await aio.sleep(0.3)
        task.cancel()
        try:
            await task
        except aio.CancelledError:
            pass

    aio.run(main())
    assert consumed, "consumers never started while prime was blocked"
    assert blocked.is_set(), "background prime never ran"


def test_sweep_fires_fattest_edge_first():
    # Two standing edges after a prime: the 5c pair must claim capital before the
    # 1c pair — allocation order is edge-descending, not dict order.
    import asyncio as aio
    ex = FakeExec()
    eng = make_engine(ex, cooldown=0.0)
    eng.require_rules_verify = False
    eng.empirical_min_obs = 0
    eng.edge_persist_secs = 0.0
    eng.set_pairs([ConfirmedPair("THIN", "kalshi", "K1", "poly", "P1"),
                   ConfirmedPair("FAT", "kalshi", "K2", "poly", "P2")])
    # thin: 0.43+0.55 -> 2c ; fat: 0.40+0.55 -> 5c
    for quote in (q("poly", "P1", yes_ask=0.43, ya=50), q("kalshi", "K1", no_ask=0.55, na=50),
                  q("poly", "P2", yes_ask=0.40, ya=50), q("kalshi", "K2", no_ask=0.55, na=50)):
        eng.livebook.update(quote)
    async def _echo(venue, market):
        return eng.livebook.get(venue, market)   # REST confirm sees the same books
    eng.depth_fetch = _echo
    aio.run(eng.prime_and_sweep())
    assert len(ex.calls) == 2
    assert ex.calls[0].event_key == "FAT", \
        f"fat edge must fire first, got {[c.event_key for c in ex.calls]}"


def test_comovement_separates_dislocation_from_conflict():
    from collections import deque
    ex = FakeExec()
    eng = make_engine(ex)
    key = ("k",)
    # coupled legs: same news moves both (dYes up, dNo down) -> corr ~ +1
    eng._comove[key] = deque([(0.03, -0.03), (-0.02, 0.02), (0.04, -0.04),
                              (0.01, -0.01), (-0.03, 0.03), (0.02, -0.02),
                              (0.05, -0.05), (-0.01, 0.01)])
    assert eng._comove_corr(key) > 0.95
    # strangers: independent moves -> corr ~ 0
    eng._comove[key] = deque([(0.03, 0.02), (-0.02, 0.03), (0.04, -0.01),
                              (0.01, 0.04), (-0.03, -0.02), (0.02, 0.01),
                              (0.05, 0.02), (-0.01, -0.03)])
    assert abs(eng._comove_corr(key)) < 0.6
    # below 8 paired ticks -> no verdict either way
    eng._comove[key] = deque([(0.03, -0.03)] * 5)
    assert eng._comove_corr(key) is None
