

def test_apply_book_message_snapshot_and_deltas():
    from bot.venues.kalshi import apply_book_message
    books, seqs = {}, {}
    # snapshot: NO bids at 40c/38c, YES bids at 55c -> yes_ask = 1 - 0.40 = 0.60
    q, gap = apply_book_message(books, seqs, {
        "type": "orderbook_snapshot", "sid": 2, "seq": 1,
        "msg": {"market_ticker": "T1", "yes": [[55, 100]], "no": [[40, 30], [38, 50]],
                "ts": 1751980000000}})
    assert not gap and q is not None
    assert q.yes_ask == 0.60 and q.yes_ask_size == 30
    assert q.no_ask == 0.45 and q.no_ask_size == 100
    assert q.exchange_ts == 1751980000.0
    assert len(q.yes_ask_levels) == 2                    # the ladder came through
    # delta: NO 40c bid grows by 20
    q, gap = apply_book_message(books, seqs, {
        "type": "orderbook_delta", "sid": 2, "seq": 2,
        "msg": {"market_ticker": "T1", "side": "no", "price": 40, "delta": 20}})
    assert not gap and q.yes_ask_size == 50
    # delta: NO 40c bid removed entirely -> next level (38c) becomes best
    q, gap = apply_book_message(books, seqs, {
        "type": "orderbook_delta", "sid": 2, "seq": 3,
        "msg": {"market_ticker": "T1", "side": "no", "price": 40, "delta": -50}})
    assert not gap and q.yes_ask == 0.62 and q.yes_ask_size == 50


def test_apply_book_message_seq_gap_demands_reconnect():
    from bot.venues.kalshi import apply_book_message
    books, seqs = {}, {}
    apply_book_message(books, seqs, {
        "type": "orderbook_snapshot", "sid": 2, "seq": 1,
        "msg": {"market_ticker": "T1", "yes": [[50, 10]], "no": [[45, 10]]}})
    # seq jumps 1 -> 3: a delta was lost; the local book is untrustworthy
    q, gap = apply_book_message(books, seqs, {
        "type": "orderbook_delta", "sid": 2, "seq": 3,
        "msg": {"market_ticker": "T1", "side": "no", "price": 45, "delta": 5}})
    assert gap and q is None
    # a delta arriving before any snapshot for its market is ignored, not applied
    q, gap = apply_book_message(books, {}, {
        "type": "orderbook_delta", "sid": 9, "seq": 1,
        "msg": {"market_ticker": "T2", "side": "yes", "price": 30, "delta": 10}})
    assert not gap and q is None


def test_apply_book_message_live_fp_shape():
    # the shape kalshi actually sends (observed 2026-07-08): string fp dollars,
    # delta_fp, ISO-8601 ts
    from bot.venues.kalshi import apply_book_message
    books, seqs = {}, {}
    q, gap = apply_book_message(books, seqs, {
        "type": "orderbook_snapshot", "sid": 2, "seq": 1,
        "msg": {"market_ticker": "T1",
                "yes_dollars": [["0.5500", "100.00"]],
                "no_dollars": [["0.4000", "30.00"]],
                "ts": "2026-07-08T22:31:17.334187Z"}})
    assert not gap and q.yes_ask == 0.60 and q.no_ask == 0.45
    assert q.exchange_ts is not None and q.exchange_ts > 1.7e9
    q, gap = apply_book_message(books, seqs, {
        "type": "orderbook_delta", "sid": 2, "seq": 2,
        "msg": {"market_ticker": "T1", "price_dollars": "0.4000",
                "delta_fp": "-30.00", "side": "no",
                "ts": "2026-07-08T22:31:18.000000Z"}})
    assert not gap and q.yes_ask is None        # the only NO bid was removed
    assert q.no_ask == 0.45                     # yes side untouched


def test_apply_book_message_preserves_subpenny_dollar_prices():
    from bot.venues.kalshi import apply_book_message
    books, seqs = {}, {}
    q, gap = apply_book_message(books, seqs, {
        "type": "orderbook_snapshot", "sid": 4, "seq": 1,
        "msg": {"market_ticker": "KXBTC15M-TEST",
                "yes_dollars": [["0.9990", "12.00"]],
                "no_dollars": [["0.9980", "7.00"]]},
    })
    assert not gap
    assert q.no_ask == 0.001 and q.yes_ask == 0.002
    assert q.timestamp > 0

    q, gap = apply_book_message(books, seqs, {
        "type": "orderbook_delta", "sid": 4, "seq": 2,
        "msg": {"market_ticker": "KXBTC15M-TEST", "side": "yes",
                "price_dollars": "0.9990", "delta_fp": "-12.00"},
    })
    assert not gap and q.no_ask is None and q.yes_ask == 0.002
