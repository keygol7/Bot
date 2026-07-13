"""Read-only evidence audit for exact Kalshi/Polymarket.com crypto windows."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from zoneinfo import ZoneInfo


_ASSETS = ("BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE", "BNB")


def _margin_reference_spot(market: dict) -> float | None:
    """Convert a Kalshi perp's per-contract reference price to underlying spot."""
    try:
        price = float(market["reference_price"]["price"])
        contract_size = float(market["contract_size"])
        return price / contract_size if contract_size > 0 else None
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


async def _chainlink_snapshot(timeout: float = 8.0) -> dict[str, tuple[float, int]]:
    """Collect one current Polymarket RTDS Chainlink tick per available asset."""
    import websockets

    prices: dict[str, tuple[float, int]] = {}
    deadline = time.monotonic() + timeout
    async with websockets.connect(
        "wss://ws-live-data.polymarket.com", open_timeout=5.0, ping_interval=None,
    ) as socket:
        await socket.send(json.dumps({
            "action": "subscribe",
            "subscriptions": [{
                "topic": "crypto_prices_chainlink", "type": "*", "filters": "",
            }],
        }))
        while time.monotonic() < deadline and len(prices) < len(_ASSETS):
            remaining = deadline - time.monotonic()
            try:
                raw = await asyncio.wait_for(socket.recv(), max(0.05, remaining))
            except TimeoutError:
                break
            if raw == "PING":
                await socket.send("PONG")
                continue
            try:
                message = json.loads(raw)
                payload = message.get("payload") or {}
                asset = str(payload.get("symbol") or "").split("/", 1)[0].upper()
                value = float(payload["value"])
                timestamp = int(payload.get("timestamp") or message.get("timestamp") or 0)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if asset in _ASSETS and value > 0:
                prices[asset] = (value, timestamp)
    return prices


async def crypto_oracle_snapshot(settings) -> int:
    """Compare each live binary's two actual settlement-source price paths."""
    import httpx

    logging.getLogger("httpx").setLevel(logging.WARNING)
    now = time.time()
    start = int(now // 900) * 900
    end = start + 900
    start_iso = datetime.fromtimestamp(start, UTC).isoformat().replace("+00:00", "Z")
    end_iso = datetime.fromtimestamp(end, UTC).isoformat().replace("+00:00", "Z")
    pairs = {(_asset(pair[1]) or "?"): pair for pair in current_pair_ids(now)}

    async with httpx.AsyncClient(timeout=12.0, http2=True) as client:
        margin_request = client.get(
            f"{settings.kalshi.api_base.rstrip('/')}/margin/markets"
        )

        async def binary_sources(asset: str, pair: tuple):
            kalshi_id = pair[1]
            try:
                kalshi_response, poly_response = await asyncio.gather(
                    client.get(
                        f"{settings.kalshi.api_base.rstrip('/')}/markets/{kalshi_id}"
                    ),
                    client.get(
                        "https://polymarket.com/api/crypto/crypto-price",
                        params={
                            "symbol": asset, "eventStartTime": start_iso,
                            "variant": "fifteen", "endDate": end_iso,
                        },
                    ),
                )
                kalshi_response.raise_for_status()
                poly_response.raise_for_status()
                return (
                    float(kalshi_response.json()["market"]["floor_strike"]),
                    float(poly_response.json()["openPrice"]),
                )
            except (httpx.HTTPError, KeyError, TypeError, ValueError):
                return None, None

        source_task = asyncio.gather(*(
            binary_sources(asset, pair) for asset, pair in pairs.items()
        ))
        chainlink_task = asyncio.create_task(_chainlink_snapshot())
        margin_response, source_rows, chainlink = await asyncio.gather(
            margin_request, source_task, chainlink_task,
        )
        margin_response.raise_for_status()

    perps = {
        str(market.get("ticker") or ""): market
        for market in margin_response.json().get("markets", [])
    }
    print("live settlement-source state (Kalshi CF RTI vs Polymarket Chainlink)")
    print(f"  window: {start_iso} to {end_iso}")
    print(f"\n  {'asset':<6}{'CF move':>10}{'CL move':>10}{'path gap':>11}"
          f"{'same side':>12}  source availability")
    for (asset, _pair), (kalshi_open, poly_open) in zip(pairs.items(), source_rows):
        cf_now = _margin_reference_spot(perps.get(f"KX{asset}PERP", {}))
        cl_tick = chainlink.get(asset)
        cl_now = cl_tick[0] if cl_tick else None
        cf_bps = ((cf_now / kalshi_open) - 1.0) * 10_000.0 if (
            cf_now is not None and kalshi_open
        ) else None
        cl_bps = ((cl_now / poly_open) - 1.0) * 10_000.0 if (
            cl_now is not None and poly_open
        ) else None
        if cf_bps is not None and cl_bps is not None:
            same = (cf_bps >= 0) == (cl_bps >= 0)
            gap = abs(cf_bps - cl_bps)
            print(f"  {asset:<6}{cf_bps:>+9.2f}bp{cl_bps:>+9.2f}bp"
                  f"{gap:>10.2f}bp{str(same):>12}  CF perp + Chainlink RTDS")
        else:
            have_cf = "CF" if cf_bps is not None else "no-CF-perp"
            have_cl = "Chainlink" if cl_bps is not None else "no-RTDS-symbol"
            print(f"  {asset:<6}{'n/a':>10}{'n/a':>10}{'n/a':>11}{'n/a':>12}  "
                  f"{have_cf} / {have_cl}")
    print("\nThis is an oracle-path safety signal, not an arb by itself. A production "
          "entry should require both paths on the same side of their own opening "
          "threshold, sufficient distance from zero, and a source-adjusted quoted edge.")
    return 0


def source_adjustment(
    double_wins: int, double_losses: int, same: int, *, z: float = 1.2815515655,
) -> tuple[float, float]:
    """Return posterior mean and one-sided lower source adjustment per contract.

    A cross-oracle basket pays $2 on a double win, $0 on a double loss and $1 when
    both venues agree. A Jeffreys Dirichlet prior keeps tiny samples from claiming
    zero source risk. The default is an approximate 90% one-sided lower bound for
    P(double-win) minus P(double-loss).
    """
    aw, al, agree = double_wins + 0.5, double_losses + 0.5, same + 0.5
    total = aw + al + agree
    mean = (aw - al) / total
    variance = ((aw + al) * total - (aw - al) ** 2) / (
        total * total * (total + 1.0)
    )
    return mean, mean - z * math.sqrt(max(0.0, variance))


def _asset(kalshi_id: str) -> str | None:
    match = re.fullmatch(r"KX(BTC|ETH|SOL|XRP|DOGE|HYPE|BNB)15M-.+", kalshi_id)
    return match.group(1) if match else None


def _poly_result(payload: dict) -> str | None:
    # A still-open market can trade at 0.995/0.005 before resolution. Treating that
    # quote as settlement fabricates realized P&L; Gamma's terminal record is closed
    # and pins the outcome vector exactly to 1/0.
    if payload.get("closed") is not True:
        return None
    try:
        prices = json.loads(payload.get("outcomePrices") or "[]")
        if len(prices) >= 2 and float(prices[0]) == 1.0 and float(prices[1]) == 0.0:
            return "yes"
        if len(prices) >= 2 and float(prices[0]) == 0.0 and float(prices[1]) == 1.0:
            return "no"
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return None


def current_pair_ids(now: float) -> list[tuple]:
    """Construct the currently active exact 15-minute ids without waiting for a scan."""
    start = int(now // 900) * 900
    local_end = datetime.fromtimestamp(
        start + 900, ZoneInfo("America/New_York")
    )
    kalshi_window = local_end.strftime("%y%b%d%H%M").upper()
    minute = local_end.strftime("%M")
    out = []
    for asset in _ASSETS:
        kalshi_id = f"KX{asset}15M-{kalshi_window}-{minute}"
        pcom_id = f"{asset.lower()}-updown-15m-{start}"
        out.append((
            "kalshi", kalshi_id, "polymarket_com", pcom_id,
            f"kalshi:{kalshi_id}|polymarket_com:{pcom_id}",
        ))
    return out


def historical_pair_ids(now: float, hours: float) -> list[tuple]:
    """Construct recently ended windows; independent of the pruned live-market cache."""
    end_before = int(now // 900) * 900
    intervals = max(1, min(192, math.ceil(max(0.25, hours) * 4)))
    out = []
    for offset in range(1, intervals + 1):
        # ``current_pair_ids`` floors to this interval and constructs its ending id.
        out.extend(current_pair_ids(end_before - offset * 900 + 1))
    return out


async def crypto_edge_snapshot(settings, *, history_hours: float = 48.0) -> int:
    """Print concurrent REST depth edges plus empirical cross-oracle source risk."""
    import httpx

    logging.getLogger("httpx").setLevel(logging.WARNING)

    from bot.dryrun import _build_venues
    from bot.strategies.arbitrage import detect_cross_venue
    from bot.venues.base import RawMarket

    now = time.time()
    # Recurring markets are pruned from the live cache once settled and are often
    # listed after a slow whole-board scan begins. Generate both current and historical
    # identifiers directly so neither the strategy nor its calibration waits on scans.
    active = current_pair_ids(now)
    settled = historical_pair_ids(now, min(history_hours, 48.0))

    counts: dict[str, Counter] = defaultdict(Counter)
    semaphore = asyncio.Semaphore(20)
    async with httpx.AsyncClient(timeout=15.0, http2=True) as client:
        async def result(pair):
            async with semaphore:
                kalshi_id, pcom_id = pair[1], pair[3]
                try:
                    ka, po = await asyncio.gather(
                        client.get(
                            f"{settings.kalshi.api_base.rstrip('/')}/markets/{kalshi_id}"
                        ),
                        client.get(
                            f"{settings.polymarket_com.gamma_base.rstrip('/')}/markets/slug/"
                            f"{pcom_id}"
                        ),
                    )
                    ka.raise_for_status()
                    po.raise_for_status()
                    kr = str(ka.json().get("market", {}).get("result") or "").lower()
                    pr = _poly_result(po.json())
                    asset = _asset(kalshi_id)
                    if asset and kr in ("yes", "no") and pr in ("yes", "no"):
                        return asset, kr, pr
                except Exception:
                    return None
                return None

        history = await asyncio.gather(*(result(pair) for pair in settled))
    for item in history:
        if item is not None:
            asset, kr, pr = item
            counts[asset][(kr, pr)] += 1
    pooled = sum(counts.values(), Counter())

    venues = {venue.name: venue for venue in _build_venues(settings)}

    async def quote_pair(pair):
        kalshi_id, pcom_id = pair[1], pair[3]
        try:
            kq, pq = await asyncio.gather(
                venues["kalshi"].fetch_quote(RawMarket(kalshi_id, "", {})),
                venues["polymarket_com"].fetch_quote(RawMarket(pcom_id, "", {})),
            )
            if kq is None or pq is None:
                return pair, None, "missing book"
            opportunities = detect_cross_venue(
                kq, pq,
                fee_a=venues["kalshi"].fee_model,
                fee_b=venues["polymarket_com"].fee_model,
                min_edge=-1.0,
            )
            return pair, max(opportunities, key=lambda o: o.edge_per_contract), None
        except Exception as exc:
            return pair, None, f"{type(exc).__name__}: {exc}"

    rows = await asyncio.gather(*(quote_pair(pair) for pair in active))
    print(f"exact active crypto windows: {len(active)}; settled source samples: "
          f"{sum(pooled.values())}")
    if not active:
        print("  (no exact shared 15-minute window is active right now)")
        return 0
    print("  source-adjusted lower edge uses a 90% one-sided Jeffreys bound; "
          "assets with <30 samples use the pooled history")
    print(f"\n  {'asset':<6}{'raw_edge':>10}{'depth':>10}{'raw_profit':>12}"
          f"{'mean_edge':>11}{'lower_edge':>12}{'after_1c':>11}  route / evidence")
    for pair, opportunity, error in sorted(rows, key=lambda row: row[0][1]):
        asset = _asset(pair[1]) or "?"
        if opportunity is None:
            print(f"  {asset:<6}{'n/a':>10}  {error}")
            continue
        asset_counts = counts.get(asset, Counter())
        evidence = asset_counts if sum(asset_counts.values()) >= 30 else pooled
        evidence_name = asset if evidence is asset_counts else "pooled"
        # Counts are expressed for YES@Kalshi + NO@Pcom. Reverse windfall/loss
        # categories when the cheapest executable direction is reversed.
        double_wins = evidence[("yes", "no")]
        double_losses = evidence[("no", "yes")]
        if opportunity.buy_yes_venue != "kalshi":
            double_wins, double_losses = double_losses, double_wins
        same = evidence[("yes", "yes")] + evidence[("no", "no")]
        mean_adj, lower_adj = source_adjustment(double_wins, double_losses, same)
        raw_edge = opportunity.total_profit / opportunity.max_contracts
        mean_edge, lower_edge = raw_edge + mean_adj, raw_edge + lower_adj
        after_reserve = lower_edge - 0.01
        print(
            f"  {asset:<6}{raw_edge:>+10.4f}{opportunity.max_contracts:>10.1f}"
            f"{opportunity.total_profit:>+12.2f}{mean_edge:>+11.4f}"
            f"{lower_edge:>+12.4f}{after_reserve:>+11.4f}  "
            f"YES@{opportunity.buy_yes_venue} / NO@{opportunity.buy_no_venue}; "
            f"{evidence_name} n={sum(evidence.values())} W/L={double_wins}/{double_losses}"
        )
    print("\nRaw edge is a concurrent quoted-depth scenario, not guaranteed arbitrage. "
          "A positive after_1c value passes only the HISTORICAL source-risk screen; "
          "it is not entry permission. Also require --crypto-oracle-snapshot to show "
          "both live source paths safely on the same side, then validate settlement "
          "and fills in shadow before funding.")
    return 0


async def crypto_performance(settings, *, hours: float = 24.0) -> int:
    """Reconcile ended shadow episodes and report first-qualified realized P&L."""
    import httpx

    from bot.data.store import Store

    logging.getLogger("httpx").setLevel(logging.WARNING)
    store = Store(settings.db_path)
    now, since = time.time(), time.time() - hours * 3600.0
    try:
        pending = store.conn.execute(
            """SELECT id, event_key, yes_venue, yes_market, no_venue, no_market
                 FROM shadow_edge_episodes
                WHERE started_ts>=? AND settlement_checked_ts IS NULL
                  AND event_key LIKE 'kalshi:KX%15M-%|polymarket_com:%-updown-15m-%'""",
            (since,),
        ).fetchall()
        eligible = []
        for row in pending:
            pcom_id = (row["yes_market"] if row["yes_venue"] == "polymarket_com"
                       else row["no_market"])
            try:
                start = int(pcom_id.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if now > start + 930:
                eligible.append(row)

        markets = {
            (row[side + "_venue"], row[side + "_market"])
            for row in eligible for side in ("yes", "no")
        }
        semaphore = asyncio.Semaphore(20)
        reference_assets: set[str] = set()
        async with httpx.AsyncClient(timeout=15.0, http2=True) as client:
            async def fetch_result(item):
                venue, market = item
                async with semaphore:
                    try:
                        if venue == "kalshi":
                            response = await client.get(
                                f"{settings.kalshi.api_base.rstrip('/')}/markets/{market}"
                            )
                            response.raise_for_status()
                            result = str(
                                response.json().get("market", {}).get("result") or ""
                            ).lower()
                        else:
                            response = await client.get(
                                f"{settings.polymarket_com.gamma_base.rstrip('/')}/"
                                f"markets/slug/{market}"
                            )
                            response.raise_for_status()
                            result = _poly_result(response.json())
                        return item, result if result in ("yes", "no") else None
                    except Exception:
                        return item, None

            results = dict(await asyncio.gather(*(fetch_result(item) for item in markets)))
            try:
                margin = await client.get(
                    f"{settings.kalshi.api_base.rstrip('/')}/margin/markets"
                )
                margin.raise_for_status()
                reference_assets = {
                    str(market.get("ticker") or "")[2:-4]
                    for market in margin.json().get("markets", [])
                    if str(market.get("ticker") or "").startswith("KX")
                    and str(market.get("ticker") or "").endswith("PERP")
                    and market.get("reference_price")
                }
            except Exception:
                pass

        reconciled = 0
        for row in eligible:
            yes_result = results.get((row["yes_venue"], row["yes_market"]))
            no_result = results.get((row["no_venue"], row["no_market"]))
            if yes_result is None or no_result is None:
                continue
            payout = float(yes_result == "yes") + float(no_result == "no")
            reconciled += store.settle_shadow_episode(
                row["id"], yes_result=yes_result, no_result=no_result, payout=payout,
            )

        rows = store.conn.execute(
            """SELECT id,event_key,qualified_ts,qualified_edge,qualified_size,
                      qualified_skew_ms,settlement_payout,realized_edge_per_contract,
                      realized_profit,yes_venue,no_venue,source_qualified_ts,
                      source_qualified_edge,source_qualified_size,
                      source_qualified_skew_ms,source_cf_move_bps,
                      source_chainlink_move_bps,source_path_gap_bps,
                      source_min_distance_bps,source_remaining_s,
                      source_uncertainty_bps,source_cf_samples,path_qualified_ts,
                      path_qualified_edge,path_qualified_size,path_qualified_skew_ms,
                      path_cf_move_bps,path_chainlink_move_bps,path_gap_bps,
                      path_min_distance_bps,path_remaining_s,path_uncertainty_bps
                 FROM shadow_edge_episodes
                WHERE started_ts>=? AND qualified_ts IS NOT NULL
                  AND settlement_checked_ts IS NOT NULL
                  AND event_key LIKE 'kalshi:KX%15M-%|polymarket_com:%-updown-15m-%'
                ORDER BY qualified_ts""",
            (since,),
        ).fetchall()
    finally:
        store.close()

    # A deployable strategy gets one entry per asset/window, at the first moment the
    # persistence+sync rule qualifies. Summing every later flicker would overstate P&L.
    first_by_event = {}
    for row in rows:
        first_by_event.setdefault(row["event_key"], row)
    selected = list(first_by_event.values())
    print(f"crypto shadow reconciliation: {reconciled} episode row(s) newly resolved; "
          f"{len(selected)} first-qualified completed window(s) in {hours:g}h")
    if not selected:
        print("  (none authoritatively settled yet; recently ended Polymarket "
              "markets may still be finalizing)")
        return 0
    print(f"\n  {'asset':<6}{'entry':>9}{'size':>9}{'skew_ms':>10}{'payout':>9}"
          f"{'realized':>11}{'after_1c':>11}{'net_profit':>12}")
    total = 0.0
    wins = 0
    covered_total = 0.0
    covered_wins = 0
    covered_count = 0
    for row in selected:
        kalshi_id = row["event_key"].split("|", 1)[0].split(":", 1)[1]
        asset = _asset(kalshi_id) or "?"
        after_reserve = float(row["realized_edge_per_contract"]) - 0.01
        net_profit = after_reserve * float(row["qualified_size"])
        total += net_profit
        wins += after_reserve > 0
        if asset in reference_assets:
            covered_total += net_profit
            covered_wins += after_reserve > 0
            covered_count += 1
        print(f"  {asset:<6}{row['qualified_edge']:>+9.4f}{row['qualified_size']:>9.1f}"
              f"{row['qualified_skew_ms']:>10.1f}{row['settlement_payout']:>9.0f}"
              f"{row['realized_edge_per_contract']:>+11.4f}{after_reserve:>+11.4f}"
              f"{net_profit:>+12.2f}")
    print(f"\n  wins after reserve: {wins}/{len(selected)}; counterfactual net at observed "
          f"qualified depth: ${total:+.2f}")
    if reference_assets:
        print(f"  reference-covered baseline ({','.join(sorted(reference_assets & set(_ASSETS)))}): "
              f"{covered_wins}/{covered_count} wins, ${covered_total:+.2f}; this excludes "
              "assets the live CF oracle cannot observe but is not itself the final gate")
    path_first = {}
    for row in rows:
        if row["path_qualified_ts"] is not None:
            path_first.setdefault(row["event_key"], row)
    path_selected = list(path_first.values())
    print(f"\n  path-aligned statistical entries: {len(path_selected)}")
    if path_selected:
        path_total = 0.0
        path_wins = 0
        print(f"  {'asset':<6}{'entry':>9}{'size':>9}{'remain':>9}{'CF_bp':>9}"
              f"{'CL_bp':>9}{'gap_bp':>9}{'after_1c':>11}{'net_profit':>12}")
        for row in path_selected:
            kalshi_id = row["event_key"].split("|", 1)[0].split(":", 1)[1]
            asset = _asset(kalshi_id) or "?"
            realized = float(row["path_qualified_edge"]) + float(
                row["settlement_payout"]
            ) - 1.0
            after_reserve = realized - 0.01
            net_profit = after_reserve * float(row["path_qualified_size"])
            path_total += net_profit
            path_wins += after_reserve > 0
            print(
                f"  {asset:<6}{row['path_qualified_edge']:>+9.4f}"
                f"{row['path_qualified_size']:>9.1f}{row['path_remaining_s']:>9.1f}"
                f"{row['path_cf_move_bps']:>+9.2f}{row['path_chainlink_move_bps']:>+9.2f}"
                f"{row['path_gap_bps']:>9.2f}{after_reserve:>+11.4f}"
                f"{net_profit:>+12.2f}"
            )
        print(f"  path-aligned wins: {path_wins}/{len(path_selected)}; "
              f"counterfactual net: ${path_total:+.2f}")
    else:
        print("  (none settled yet with continuously observed aligned source paths)")
    source_first = {}
    for row in rows:
        if row["source_qualified_ts"] is not None:
            source_first.setdefault(row["event_key"], row)
    source_selected = list(source_first.values())
    print(f"\n  source-gated first entries: {len(source_selected)}")
    if source_selected:
        print(f"  {'asset':<6}{'entry':>9}{'size':>9}{'remain':>9}{'CF_bp':>9}"
              f"{'CL_bp':>9}{'gap_bp':>9}{'after_1c':>11}{'net_profit':>12}")
        source_total = 0.0
        source_wins = 0
        for row in source_selected:
            kalshi_id = row["event_key"].split("|", 1)[0].split(":", 1)[1]
            asset = _asset(kalshi_id) or "?"
            realized = float(row["source_qualified_edge"]) + float(
                row["settlement_payout"]
            ) - 1.0
            after_reserve = realized - 0.01
            net_profit = after_reserve * float(row["source_qualified_size"])
            source_total += net_profit
            source_wins += after_reserve > 0
            print(
                f"  {asset:<6}{row['source_qualified_edge']:>+9.4f}"
                f"{row['source_qualified_size']:>9.1f}{row['source_remaining_s']:>9.1f}"
                f"{row['source_cf_move_bps']:>+9.2f}{row['source_chainlink_move_bps']:>+9.2f}"
                f"{row['source_path_gap_bps']:>9.2f}{after_reserve:>+11.4f}"
                f"{net_profit:>+12.2f}"
            )
        print(f"  source-gated wins: {source_wins}/{len(source_selected)}; "
              f"counterfactual net: ${source_total:+.2f}")
    else:
        print("  (none yet; continuous oracle gating was introduced after the earlier "
              "episodes and only permits entries in the final 15 seconds)")
    print("\n  Settlement-realized shadow P&L still assumes both FOK legs would fill "
          "at the observed books.")
    return 0
