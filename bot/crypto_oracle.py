"""Continuous settlement-source monitor for Kalshi/Pcom short crypto windows.

Kalshi's binary uses a CF Benchmarks one-minute average; Polymarket.com uses a
Chainlink point price.  A cheap YES+NO basket is therefore not a lock unless both
source paths finish on the same side of their own opening thresholds.  This module
keeps the two actual sources hot and exposes a small synchronous assessment callable
for the streaming engine's per-book-update path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from bot.crypto_audit import _ASSETS, current_pair_ids
from bot.venues.kalshi import build_signature_headers, load_private_key


log = logging.getLogger("bot.crypto_oracle")

_MARGIN_WS = "wss://external-api-margin-ws.kalshi.com/trade-api/ws/v2/margin"
_MARGIN_WS_PATH = "/trade-api/ws/v2/margin"
_RTDS_WS = "wss://ws-live-data.polymarket.com"


@dataclass(frozen=True)
class OracleAssessment:
    eligible: bool
    reason: str
    path_aligned: bool = False
    path_reason: str = "unavailable"
    asset: str | None = None
    window_start: int | None = None
    remaining_s: float | None = None
    cf_move_bps: float | None = None
    chainlink_move_bps: float | None = None
    path_gap_bps: float | None = None
    min_distance_bps: float | None = None
    cf_age_ms: float | None = None
    chainlink_age_ms: float | None = None
    cf_samples: int = 0
    uncertainty_bps: float | None = None


class CryptoOracleMonitor:
    """Maintain current CF/Chainlink ticks and exact window opening thresholds."""

    def __init__(
        self, settings, *, store=None, entry_last_s: float = 15.0,
        max_tick_age_s: float = 2.5, base_buffer_bps: float = 2.0,
    ) -> None:
        self.settings = settings
        self.store = store
        self.entry_last_s = entry_last_s
        self.max_tick_age_s = max_tick_age_s
        self.base_buffer_bps = base_buffer_bps
        self.thresholds: dict[tuple[int, str], dict[str, Any]] = {}
        self.contract_sizes: dict[str, float] = {}
        self.cf_ticks: dict[str, tuple[float, float]] = {}
        self.chainlink_ticks: dict[str, tuple[float, float]] = {}
        self.cf_final_minute: dict[tuple[int, str], deque] = defaultdict(
            lambda: deque(maxlen=90)
        )
        self.chainlink_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=180)
        )
        self._private_key = None

    @staticmethod
    def _asset_window(event_key: str) -> tuple[str | None, int | None]:
        try:
            pcom = next(
                leg for leg in event_key.split("|") if leg.startswith("polymarket_com:")
            ).split(":", 1)[1]
            asset, marker, start = pcom.partition("-updown-15m-")
            if marker and asset.upper() in _ASSETS:
                return asset.upper(), int(start)
        except (StopIteration, ValueError):
            pass
        return None, None

    def _update_cf(self, asset: str, spot: float, exchange_ts: float) -> None:
        if asset not in _ASSETS or spot <= 0:
            return
        self.cf_ticks[asset] = (spot, exchange_ts)
        start = int(exchange_ts // 900) * 900
        if start + 840 <= exchange_ts <= start + 902:
            samples = self.cf_final_minute[(start, asset)]
            # Coalesced ticker messages are at most 1Hz. Replace an existing sample
            # in the same second instead of overweighting reconnect/snapshot bursts.
            second = int(exchange_ts)
            if samples and int(samples[-1][0]) == second:
                samples[-1] = (exchange_ts, spot)
            else:
                samples.append((exchange_ts, spot))

    def _update_chainlink(self, asset: str, value: float, exchange_ts: float) -> None:
        if asset not in _ASSETS or value <= 0:
            return
        self.chainlink_ticks[asset] = (value, exchange_ts)
        history = self.chainlink_history[asset]
        if history and int(history[-1][0] * 10) == int(exchange_ts * 10):
            history[-1] = (exchange_ts, value)
        else:
            history.append((exchange_ts, value))

    def _chainlink_uncertainty(self, asset: str, remaining_s: float) -> float:
        """Approximate 99% remaining point-price motion in basis points."""
        history = list(self.chainlink_history.get(asset) or ())
        cutoff = time.time() - 30.0
        history = [row for row in history if row[0] >= cutoff]
        moves = []
        for (ta, pa), (tb, pb) in zip(history, history[1:]):
            dt = max(0.05, tb - ta)
            moves.append(((pb / pa - 1.0) * 10_000.0) / math.sqrt(dt))
        sigma = statistics.pstdev(moves) if len(moves) >= 5 else 1.0
        return 2.58 * sigma * math.sqrt(max(0.0, remaining_s))

    def assess(self, event_key: str, *, now: float | None = None) -> OracleAssessment:
        """Return the current hold-to-settlement safety decision for one exact pair."""
        asset, start = self._asset_window(event_key)
        if asset is None or start is None:
            return OracleAssessment(False, "not_exact_crypto")
        now = time.time() if now is None else now
        remaining = start + 900 - now
        base = dict(asset=asset, window_start=start, remaining_s=remaining)
        threshold = self.thresholds.get((start, asset))
        if not threshold or not threshold.get("kalshi_open") or not threshold.get("pcom_open"):
            return OracleAssessment(False, "threshold_missing", **base)
        cf_tick = self.cf_ticks.get(asset)
        cl_tick = self.chainlink_ticks.get(asset)
        if cf_tick is None:
            return OracleAssessment(False, "cf_reference_missing", **base)
        if cl_tick is None:
            return OracleAssessment(False, "chainlink_tick_missing", **base)
        cf_age = max(0.0, now - cf_tick[1])
        cl_age = max(0.0, now - cl_tick[1])
        ages = dict(cf_age_ms=cf_age * 1000.0, chainlink_age_ms=cl_age * 1000.0)
        if cf_age > self.max_tick_age_s or cl_age > self.max_tick_age_s:
            return OracleAssessment(False, "oracle_tick_stale", **base, **ages)
        if remaining <= 0:
            return OracleAssessment(False, "window_ended", **base, **ages)
        samples = []
        if remaining <= 60.0:
            samples = [
                value for ts, value in self.cf_final_minute.get((start, asset), ())
                if start + 840 <= ts <= now
            ]
            elapsed_final_minute = max(0.0, min(60.0, now - (start + 840)))
            minimum_samples = max(1, int(elapsed_final_minute) - 2)
            if len(samples) < minimum_samples:
                return OracleAssessment(
                    False, "cf_average_incomplete", cf_samples=len(samples), **base, **ages,
                )
            # Kalshi settles on all 60 one-second observations, not the partial
            # average visible right now. Project each still-unobserved slot from
            # the live CF reference so a late RTI reversal cannot masquerade as a
            # locked partial average.
            remaining_cf_slots = max(0, 60 - len(samples))
            cf_value = (
                sum(samples) + remaining_cf_slots * cf_tick[0]
            ) / (len(samples) + remaining_cf_slots)
        else:
            # Before Kalshi's averaging minute begins, spot is only a path signal;
            # the conservative entry gate remains closed, but persisting these metrics
            # lets the offline audit evaluate broader statistical strategies.
            cf_value = cf_tick[0]
        cl_value = cl_tick[0]
        cf_move = (cf_value / threshold["kalshi_open"] - 1.0) * 10_000.0
        cl_move = (cl_value / threshold["pcom_open"] - 1.0) * 10_000.0
        gap = abs(cf_move - cl_move)
        distance = min(abs(cf_move), abs(cl_move))
        chainlink_uncertainty = self._chainlink_uncertainty(asset, remaining)
        # Translate possible remaining CF point motion into its weighted effect on
        # the final 60-second mean. The Chainlink point-price side retains the full
        # bound, so the larger of the two governs the entry decision.
        cf_uncertainty = (
            chainlink_uncertainty * remaining_cf_slots / 60.0
            if remaining <= 60.0 else chainlink_uncertainty
        )
        uncertainty = max(chainlink_uncertainty, cf_uncertainty)
        metrics = dict(
            cf_move_bps=cf_move, chainlink_move_bps=cl_move,
            path_gap_bps=gap, min_distance_bps=distance,
            cf_samples=len(samples), uncertainty_bps=uncertainty,
        )
        same_side = (cf_move >= 0) == (cl_move >= 0)
        alignment_required = self.base_buffer_bps + gap
        path_aligned = same_side and distance >= alignment_required
        path_reason = (
            "aligned" if path_aligned else
            "source_paths_disagree" if not same_side else
            "source_margin_too_small"
        )
        metrics.update(path_aligned=path_aligned, path_reason=path_reason)
        if not same_side:
            return OracleAssessment(False, "source_paths_disagree", **base, **ages, **metrics)
        if remaining > self.entry_last_s:
            return OracleAssessment(
                False, "too_early_for_source_lock", **base, **ages, **metrics,
            )
        required = max(self.base_buffer_bps + gap, uncertainty)
        if distance < required:
            return OracleAssessment(False, "source_margin_too_small", **base, **ages, **metrics)
        return OracleAssessment(True, "eligible", **base, **ages, **metrics)

    async def _refresh_thresholds(self) -> None:
        import httpx

        now = time.time()
        pairs = current_pair_ids(now)
        start = int(now // 900) * 900
        cutoff = start - 4 * 3600
        self.thresholds = {key: value for key, value in self.thresholds.items()
                           if key[0] >= cutoff}
        for key in [key for key in self.cf_final_minute if key[0] < cutoff]:
            del self.cf_final_minute[key]
        start_iso = datetime.fromtimestamp(start, UTC).isoformat().replace("+00:00", "Z")
        end_iso = datetime.fromtimestamp(start + 900, UTC).isoformat().replace(
            "+00:00", "Z"
        )
        async with httpx.AsyncClient(timeout=8.0, http2=True) as client:
            async def fetch(pair):
                asset = pair[1].split("15M-", 1)[0][2:]
                try:
                    ka, po = await asyncio.gather(
                        client.get(
                            f"{self.settings.kalshi.api_base.rstrip('/')}/markets/{pair[1]}"
                        ),
                        client.get(
                            "https://polymarket.com/api/crypto/crypto-price",
                            params={
                                "symbol": asset, "eventStartTime": start_iso,
                                "variant": "fifteen", "endDate": end_iso,
                            },
                        ),
                    )
                    ka.raise_for_status()
                    po.raise_for_status()
                    return pair, asset, float(ka.json()["market"]["floor_strike"]), float(
                        po.json()["openPrice"]
                    )
                except (httpx.HTTPError, KeyError, TypeError, ValueError):
                    return pair, asset, None, None

            rows = await asyncio.gather(*(fetch(pair) for pair in pairs))
        ready = 0
        for pair, asset, kalshi_open, pcom_open in rows:
            key = (start, asset)
            previous = self.thresholds.get(key, {})
            self.thresholds[key] = {
                "kalshi_open": kalshi_open or previous.get("kalshi_open"),
                "pcom_open": pcom_open or previous.get("pcom_open"),
                "kalshi_market": pair[1], "pcom_market": pair[3],
            }
            if self.thresholds[key]["kalshi_open"] and self.thresholds[key]["pcom_open"]:
                ready += 1
            if self.store is not None:
                self.store.upsert_crypto_oracle_window(
                    window_start=start, asset=asset, kalshi_market=pair[1],
                    polymarket_com_market=pair[3], kalshi_open=kalshi_open,
                    polymarket_com_open=pcom_open,
                )
        log.info("crypto oracle: thresholds ready for %d/%d current assets", ready, len(rows))

    async def _threshold_loop(self) -> None:
        # ``run`` performs the initial refresh before launching this task.
        last_start = int(time.time() // 900) * 900
        while True:
            start = int(time.time() // 900) * 900
            if start != last_start or any(
                not row.get("kalshi_open") or not row.get("pcom_open")
                for (ws, _), row in self.thresholds.items() if ws == start
            ):
                await self._refresh_thresholds()
                last_start = start
            await asyncio.sleep(2.0 if start + 15 > time.time() else 10.0)

    async def _chainlink_loop(self) -> None:
        import websockets

        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    _RTDS_WS, open_timeout=8.0, ping_interval=None, compression=None,
                ) as socket:
                    await socket.send(json.dumps({
                        "action": "subscribe",
                        "subscriptions": [{
                            "topic": "crypto_prices_chainlink", "type": "*", "filters": "",
                        }],
                    }))
                    backoff = 1.0
                    async for raw in socket:
                        if raw == "PING":
                            await socket.send("PONG")
                            continue
                        try:
                            message = json.loads(raw)
                            payload = message.get("payload") or {}
                            asset = str(payload.get("symbol") or "").split("/", 1)[0].upper()
                            ts = float(payload.get("timestamp") or message.get("timestamp")) / 1000.0
                            self._update_chainlink(asset, float(payload["value"]), ts)
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                            continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Chainlink RTDS disconnected (%s); retrying in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)

    async def _refresh_margin_rest(self) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=8.0, http2=True) as client:
            response = await client.get(
                f"{self.settings.kalshi.api_base.rstrip('/')}/margin/markets"
            )
            response.raise_for_status()
            for market in response.json().get("markets", []):
                ticker = str(market.get("ticker") or "")
                if not ticker.startswith("KX") or not ticker.endswith("PERP"):
                    continue
                asset = ticker[2:-4]
                try:
                    size = float(market["contract_size"])
                    ref = market["reference_price"]
                    spot = float(ref["price"]) / size
                    ts = float(ref["ts_ms"]) / 1000.0
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    continue
                self.contract_sizes[asset] = size
                self._update_cf(asset, spot, ts)

    async def _margin_rest_fallback_loop(self) -> None:
        while True:
            try:
                freshest = max((ts for _, ts in self.cf_ticks.values()), default=0.0)
                if time.time() - freshest > 1.5:
                    await self._refresh_margin_rest()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Kalshi margin REST fallback failed: %s", exc)
            await asyncio.sleep(1.0)

    async def _margin_ws_loop(self) -> None:
        import websockets

        cfg = self.settings.kalshi
        if not cfg.api_key_id:
            log.warning("crypto oracle: no Kalshi key; CF reference using 1Hz REST fallback")
            await asyncio.Future()
        self._private_key = self._private_key or load_private_key(cfg.private_key_path)
        backoff = 1.0
        while True:
            try:
                headers = build_signature_headers(
                    cfg.api_key_id, self._private_key, "GET", _MARGIN_WS_PATH
                )
                async with websockets.connect(
                    _MARGIN_WS, additional_headers=headers, open_timeout=8.0,
                    compression=None, ping_timeout=30.0,
                ) as socket:
                    await socket.send(json.dumps({
                        "id": 1, "cmd": "subscribe", "params": {
                            "channels": ["ticker"], "send_initial_snapshot": True,
                        },
                    }))
                    backoff = 1.0
                    async for raw in socket:
                        try:
                            data = json.loads(raw)
                            if data.get("type") != "ticker":
                                continue
                            msg = data.get("msg") or {}
                            ticker = str(msg.get("market_ticker") or "")
                            asset = ticker[2:-4] if ticker.startswith("KX") and ticker.endswith(
                                "PERP"
                            ) else ""
                            size = self.contract_sizes.get(asset)
                            ref = msg.get("reference_price") or {}
                            if not size:
                                continue
                            self._update_cf(
                                asset, float(ref["price"]) / size,
                                float(ref["ts_ms"]) / 1000.0,
                            )
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                            continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Kalshi margin WS disconnected (%s); retrying in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)

    async def run(self) -> None:
        """Run threshold refresh plus both source feeds until cancelled."""
        await self._refresh_margin_rest()
        await self._refresh_thresholds()
        tasks = [
            asyncio.create_task(self._threshold_loop()),
            asyncio.create_task(self._chainlink_loop()),
            asyncio.create_task(self._margin_ws_loop()),
            asyncio.create_task(self._margin_rest_fallback_loop()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
