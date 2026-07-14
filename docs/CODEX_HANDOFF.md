# Codex Handoff: Multi-Region Prediction-Arbitrage Bot

Last verified: 2026-07-14 UTC

This document is deliberately secret-free. Never add values from `.env`, wallet keys, venue API
credentials, private-key files, or authenticated request headers.

## Objective

Build and validate a profitable cross-venue prediction-market scanner, with particular focus on
crypto contracts between Kalshi and Polymarket.com, while retaining Kalshi and Polymarket US. The
current phase is evidence gathering in read-only shadow mode. Polymarket.com is not funded.

The immediate infrastructure objective is to retire the original Ubuntu EC2 instance after Codex
context, operational knowledge, and rollback data have moved to the US ARM node.

## Topology

| Role | Region | Private IP | Platform | Size | Repository |
|---|---|---|---|---|---|
| Original host, retiring | `us-east-2` | `172.31.1.77` | Ubuntu x86_64 | `c7i-flex.large` | `/home/ubuntu/Bot` |
| Canonical US node | `us-east-2` | `172.31.47.73` | Amazon Linux 2023 ARM64 | `c6g.large` | `/home/ec2-user/Bot` |
| EU latency node | `eu-west-1` | `10.20.1.102` | Amazon Linux 2023 ARM64 | `c6g.large` | `/home/ec2-user/Bot` |

- The EU VPC is `10.20.0.0/16`, with public subnet `10.20.1.0/24`.
- The US VPC is the existing Ohio VPC. Verify its CIDR in AWS before editing routes; it is expected
  to be the default-style `172.31.0.0/16` range.
- Inter-region VPC peering is active and private connectivity has been verified.
- Security groups should restrict coordination ports to the individual peer private IPs or the
  narrowest required CIDR. Do not expose a coordination port publicly.

## Repository state

At the 2026-07-14 audit, both ARM nodes were on:

```text
branch: migration/arm-multiregion-20260713
commit: 4edcb68
working tree: clean
```

Both ARM virtual environments reported:

```text
Python 3.12.13
architecture: aarch64
pip check: no broken requirements
numpy, cryptography, py_clob_client_v2: imports successful
```

The old x86 `.venv` must never be copied to ARM. Rebuild virtual environments from `pyproject.toml`.

## US node state

The persistent unit `/etc/systemd/system/bot-pcom-shadow.service` was enabled and running as
`ec2-user`. It uses:

```bash
/home/ec2-user/Bot/.venv/bin/python -m bot.dryrun \
  --shadow-stream \
  --shadow-venue polymarket_com \
  --interval 15 \
  --limit 0 \
  --close-within-days 0 \
  --min-edge 0
```

Verified characteristics:

- `Restart=always`.
- Approximately 1 GiB service memory and about 40% instantaneous CPU during the audit.
- Root filesystem: 30 GiB XFS, about 13% used.
- Chrony synchronized to Amazon Time Sync Service.
- `.env` and all private-key files had mode `600`.
- `data/bot.db` was approximately 896 MiB and actively using WAL mode.
- Database files were mode `644`; consider service `UMask=0077` and mode `600` as a hardening task.
- The audit performed read-only schema/data-version queries, not a full integrity check while the
  writer was active.

Useful commands:

```bash
sudo systemctl status bot-pcom-shadow.service
sudo journalctl -u bot-pcom-shadow.service --since '30 minutes ago'
cd /home/ec2-user/Bot
.venv/bin/python -m bot.dryrun --edge-report --hours 1
```

## EU node state

At the audit, EU had:

- The correct repository and clean Git state.
- A working Python 3.12 ARM virtual environment and venue dependencies.
- No `.env`.
- No database.
- No bot service or bot process.

This is intentionally incomplete. The next deployment should create a public/read-only EU
configuration, a fresh local SQLite database, and a distinct service such as
`bot-pcom-eu-shadow.service`. Do not copy the canonical US database for simultaneous use, and do
not copy funded/private execution credentials during the read-only phase.

## Verified network and latency results

Twelve-request REST measurements on 2026-07-14:

| Endpoint | Ohio average TTFB | Ireland average TTFB |
|---|---:|---:|
| `https://clob.polymarket.com/time` | 129.51 ms | 37.12 ms |
| Kalshi exchange status | 28.12 ms | 25.44 ms |

The Ireland node improved the measured Polymarket REST TTFB by about 92 ms. The Kalshi status
endpoint is CDN-backed and is not proof of order-entry or WebSocket latency.

A temporary private HTTP listener verified the peering path:

```text
US -> EU TCP connect average: 78.71 ms
US -> EU HTTP TTFB average: 158.29 ms
samples: 10
failures: 0
```

The temporary listener was stopped after the test.

## Recent shadow evidence

During a one-hour US report immediately after migration:

```text
edge observations: 76,497
read-only WebSocket edge episodes: 3,523
locked SUCCESS: 0
```

WebSocket logs showed current crypto pairs for BTC, ETH, SOL, XRP, DOGE, BNB, and HYPE. Example
episodes included eligible, below-divergence-floor, crypto-oracle-divergence, rules-pending, and
fat-edge-unproven classifications.

These observations confirm that feeds and pairing are active; they do not yet prove executable,
profitable two-leg arbitrage. The current major blockers remain rules equivalence, oracle-path
proof, false-match rejection, persistence, fill sequencing, and cross-region coordination.

## Critical architectural constraint

VPC peering only connects the hosts. The application currently has local SQLite and local in-memory
execution/locking state. Never enable independent live execution on both servers.

A safe future live design needs:

1. Exactly one leader/coordinator.
2. EU ownership of Polymarket WebSocket/order/private streams.
3. US ownership of Kalshi WebSocket/order/private streams.
4. A persistent authenticated mTLS channel with sequence numbers.
5. Unique durable `trade_id` values and idempotent commands.
6. Venue-local book revalidation immediately before submission.
7. Confirmed first-leg fill before issuing the hedge command.
8. Hard local and global exposure caps.
9. Durable execution journaling and reconnect reconciliation.
10. Split-brain and duplicate-order protection.

## Next safe steps

1. Install Codex on the US node and authenticate there independently; never copy `auth.json`.
2. Commit this handoff and `AGENTS.md`, push, and pull them on the US node.
3. Optionally copy the current local Codex transcript and resume thread
   `019f59b8-bbfd-7fc3-9b8e-da1149d50325`; treat this document as the durable fallback.
4. Build the EU public/read-only `.env` without funded execution credentials.
5. Create the EU local database and persistent read-only service.
6. Measure WebSocket feed lag and episode detection on both nodes over the same time window.
7. Define and implement the inter-node coordinator protocol before any live deployment.
8. Stop the old host, retain a rollback snapshot for several days, then terminate it only after the
   US Codex installation and service operations are verified.

## Codex continuity

The old Codex local session directory is under `/home/ubuntu/.codex/sessions`. Do not migrate the
entire `.codex` directory: it contains authentication, caches, sockets, shell snapshots, and
host-specific internal state. Authenticate fresh on US and copy only deliberately reviewed session
artifacts. If resume fails, start Codex in `/home/ec2-user/Bot` and instruct it to read `AGENTS.md`
and this document before acting.

