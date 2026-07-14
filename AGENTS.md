# Prediction-Arbitrage Bot Guidance

## Mission and current deployment

- This repository operates a read-only prediction-market arbitrage scanner covering Kalshi,
  Polymarket US, and Polymarket.com.
- The canonical control node is the Amazon Linux ARM server in `us-east-2` at private address
  `172.31.47.73`. Its repository path is `/home/ec2-user/Bot`.
- The latency node is the Amazon Linux ARM server in `eu-west-1` at private address
  `10.20.1.102`. Its repository path is `/home/ec2-user/Bot`.
- The original Ubuntu host at `172.31.1.77` and `/home/ubuntu/Bot` is temporary and is being
  retired after Codex context and rollback data are transferred.
- Read `docs/CODEX_HANDOFF.md` before changing deployment, service, database, matching, risk, or
  venue code.

## Safety boundaries

- Keep all processes in read-only or shadow mode unless the user explicitly authorizes live
  trading in the current conversation.
- Never run independent live execution processes on both nodes. The current implementation has
  local SQLite and in-memory execution state and does not provide cross-region leader election,
  idempotent commands, or split-brain protection.
- Never display, commit, copy into documentation, or log values from `.env`, private keys, wallet
  credentials, API secrets, passphrases, or authenticated headers.
- Do not commit `.env`, `secrets/`, `.venv/`, `data/`, SQLite WAL/SHM files, PEM files, or private
  keys.
- Do not share a writable SQLite database between hosts or mount it over NFS/EFS. Each active
  collector must use its own local database.
- Before transferring the canonical database, stop its writer and run
  `PRAGMA wal_checkpoint(TRUNCATE);`. Never copy only `bot.db` while its WAL writer is active.
- Preserve user changes and unrelated work. Inspect `git status --short` before editing.
- Treat AWS mutations, credential changes, funding, and order placement as explicit-authority
  operations.

## Host roles

### US control node

- Owns the canonical repository, historical database, credentials, and persistent shadow service.
- Service: `bot-pcom-shadow.service`.
- Service command:

  ```bash
  /home/ec2-user/Bot/.venv/bin/python -m bot.dryrun \
    --shadow-stream --shadow-venue polymarket_com \
    --interval 15 --limit 0 --close-within-days 0 --min-edge 0
  ```

- Use this node for Codex and administrative work. Run heavy interactive work at reduced priority
  when the collector is active, for example `nice -n 10 codex`.

### EU latency node

- Intended for Polymarket.com public WebSocket collection and future venue-local order handling.
- Keep it minimal to reduce CPU, disk, and network jitter.
- Until a coordinator exists, it must remain read-only, use a fresh local database, and must not
  receive funded trading credentials.

## Development and verification

- Amazon Linux nodes use Python 3.12 explicitly; do not change the AL2023 system `python3` symlink.
- Create environments with `python3.12 -m venv .venv` and install with
  `.venv/bin/pip install -e '.[venues,dev]'`.
- Run `.venv/bin/pip check` after dependency changes.
- Run `.venv/bin/pytest -q` after code changes. Add focused tests for money, matching, resolution,
  timing, order-book, or execution changes.
- Check services with `sudo systemctl status <unit>` and logs with
  `sudo journalctl -u <unit> --since '30 minutes ago'`.
- Do not restart a healthy collector merely to inspect it. Explain and verify service-impacting
  changes.
- Prefer deterministic market matching and explicit rules verification over semantic guesses for
  execution eligibility.

## Deployment invariants

- VPC peering provides network connectivity only; it does not make the application distributed or
  execution atomic.
- A future live design requires one leader/coordinator, durable trade IDs, idempotent commands,
  mTLS, local book revalidation, confirmed first-leg fills, hard exposure caps, reconciliation, and
  split-brain protection.
- Polymarket and Kalshi contract equivalence must include asset, threshold, reference source,
  direction, time window, timezone, early-close behavior, and resolution rules.
- Large apparent edges are observations, not proof of executable arbitrage. Validate freshness,
  depth, fees, rules, oracle path, and achievable two-leg timing before changing eligibility.

