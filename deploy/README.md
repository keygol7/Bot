# Running the bot long-term (systemd)

Two service units are provided. Pick **one** based on what you want:

| Unit | What it does | Money at risk |
|---|---|---|
| `arbbot.service` | Read-only monitor (`--embed --llm`, **no orders**) | None |
| `arbbot-stream.service` | **Streaming live** (`--stream`, **real orders**) | **Real** |

> ⚠️ **There is no Polymarket US demo.** `--stream` always places *real* Polymarket
> orders. For cross-venue arb to be coherent, **both** venues must be on production in
> `.env` (Kalshi prod, not demo). A demo-Kalshi + real-Polymarket mix leaves you with a
> real naked position. Start with `arbbot.service` (read-only) until you're ready.

## Prerequisites
- Repo cloned at `/home/<user>/Bot`, venv created, deps installed (`pip install -e ".[venues,dev]"`).
- `.env` filled in (LLM URL, venue keys, `RISK_*` caps). `python -m bot.dryrun --check-llm` returns OK.
- The units ship configured for `User=ubuntu` and `/home/ubuntu/Bot`; edit `User=` and the two `/home/ubuntu/Bot` paths in the unit file if your username/path differ.

## Install (read-only monitor)
```bash
sudo cp deploy/arbbot.service /etc/systemd/system/arbbot.service
sudo systemctl daemon-reload
sudo systemctl enable --now arbbot          # start now + on every boot
systemctl status arbbot
journalctl -u arbbot -f                      # live logs (Ctrl-C stops watching)
```

## Install (streaming live — real money)
Only after you've validated and accepted real-money trading on BOTH venues:
```bash
sudo cp deploy/arbbot-stream.service /etc/systemd/system/arbbot-stream.service
sudo systemctl daemon-reload
sudo systemctl enable --now arbbot-stream
journalctl -u arbbot-stream -f
```

## Manage
```bash
sudo systemctl stop arbbot          # stop
sudo systemctl restart arbbot       # restart (picks up .env / code changes)
sudo systemctl disable arbbot       # don't start on boot
systemctl status arbbot             # is it running? recent logs
journalctl -u arbbot --since "1 hour ago"
```
After `git pull` (code change) or editing `.env`: `sudo systemctl restart arbbot`.
After editing the unit file itself: `sudo systemctl daemon-reload && sudo systemctl restart arbbot`.

## Logs & disk
Logs go to **journald** (no `soak.log` file to rotate). Cap journald disk use if needed:
```bash
sudo journalctl --vacuum-time=14d     # keep ~2 weeks
# or set SystemMaxUse=500M in /etc/systemd/journal.conf
```

## Safety reminders for the streaming (live) unit
- **Startup reconciliation guard:** on boot the stream reads each trading venue's
  balance and positions; it refuses to trade (trips the kill switch, then aborts)
  unless every venue is funded (≥ `STARTUP_MIN_BALANCE`) and **flat** — no held
  positions and no resting orders. Set `STARTUP_ALLOW_POSITIONS=true` only if you
  deliberately want to run with pre-existing positions. After a crash, just restart:
  the guard catches a leftover/abandoned leg rather than trading on top of it.
  - The guard logs the snapshot it read (`startup: <venue> balance $X`). On the first
    live run, confirm those balances match your accounts — the Polymarket US
    portfolio-payload field names are parsed defensively and an unrecognized shape
    fails closed (the guard aborts) rather than assuming you're flat.
- `Restart=on-failure` only — after **any crash during live trading**, run
  `systemctl status arbbot-stream` and check the startup-guard output (or your venue
  UIs) before relying on it again.
- The kill switch is in-memory; a restart resets it. If it tripped (see the logs for
  `HALT` or `STARTUP GUARD FAILED`), investigate the cause before re-enabling.
- Review results: `sqlite3 ~/Bot/data/bot.db "SELECT * FROM fills; SELECT * FROM pnl;"`
