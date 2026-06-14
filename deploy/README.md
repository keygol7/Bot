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
- Edit `User=` and the two `/home/keyahn/Bot` paths in the unit file if your username/path differ.

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
- `Restart=on-failure` only — after **any crash during live trading**, run
  `systemctl status arbbot-stream` and **manually check your open positions on both
  venues** before restarting. The bot does not reconcile pre-existing positions on
  startup.
- The kill switch is in-memory; a restart resets it. If it tripped (see the logs for
  `HALT`), investigate the cause before re-enabling.
- Review results: `sqlite3 ~/Bot/data/bot.db "SELECT * FROM fills; SELECT * FROM pnl;"`
