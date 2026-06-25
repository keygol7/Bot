#!/usr/bin/env bash
# Read-only latency probe to the venue endpoints. Run on a CANDIDATE host before committing a
# region — the hedge (taker) leg fires to Polymarket, so Polymarket's RTT is the latency that
# actually binds the naked window, not Kalshi's. Pure curl; places no orders, needs no creds.
set -uo pipefail

HOSTS=(external-api.kalshi.com api.polymarket.us gateway.polymarket.us)

printf "%-28s  %-22s  %8s  %8s  %8s\n" "host" "resolved" "connect" "tls" "ttfb"
printf '%.0s-' {1..82}; echo
for h in "${HOSTS[@]}"; do
  ip=$(getent hosts "$h" | awk '{print $1; exit}')
  # 3 samples, take the median (middle row sorted by ttfb) to ignore a cold outlier
  line=$(for _ in 1 2 3; do
           curl -o /dev/null -s -w "%{time_connect} %{time_appconnect} %{time_starttransfer}\n" \
                --max-time 8 "https://$h/" 2>/dev/null
         done | sort -k3 -n | sed -n 2p)
  printf "%-28s  %-22s  %8s  %8s  %8s\n" "$h" "${ip:-?}" $line
done

cat <<'EOF'

  connect = raw network round-trip to the server.
    - Colocated in the venue's own cloud region this should be ~0.001-0.003s.
    - From Denver it was ~0.038s to Kalshi (AWS us-east-1). If you still see ~0.03s+ here,
      you are NOT in the venue's region.
  Polymarket fronts on Cloudflare (close edge -> low 'connect'), so judge it by 'ttfb':
  if ttfb stays ~0.08s+ even at low connect, its ORIGIN is far from this box — colocating
  near Kalshi did not help the hedge leg, and the region choice needs a rethink.
EOF
