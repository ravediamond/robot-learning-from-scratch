#!/usr/bin/env bash
# Poll the TOPReward instance; when DONE (or the run exited), copy out/, verify, destroy. Logs to fetch.log.
HERE=$(cd "$(dirname "$0")" && pwd); read ID PRICE T0 < "$HERE/.instance"; S=$(cat "$HERE/.ssh")
KEY=$HOME/.ssh/id_ed25519_vast; HOST=${S##* }; PORT=$(echo "$S" | grep -o -- '-p [0-9]*' | cut -d' ' -f2)
mkdir -p "$HERE/out"
while true; do
  st=$(ssh $S "test -f /root/wm/out/DONE && echo done || (test -f /root/wm/EXIT_CODE && echo exited || echo running)" 2>/dev/null | tail -1)
  [ "$st" = done ] || [ "$st" = exited ] && break
  ST=$(vastai show instance "$ID" --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status'))" 2>/dev/null)
  [ "$ST" = running ] || { echo "$(date -Is) instance status $ST" >> "$HERE/fetch.log"; break; }
  sleep 120
done
scp -q -i "$KEY" -o IdentitiesOnly=yes -P "$PORT" "$HOST:/root/wm/out/*" "$HERE/out/" 2>> "$HERE/fetch.log"
scp -q -i "$KEY" -o IdentitiesOnly=yes -P "$PORT" "$HOST:/root/wm/run.log" "$HERE/out/" 2>> "$HERE/fetch.log"
echo "$(date -Is) fetched: $(wc -l < "$HERE/out/ab.jsonl") ab, $(wc -l < "$HERE/out/own.jsonl") own" >> "$HERE/fetch.log"
vastai destroy instance "$ID" -y >> "$HERE/fetch.log" 2>&1
python3 -c "import time;h=(time.time()-$T0)/3600;print(f'$(date -Is) destroyed after {h:.2f} h, ~\${h*$PRICE:.2f}')" >> "$HERE/fetch.log"
