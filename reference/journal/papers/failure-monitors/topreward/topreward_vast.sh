#!/usr/bin/env bash
# TOPREWARD_PROTOCOL.md on vast.ai (same safety as robometer/vast_g9a_ab.sh: on-demand offer, bootstrap.sh hard deadline
# + self-destroy, a local destroy timer; the caller destroys the instance right after the verified fetch).
#   topreward_vast.sh [--dry-run] [--cap USD]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); VAST=$HOME/workspace/road-to-robotics/projects/vla-jepa-world-model/wm_train/vast
MAX_MIN=180; GRACE=20; DRY=0; CAP=5; KEY=$HOME/.ssh/id_ed25519_vast; DISK=100; GPU_RAM=24; DOWNLOAD_GB=45
while [ $# -gt 0 ]; do case $1 in --dry-run) DRY=1; shift;; --cap) CAP=$2; shift 2;; *) echo "unknown $1" >&2; exit 2;; esac; done
QUERY="num_gpus=1 gpu_ram>=$GPU_RAM compute_cap>=800 cpu_ram>=32 cpu_cores_effective>=8 reliability>0.98 disk_space>=$DISK inet_down>=500 cuda_vers>=12.8 verified=true rentable=true"
read -r OFFER PRICE GPU JOBCAP <<< "$(vastai search offers "$QUERY" -d --storage $DISK -o dph_total --raw | python3 -c "
import json, sys
offers = [o for o in json.load(sys.stdin) if not o['gpu_name'].startswith('CMP') and not str(o.get('geolocation', '')).endswith('CN')]
cost = lambda o: ($MAX_MIN + $GRACE) / 60 * o['dph_total'] + $DOWNLOAD_GB * o.get('inet_down_cost', 0)
offers.sort(key=cost); o = offers[0] if offers else None
print(o['id'], round(o['dph_total'], 4), o['gpu_name'].replace(' ', '_'), round(cost(o), 3)) if o else print('none 0 none 0')")"
[ "$OFFER" != none ] || { echo "no matching offer" >&2; exit 1; }
echo "offer $OFFER: $GPU at \$$PRICE/h | <= $MAX_MIN + $GRACE min + ${DOWNLOAD_GB} GB -> worst case ~\$$JOBCAP (cap \$$CAP)"
python3 -c "import sys; sys.exit(0 if $JOBCAP <= $CAP else 1)" || { echo "over the cap: not launching" >&2; exit 1; }
[ $DRY = 1 ] && exit 0
ID=$(vastai create instance "$OFFER" --image nvidia/cuda:12.8.1-base-ubuntu24.04 --disk $DISK --ssh --direct --cancel-unavail \
  --label topreward --env "-e WM_MAX_MIN=$MAX_MIN -e WM_GRACE_MIN=$GRACE" --onstart-cmd "$(cat "$VAST/bootstrap.sh")" --raw |
  python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('success'), d; print(d['new_contract'])")
echo "instance $ID"; echo "$ID $PRICE $(date +%s)" > "$HERE/.instance"
nohup bash -c "sleep $(( (MAX_MIN + GRACE + 10) * 60 )); vastai destroy instance $ID -y" > /dev/null 2>&1 &
vastai attach ssh "$ID" "$(cat "$KEY.pub")" > /dev/null
for _ in $(seq 120); do ST=$(vastai show instance "$ID" --raw | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status'))" || true); [ "$ST" = running ] && break; sleep 10; done
[ "$ST" = running ] || { echo "not running after 20 min: destroying" >&2; vastai destroy instance "$ID" -y; exit 1; }
SSH_URL=$(vastai ssh-url "$ID"); HOST=${SSH_URL#ssh://}; PORT=${HOST##*:}; HOST=${HOST%:*}
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT"
OK=0; for _ in $(seq 30); do ssh $SSHO "$HOST" true 2> /dev/null && { OK=1; break; }; sleep 10; done
[ $OK = 1 ] || { echo "ssh never worked: destroying" >&2; vastai destroy instance "$ID" -y; exit 1; }
scp -q -i "$KEY" -o IdentitiesOnly=yes -P "$PORT" "$HERE/topreward_score.py" "$HOST:/root/wm/"
scp -q -i "$KEY" -o IdentitiesOnly=yes -P "$PORT" "$HERE/remote_topreward.sh" "$HOST:/root/wm/run_remote.sh"
ssh $SSHO "$HOST" "touch /root/wm/READY"
echo "$SSHO $HOST" > "$HERE/.ssh"; echo "started"
