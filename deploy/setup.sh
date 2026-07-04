#!/usr/bin/env bash
# One-command Quên deploy for a fresh Ubuntu 22.04/24.04 ECS instance.
#
#   git clone -b claude/quen-memory-agent-spec-p3izss \
#     https://github.com/phamthanhhang208/quen.git /opt/quen
#   sudo bash /opt/quen/deploy/setup.sh
#
# Idempotent: safe to re-run after a git pull. Reads DASHSCOPE_API_KEY from
# /opt/quen/.env if present (live mode); otherwise serves the offline demo.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/quen}"
PORT="${PORT:-80}"

echo "==> apt dependencies (python, node, build tools)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip curl ca-certificates git >/dev/null
if ! command -v node >/dev/null || [ "$(node -e 'console.log(process.versions.node.split(".")[0])' 2>/dev/null || echo 0)" -lt 18 ]; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null
  apt-get install -y -qq nodejs >/dev/null
fi

cd "$REPO_DIR"

echo "==> python venv + package"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e .

echo "==> dashboard build (same-origin API)"
cd dashboard
npm install --no-audit --no-fund --silent
VITE_API_BASE="" npm run build --silent
cd "$REPO_DIR"

echo "==> seed the 80-day demo narrative (offline, deterministic)"
mkdir -p data
[ -f data/demo.db ] || .venv/bin/python scripts/seed_demo.py data/demo.db >/dev/null

echo "==> systemd service on port $PORT"
ENV_LINES="Environment=QUEN_DB_PATH=$REPO_DIR/data/demo.db
Environment=QUEN_VERIFY_REPO=$REPO_DIR/scripts/demo_repo
Environment=QUEN_DASHBOARD_DIST=$REPO_DIR/dashboard/dist
Environment=QUEN_PORT=$PORT"
if [ -f "$REPO_DIR/.env" ] && grep -q "^DASHSCOPE_API_KEY=" "$REPO_DIR/.env"; then
  echo "    .env found -> LIVE mode (Qwen on DashScope)"
  ENV_LINES="$ENV_LINES
EnvironmentFile=$REPO_DIR/.env"
else
  echo "    no .env -> OFFLINE demo mode (QUEN_OFFLINE=1, zero API cost)"
  ENV_LINES="$ENV_LINES
Environment=QUEN_OFFLINE=1"
fi

cat > /etc/systemd/system/quen.service <<UNIT
[Unit]
Description=Quen memory engine (API + dashboard)
After=network.target

[Service]
WorkingDirectory=$REPO_DIR
$ENV_LINES
ExecStart=$REPO_DIR/.venv/bin/quen-api
Restart=on-failure
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now quen
sleep 2
systemctl --no-pager -l status quen | head -8 || true

IP=$(curl -fs --max-time 3 http://100.100.100.200/latest/meta-data/eipv4 \
  || curl -fs --max-time 3 http://100.100.100.200/latest/meta-data/public-ipv4 \
  || hostname -I | awk '{print $1}')
echo
echo "==> done. Dashboard + API:  http://$IP:$PORT/  (health: /vitals)"
echo "    Make sure the ECS security group opens TCP $PORT."
