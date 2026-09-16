#!/usr/bin/env bash
# System install of Neo4j Community 5.26.14 (LTS) + Java 21 on Ubuntu 24.04.
# One sudo run:  cd energy-audit && sudo bash scripts/install_neo4j.sh
# The script is safe to re-run (skips completed steps).
set -euo pipefail

NEO4J_VERSION="5.26.14"
NEO4J_TARBALL="neo4j-community-${NEO4J_VERSION}-unix.tar.gz"
TARBALL_URL="https://dist.neo4j.org/${NEO4J_TARBALL}"
EXPECTED_SHA256="cda95043254225d29f276f0a6beb76b4501fe42d1327471925956f1d6df7af9a"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARBALL_SRC="${TARBALL_SRC:-/tmp/opencode/${NEO4J_TARBALL}}"

log() { printf '\n== %s\n' "$*"; }
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }

# 1. Java 21 JRE
log "Java 21 JRE"
if java -version 2>&1 | grep -q '"21\.'; then
  echo "java 21 already present"
else
  apt-get update -qq
  apt-get install -y -qq openjdk-21-jre-headless
fi
java -version

# 2. neo4j service user
log "neo4j user"
if ! getent group neo4j >/dev/null; then
  groupadd neo4j
fi
if ! id neo4j >/dev/null 2>&1; then
  useradd --system --gid neo4j --shell /usr/sbin/nologin --home-dir /opt/neo4j neo4j
fi

# 3. tarball (use local copy if present, else download + verify sha256)
log "tarball"
T="$TARBALL_SRC"
if [ ! -f "$T" ]; then
  echo "downloading $TARBALL_URL"
  mkdir -p /var/cache/neo4j-install
  curl -fL --retry 3 -o "$T" "$TARBALL_URL"
fi
echo "$EXPECTED_SHA256  $T" | sha256sum -c -
[ "$T" ] && true

# 4. install to /opt (skip if already installed)
NEO4J_HOME="/opt/neo4j-community-${NEO4J_VERSION}"
if [ -d "$NEO4J_HOME" ]; then
  echo "already installed at $NEO4J_HOME"
else
  log "unbundle to /opt"
  tar zxf "$T" -C /opt
fi
ln -sfn "$NEO4J_HOME" /opt/neo4j
chown -R neo4j:neo4j /opt/neo4j

# 5. initial password (from repo .env, or NEO4J_INITIAL_PASSWORD)
log "initial password"
PW="${NEO4J_INITIAL_PASSWORD:-}"
if [ -z "$PW" ] && [ -f "$REPO_ROOT/.env" ]; then
  PW="$(sed -n 's/^NEO4J_PASSWORD=//p' "$REPO_ROOT/.env" | head -1)"
fi
if [ -z "$PW" ]; then
  PW="$(tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c 24)"
  echo "generated NEO4J_PASSWORD=$PW  (no .env found; add it to .env yourself)"
fi
if [ -f /opt/neo4j/data/databases/neo4j/system.db.lock ] || [ -d "$NEO4J_HOME/data/databases/neo4j" ]; then
  echo "data dir already initialized; skipping set-initial-password"
else
  runuser -u neo4j -- /opt/neo4j/bin/neo4j-admin dbms set-initial-password "$PW"
fi

# 6. systemd service
log "systemd service"
cat > /etc/systemd/system/neo4j.service <<'UNIT'
[Unit]
Description=Neo4j Graph Database
After=network-online.target
Wants=network-online.target
[Service]
ExecStart=/opt/neo4j/bin/neo4j console
Restart=on-abnormal
User=neo4j
Group=neo4j
Environment="NEO4J_CONF=/opt/neo4j/conf" "NEO4J_HOME=/opt/neo4j"
LimitNOFILE=60000
TimeoutSec=120
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable neo4j
systemctl restart neo4j

# 7. wait for bolt :7687 to accept connections (up to 90s)
log "waiting for bolt :7687"
for i in $(seq 1 45); do
  if (exec 3<>/dev/tcp/127.0.0.1/7687) 2>/dev/null; then
    exec 3>&- 3<&- || true
    echo "bolt :7687 is UP"
    break
  fi
  sleep 2
  [ "$i" -eq 45 ] && { systemctl status neo4j --no-pager -l | tail -20; echo "timed out waiting :7687"; exit 1; }
done

log "done"
echo "Next (as your normal user, no sudo):"
echo "  cd $REPO_ROOT && python3 scripts/check_local_stack.py"
