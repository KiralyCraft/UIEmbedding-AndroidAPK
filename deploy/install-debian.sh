#!/usr/bin/env bash
# Run as root inside the dedicated Debian VM after copying this repository to /opt/uiembeddings.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends mariadb-server python3-venv qemu-guest-agent nftables curl
systemctl enable --now mariadb qemu-guest-agent
id uiembeddings >/dev/null 2>&1 || useradd --system --home-dir /var/lib/uiembeddings --create-home --shell /usr/sbin/nologin uiembeddings
install -d -m 0750 -o root -g uiembeddings /etc/uiembeddings
install -d -m 0700 -o uiembeddings -g uiembeddings /var/lib/uiembeddings
python3 -m venv /opt/uiembeddings/venv
/opt/uiembeddings/venv/bin/pip install -r /opt/uiembeddings/server/requirements.txt
# Database and secrets are created once; subsequent deployments keep the existing credentials.
if [[ ! -e /etc/uiembeddings/server.env ]]; then
    python3 - <<'PY'
import pathlib, secrets, subprocess
password=secrets.token_hex(32)
sql="CREATE DATABASE uiembeddings CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;\nCREATE USER 'uiembeddings'@'127.0.0.1' IDENTIFIED BY '"+password+"';\nGRANT ALL PRIVILEGES ON uiembeddings.* TO 'uiembeddings'@'127.0.0.1';\n"
subprocess.run(['mariadb'], input=sql, text=True, check=True)
p=pathlib.Path('/etc/uiembeddings/server.env')
p.write_text('DATABASE_URL=mysql+pymysql://uiembeddings:'+password+'@127.0.0.1:3306/uiembeddings?charset=utf8mb4\nROOT_PATH=/projects/uiembeddings\nTOKEN_DAYS=90\nPYTHONDONTWRITEBYTECODE=1\n')
p.chmod(0o640)
PY
    chown root:uiembeddings /etc/uiembeddings/server.env
fi
# Keep MariaDB local and bound memory use on the 2 GiB VM.
cat > /etc/mysql/mariadb.conf.d/60-uiembeddings.cnf <<'CNF'
[mariadb]
bind-address = 127.0.0.1
innodb_buffer_pool_size = 256M
max_connections = 40
CNF
systemctl restart mariadb
set -a
source /etc/uiembeddings/server.env
set +a
cd /opt/uiembeddings/server
../venv/bin/python -m collector.cli init-db
../venv/bin/python -m collector.cli register-model /opt/uiembeddings/f6_manifest.json
install -m 0644 /opt/uiembeddings/deploy/uiembeddings.service /etc/systemd/system/uiembeddings.service
systemctl daemon-reload
systemd-analyze verify /etc/systemd/system/uiembeddings.service
nft -c -f /opt/uiembeddings/deploy/uiembeddings.nft
install -m 0644 /opt/uiembeddings/deploy/uiembeddings.nft /etc/nftables.conf
systemctl enable --now nftables
systemctl enable --now uiembeddings
