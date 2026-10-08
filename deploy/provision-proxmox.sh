#!/usr/bin/env bash
set -euo pipefail
# Reviewed against BusOSINT VM 205 and WebProxy VM 107.
test ! -e /etc/pve/qemu-server/206.conf
cd /var/lib/vz/template/uiembeddings
sha512sum --check --ignore-missing SHA512SUMS
python3 - <<'PY'
from pathlib import Path
from urllib.parse import unquote
config=Path('/etc/pve/qemu-server/205.conf').read_text()
key=next(line.split(': ',1)[1] for line in config.splitlines() if line.startswith('sshkeys:'))
Path('authorized.pub').write_text(unquote(key))
PY
qm create 206 --name UIEmbeddings --description 'Debian 13 UI embedding collector and administration with local MariaDB' --memory 2048 --balloon 0 --cores 2 --cpu host --ostype l26 --scsihw virtio-scsi-pci --net0 virtio,bridge=vmbr9,tag=3 --serial0 socket --vga serial0 --agent enabled=1 --onboot 1
qm importdisk 206 debian-13-genericcloud-amd64.qcow2 black --format raw
qm set 206 --scsi0 black:vm-206-disk-0,discard=on,ssd=1 --ide2 black:cloudinit --boot order=scsi0 --ciuser kiraly --sshkeys authorized.pub --ipconfig0 ip=192.168.2.75/26,gw=192.168.2.65 --nameserver 192.168.2.65 --searchdomain lan
qm resize 206 scsi0 128G
qm start 206
qm config 206 | grep -v sshkeys
