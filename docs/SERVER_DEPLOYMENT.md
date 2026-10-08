# UI Embeddings deployment

The capture service and administrator console are at **https://kiralycraft.com/projects/uiembeddings/**. Use this same address in the Android application's server settings. Signed-in users create participant accounts in **People & devices**; there is no anonymous registration.

## Infrastructure

This follows the native Debian/MariaDB deployment used by BusOSINT, Codex session `01a0f81b-5982-7320-b36f-9df6ee6edaa0` on `kiraly@192.168.5.144`. The existing BusOSINT VM is unchanged.

| Component | Configuration |
| --- | --- |
| Proxmox | `root@192.168.0.2`, R720, VM **206 UIEmbeddings** |
| Guest | Debian 13, 2 host CPU cores, 2 GiB RAM, automatic start |
| Storage | 128 GiB thin root disk and cloud-init volume, both on **black** |
| Network | `vmbr9`, VLAN tag **3**, VMRISK |
| Address | `192.168.2.75/26`, gateway and DNS `192.168.2.65` |
| SSH | `kiraly@192.168.2.75`, existing public key from BusOSINT |
| Application | `/opt/uiembeddings/server`, `/opt/uiembeddings/venv` |
| Service | `uiembeddings.service`, two Uvicorn workers on port 8000 |
| Database | local MariaDB, `uiembeddings` database, `/var/lib/mysql`, localhost only |
| Configuration | `/etc/uiembeddings/server.env`, root:uiembeddings mode 0640 |
| Proxy | `root@192.168.2.67`, `/etc/httpd/conf/extra/e24-projects.ext` |
| Router | `root@192.168.0.1`, DHCP reservation and `A_WEBPROXY_UIEMBEDDINGS_HTTP` |

The application requires and starts after `mariadb.service`. Guest nftables accepts API connections only from `192.168.2.67`, and SSH from the existing LAN/SANDRI management ranges. The proxy and guest are on the same VLAN: their direct traffic does not pass through the router, so the guest rule enforces this restriction. MariaDB has no external listener.

The Apache Location block follows the existing project's ProxyPass / ProxyPassReverse / ProxyPreserveHost / X-Forwarded-Proto convention. It preserves the URL prefix. FastAPI uses `ROOT_PATH=/projects/uiembeddings`, and Uvicorn trusts forwarded headers only from the proxy. See the [FastAPI proxy documentation](https://fastapi.tiangolo.com/advanced/behind-a-proxy/). Both the UI and Android relative `v1/...` endpoints operate below this prefix.

## Administration and devices

The initial administrator is **kiraly**. Its generated initial password is kept outside source control in the deployment operator's private evidence directory and in `/etc/uiembeddings/initial-admin.json` (root-only) on the VM. Change it through the Password action after first sign-in. That action revokes every session for the account, including Android sessions.

The console supports participant creation, password resets, session revocation and account disabling. Administrators cannot be disabled from the UI. Additional administrators can be created through the CLI:

```bash
sudo bash
set -a
source /etc/uiembeddings/server.env
set +a
cd /opt/uiembeddings/server
../venv/bin/python -m collector.cli create-user another-admin --admin
```

Browser sessions last eight hours, use Secure/HttpOnly/SameSite=Strict cookies scoped to the project path, and require CSRF tokens for changes. Participants can also sign in to the web console, view only their own device statistics, and create other participant accounts. Password resets, account/device controls, and other users' recording statistics remain administrator-only. Public pages disclose no participant statistics.

Each device is identified by **(user ID, installation UUID)**, not its model name. A participant can use multiple phones, including identical models. Run ordinals remain separate per user/device/application. Devices register automatically on the first run or calibration upload; reinstalling the Android application creates a new device identity.

The console shows upload receipt times, committed embeddings, total/complete runs, application counts, latest calibration rate, and the latest uploaded preprocessing/inference telemetry for each device. Upload time is not a live recording-status indicator. The sample counters include incomplete runs. Device names are editable. Disabling one device blocks its uploads without affecting its sibling devices or deleting data; the phone may continue recording into its local queue. After re-enabling it, use **Retry upload** in the Android app.

## Reproduction and updates

`deploy/provision-proxmox.sh` records the reviewed creation of VM 206. It refuses an existing VM ID. It expects the official Debian 13 genericcloud image and its `SHA512SUMS` under `/var/lib/vz/template/uiembeddings`, verifies the checksum, copies only BusOSINT's public SSH key, and creates a fresh guest; it does not clone BusOSINT's application or data. Debian images are available from [Debian's cloud image service](https://cloud.debian.org/images/cloud/trixie/latest/).

For a fresh dedicated guest, copy `server/`, `deploy/`, and the validated F6 manifest to `/opt/uiembeddings` (manifest filename `f6_manifest.json`), then run `sudo bash /opt/uiembeddings/deploy/install-debian.sh`. The installer generates database credentials once and installs the native services. Its nftables configuration is for this dedicated VM and replaces that VM's ruleset. `deploy/apache-uiembeddings.ext` and `deploy/openwrt-uiembeddings.conf` record the additions for the existing proxy and router; append only to the appropriate configuration sections, validate, then reload.

For routine code updates, back up the source and database, copy the updated `server/collector` directory, install changed pinned requirements when necessary, and restart `uiembeddings.service`. Verify public health, admin login and a committed upload. Do not reinitialize an existing database as a substitute for a schema migration. The new devices/admin tables in this initial deployment were created on an empty database; adopting this version on an older populated installation requires backfilling device identities and adding the composite indexes before its dashboard totals are used.

Database backup example, run locally on the guest as root:

```bash
install -d -m 0700 /var/backups/uiembeddings
umask 077
mariadb-dump --single-transaction --routines --events uiembeddings \
  | gzip > /var/backups/uiembeddings/collector-$(date +%Y%m%d-%H%M%S).sql.gz
```

The application does not discard committed records automatically. Monitor both guest filesystem usage and the Proxmox thin pool. This deployment has not configured an off-VM backup schedule or established a long-term capacity estimate.

## Validation and recovery

Acceptance on 2026-10-08 covered 47 tests on SQLite and the same 47 on an isolated MariaDB database inside the production guest. The MariaDB suite caught decimal aggregate counts being serialized as strings; the API now explicitly returns integers. Tests cover authentication, CSRF, account administration, same-user multiple devices, device isolation/disable, calibration-only devices, prefix handling, ingestion ordering and retry integrity.

Live checks through public HTTPS used two synthetic phones under one disposable participant: two complete runs, exact vector retrieval, unchanged retries, calibration upload and per-device disabling. Browser checks covered login, navigation, separate device statistics, and responsive desktop/mobile layouts. Synthetic data was removed after validation. A subsequent user-authorized Xperia check signed in to the real account through the Android UI and verified new phone-generated embeddings reaching the server. The pre-existing local-only backlog was not transferred.

Before changing router/proxy configuration, exact backups were saved on their respective hosts with a `.uiembeddings-<timestamp>` suffix, with private copies under ignored `reports/work/server-deployment/`. Apache syntax and OpenWrt nftables validation passed. To withdraw public access, remove only the UIEmbeddings Location block and reload Apache after `httpd -t`; stop `uiembeddings.service` if needed. Preserve the VM disks and database for recovery. Router rollback should remove only the new rule/reservation, preserving any later unrelated changes.

A VM reboot confirmed automatic MariaDB, API, nftables and guest-agent startup, zero application restarts, and preservation of both synthetic device records. The application became HTTP-ready about eleven seconds after its systemd process started. External direct API access from the management workstation timed out, while proxy access and public HTTPS health succeeded. The phone was not rebooted.

The connected Sony Xperia XQ-DQ72 was configured through the installed APK using the public HTTPS address. Its persistent device identity is labeled **Sony Xperia XQ-DQ72** in the console. New account-bound capture data uploads automatically; the pre-existing unbound local queue remains on the device until explicitly transferred.

## Internet-facing login limits

Both `/admin/login` and `/v1/auth/login` share persistent per-client budgets: **30 attempts per 60 seconds** and **100 per 900 seconds**. Successes also consume the IP budget; switching usernames or alternating web and Android endpoints does not reset it. The existing **10 failed attempts per account per 900 seconds** remains, with row-locked password checks for existing accounts to prevent concurrent bypass. A successful sign-in clears that account's failure history. Limits return HTTP 429 with Retry-After; existing authenticated capture uploads do not consume login budgets.

Counter admission is transactional and shared across Uvicorn workers. Only hashed client identifiers, counters and expiry times are stored. Expired counters are cleaned on subsequent login attempts. Database failures fail sign-in closed with a retryable HTTP 503.

The limiter uses `Request.client.host`, after [Uvicorn's trusted-proxy processing](https://www.uvicorn.org/settings/), and never reads a raw forwarding header. Only the web proxy `192.168.2.67` is trusted. [Apache appends its actual client connection address](https://httpd.apache.org/docs/current/mod/mod_proxy.html) to the forwarding chain; forged preceding values cannot override the last untrusted hop. The live public-address test matched the stored limiter key to Apache's connection IP (the workstation's public egress IP, rather than its LAN address or the proxy address).

The additive migration for installations deployed before these limits is:

```bash
# As root, with /etc/uiembeddings/server.env exported and cwd /opt/uiembeddings/server:
../venv/bin/python -m collector.cli upgrade-rate-limits
systemctl restart uiembeddings
```

It only creates `login_rate_limits` if absent. A database snapshot was taken at `/var/backups/uiembeddings/before-rate-limit.sql.gz` before the live migration. Tests cover atomic concurrent admission, shared endpoints/workers, persistence, expiry, longer-window enforcement, spoofed forwarding headers and continued authenticated ingestion. The live HTTPS probe returned 30 HTTP 401 responses followed by HTTP 429 while changing its forged forwarding header on every request. The stored long-window counter survived a service restart, and the short-window limit expired normally.


## Account creation history

Every account created through the web console records an immutable creator user ID and server creation time in `account_creations`, in the same transaction as the new account. Administrators see **Created by** and **Created** for each person; participants see **Accounts you created**, containing only their direct creations. Creating an account does not grant access to its recordings, devices, password controls or sessions. New accounts are always participants. Share the chosen credentials directly; this flow does not send email or generate invitation links.

Participant creators are limited to 20 successful creations per rolling 24 hours, serialized by a database lock across workers. Administrators are exempt. Existing shared login limits, CSRF checks, revocation and disabled-account checks apply to participant web sessions too. Historical accounts with no recorded provenance explicitly show **Not recorded**, rather than inventing a creator or timestamp. CLI bootstrap accounts also have no recorded web creator.

Before deploying this version on an existing database, back up source and database, copy updated code, then run (as root with the environment exported and cwd `/opt/uiembeddings/server`):

```bash
../venv/bin/python -m collector.cli upgrade-account-creations
systemctl restart uiembeddings
```

This additive, repeatable migration creates only the new history table. Existing credentials, sessions and capture records are preserved.

Account-creation update validation: 50 tests pass on SQLite and on isolated MariaDB, including participant access boundaries, creator attribution, nested creations, disabled creators, CSRF, rolling limits, concurrent quota admission and repeatable migration. Live web checks confirm creator attribution and participant-only statistics at desktop and 390-pixel mobile width.
