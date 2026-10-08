#!/usr/bin/env python3
"""Reject placeholder/URL-unsafe database credentials in the deployment .env."""
from pathlib import Path
import re
values = {}
for line in Path('.env').read_text().splitlines():
    stripped = line.strip()
    if stripped and not stripped.startswith('#') and '=' in stripped:
        key, value = stripped.split('=', 1)
        values[key.strip()] = value.strip()
for key in ('MYSQL_PASSWORD', 'MYSQL_ROOT_PASSWORD'):
    if re.fullmatch(r'[0-9a-fA-F]{64}', values.get(key, '')) is None:
        raise SystemExit(f'{key} must be an independently generated 64-character hexadecimal secret')
if values['MYSQL_PASSWORD'] == values['MYSQL_ROOT_PASSWORD']:
    raise SystemExit('Database user and root passwords must differ')
if '*' in values.get('TRUSTED_PROXY_IPS', ''):
    raise SystemExit('Do not trust forwarded headers from arbitrary addresses')
print('Deployment environment checks passed')
