#!/usr/bin/env python3
"""Print download and server status without changing either service."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import urllib.request
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8021', help='Base URL of the running server.')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parent
model = ROOT / 'model'
cache = model / '.cache/huggingface/download'
completed = sum(p.stat().st_size for p in model.rglob('*') if p.is_file() and '.cache' not in p.relative_to(model).parts)
completed_hashes = set()
for meta in cache.rglob('*.metadata'):
    lines = meta.read_text().splitlines()
    final = model / meta.relative_to(cache).with_suffix('')
    if final.is_file() and len(lines) >= 2:
        completed_hashes.add(lines[1])
partials = {}
stale = 0
for part in cache.rglob('*.incomplete'):
    hashes = re.findall(r'[0-9a-f]{64}', part.name)
    if not hashes:
        continue
    key = hashes[0]
    if key in completed_hashes:
        stale += part.stat().st_size
    elif key not in partials or part.stat().st_mtime_ns > partials[key].stat().st_mtime_ns:
        partials[key] = part
stored = completed + sum(p.stat().st_blocks * 512 for p in partials.values())
print(f'Model progress: {stored / 1024**3:.2f} GiB / approximately 30.86 GiB')
if stale:
    print(f'Excluded stale partials: {stale / 1024**3:.2f} GiB (clean up after verification)')
print('Checksum receipt:', 'present' if (ROOT / 'verified.json').exists() else 'not yet complete')
try:
    result = subprocess.run(['systemctl', '--user', 'show', 'qwen21-server', '-p', 'ActiveState', '-p', 'SubState', '-p', 'ExecMainStatus'], capture_output=True, text=True)
    print('User service: ' + (result.stdout.strip().replace('\n', ', ') if result.returncode == 0 else 'unavailable'))
except FileNotFoundError:
    print('User systemd unavailable; checking the HTTP endpoint only.')
try:
    with urllib.request.urlopen(args.url.rstrip('/') + '/health', timeout=2) as r:
        print('Server:', json.load(r))
except OSError:
    print(f'Server health: not ready at {args.url}')
