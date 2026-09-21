#!/usr/bin/env python3
"""Preflight, obtain consent for RAM handoff, start a persistent user service."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv/bin/python'
UNIT = 'qwen21-server.service'
SERVER = re.compile(
    r'(?:^|[\s/])(?:sglang(?:\.launch_server|::\w+)?|vllm(?:\.[\w.]+)?|'
    r'llama-server|llama\.server|text-generation-launcher|aphrodite(?:\.[\w.]+)?)(?=\s|$)|'
    r'(?:^|[\s/])ollama\s+(?:serve|runner)(?=\s|$)'
)


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError('Cannot determine available RAM.')


def identity(pid):
    # Linux start time distinguishes a process from a later reuse of its PID.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def discover():
    containers, hosts, container_pids = [], [], set()
    if shutil.which('docker'):
        # Fail closed when Docker exists but inspection is unavailable.
        ids = run('docker', 'ps', '-q').split()
        for item in json.loads(run('docker', 'inspect', *ids)) if ids else []:
            commands = []
            for row in run('docker', 'top', item['Id'], '-eo', 'pid,args').splitlines()[1:]:
                pid, cmd = row.strip().split(None, 1)
                container_pids.add(int(pid))
                commands.append(cmd)
            if any(SERVER.search(cmd) for cmd in commands):
                containers.append({'id': item['Id'], 'name': item['Name'].lstrip('/')})
    for row in run('ps', '-eo', 'pid=,args=').splitlines():
        pid, cmd = row.strip().split(None, 1)
        pid = int(pid)
        if pid in container_pids or pid == os.getpid() or not SERVER.search(cmd):
            continue
        if Path(cmd.split()[0]).name in ('bash', 'sh', 'zsh', 'sudo', 'timeout'):
            continue
        try:
            hosts.append({'pid': pid, 'command': cmd, 'identity': identity(pid)})
        except FileNotFoundError:
            pass
    return containers, hosts


def confirm():
    while True:
        try:
            answer = input('Stop these inference servers and start Qwen Image 2.1? [Y/n] ').strip().lower()
        except EOFError:
            return False
        if answer in ('', 'y', 'yes'):
            return True
        if answer in ('n', 'no'):
            return False
        print('Please enter y or n.')


def preflight(host, port, *, foreground=False):
    if not PYTHON.is_file():
        raise RuntimeError('Project environment is missing. Run uv sync --locked first.')
    receipt_path = ROOT / 'verified.json'
    if not receipt_path.exists():
        raise RuntimeError('Verified weights are required. Run uv run --no-sync download.py first.')
    receipt = json.loads(receipt_path.read_text())
    from download import REPO, REVISION
    if receipt['repo'] != REPO or receipt['revision'] != REVISION or not receipt['files']:
        raise RuntimeError('Verification receipt is not for the pinned model.')
    for f in receipt['files']:
        p = ROOT / 'model' / f['path']
        st = p.stat()
        if st.st_size != f['size'] or st.st_mtime_ns != f['mtime_ns']:
            raise RuntimeError(f'Model changed since verification: {p}; rerun download.py.')
    with socket.socket() as probe:
        probe.bind((host, port))
    if not foreground:
        if not all(shutil.which(command) for command in ('systemctl', 'systemd-run')):
            raise RuntimeError('User systemd is unavailable. Use --foreground to run in this terminal.')
        try:
            run('systemctl', '--user', 'show-environment')
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('No usable user systemd session. Use --foreground.') from exc
    subprocess.run([str(PYTHON), '-c',
        'import torch; from diffusers import QwenImage21Pipeline; import server; '
        'assert torch.cuda.is_available(), "CUDA unavailable"; '
        'x=torch.ones((8,8),device="cuda",dtype=torch.bfloat16); '
        'assert (x @ x).sum().item() == 512; '
        'print("CUDA/BF16 and pipeline imports passed:",torch.cuda.get_device_name())'],
        cwd=ROOT, check=True)


def stop_targets(containers, hosts):
    if containers:
        subprocess.run(['docker', 'stop', '--timeout', '60', *[c['id'] for c in containers]], check=True)
    for p in hosts:
        try:
            if identity(p['pid']) == p['identity']:
                os.kill(p['pid'], signal.SIGTERM)
        except FileNotFoundError:
            pass
    deadline = time.monotonic() + 75
    while True:
        remaining = discover()
        if not any(remaining):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Inference processes remain or restarted. Stop their supervisor and retry.')
        time.sleep(2)


def ready(url):
    try:
        with urllib.request.urlopen(url + '/health', timeout=2) as r:
            data = json.load(r)
        return data.get('status') == 'ready' and data.get('model') == 'Qwen/Qwen-Image-2.1'
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8021)
    parser.add_argument('--min-free-gib', type=float, default=56,
                        help='Conservative free shared RAM budget before loading (default: 56 GiB).')
    parser.add_argument('--check', action='store_true', help='Preview RAM and inference processes; no changes.')
    parser.add_argument('--foreground', action='store_true',
                        help='Run in this terminal without requiring user systemd; Ctrl-C stops the server.')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.min_free_gib < 40:
        parser.error('Port must be 1..65535 and RAM budget at least 40 GiB.')
    url = f'http://{("127.0.0.1" if args.host == "0.0.0.0" else args.host)}:{args.port}'
    active = (not args.foreground and shutil.which('systemctl') is not None and
              subprocess.run(['systemctl', '--user', 'is-active', '--quiet', UNIT]).returncode == 0)
    if active and not args.check:
        working_directory = run('systemctl', '--user', 'show', UNIT, '--property=WorkingDirectory', '--value')
        if not working_directory or Path(working_directory).resolve() != ROOT:
            raise RuntimeError(f'{UNIT} belongs to another checkout. Use --foreground on a free port, '
                               'or stop the existing service before starting this checkout.')
        print(f'{UNIT} is already running. Ready: {ready(url)}. Logs: journalctl --user -u {UNIT} -f')
        return 0 if ready(url) else 1
    free = available_gib()
    print(f'Available RAM: {free:.1f} GiB; startup budget: {args.min_free_gib:.1f} GiB', flush=True)
    # Docker access is needed only for a resource handoff or an explicit preview.
    containers, hosts = discover() if args.check or free < args.min_free_gib else ([], [])
    for c in containers:
        print(f'  Docker: {c["name"]} ({c["id"][:12]})')
    for p in hosts:
        print(f'  PID {p["pid"]}: {p["command"]}')
    if args.check:
        print('Preview only. No processes stopped.')
        return 0
    preflight(args.host, args.port, foreground=args.foreground)
    if free < args.min_free_gib:
        if not (containers or hosts):
            raise RuntimeError('Insufficient RAM, and no recognized inference servers to offer to stop.')
        if not confirm():
            print('Cancelled. No processes stopped; Qwen was not started.')
            return 0
        stop_targets(containers, hosts)
        deadline = time.monotonic() + 30
        while available_gib() < args.min_free_gib and time.monotonic() < deadline:
            time.sleep(2)
    free = available_gib()
    if free < args.min_free_gib:
        raise RuntimeError(f'Only {free:.1f} GiB available; refusing to start.')
    print(f'Starting with {free:.1f} GiB available. Loading weights can take several minutes.', flush=True)
    command = [str(PYTHON), '-m', 'uvicorn', 'server:app', '--host', args.host,
               '--port', str(args.port), '--workers', '1']
    if args.foreground:
        return subprocess.run(command, cwd=ROOT, env={
            **os.environ, 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        }).returncode
    subprocess.run(['systemctl', '--user', 'reset-failed', UNIT], capture_output=True)
    subprocess.run([
        'systemd-run', '--user', '--unit=' + UNIT, '--collect',
        '--property=WorkingDirectory=' + str(ROOT), '--property=TimeoutStopSec=120',
        '--setenv=HF_HUB_OFFLINE=1', '--setenv=TOKENIZERS_PARALLELISM=false',
        *command,
    ], check=True)
    deadline = time.monotonic() + 900
    last_report = 0
    while time.monotonic() < deadline:
        if ready(url):
            print(f'Ready: {url}\nAPI docs: {url}/docs\nStop: systemctl --user stop {UNIT}')
            print('Server continues running in the background. This launcher now exits.')
            return 0
        if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', UNIT]).returncode != 0:
            raise RuntimeError(f'Server exited. Inspect: journalctl --user -u {UNIT} -n 100')
        if time.monotonic() - last_report > 30:
            print(f'Waiting for weights to load... logs: journalctl --user -u {UNIT} -f', flush=True)
            last_report = time.monotonic()
        time.sleep(2)
    raise RuntimeError(f'Readiness timed out; service may still be loading. Inspect journalctl --user -u {UNIT}.')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nInterrupted. If startup already began, inspect systemctl --user status ' + UNIT, file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Startup failed: {exc}', file=sys.stderr)
        sys.exit(1)
