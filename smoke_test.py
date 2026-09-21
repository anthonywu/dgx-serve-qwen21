#!/usr/bin/env python3
"""Exercise health, model listing and two real requests without reloading weights."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8021')
args = parser.parse_args()
root = Path(__file__).resolve().parent
(root / 'outputs').mkdir(exist_ok=True)


def get(path):
    with urllib.request.urlopen(args.url + path, timeout=10) as r:
        return json.load(r)


before = get('/health')
assert before['status'] == 'ready'
assert get('/v1/models')['data'][0]['id'] == 'Qwen/Qwen-Image-2.1'
for seed in (42, 43):
    body = json.dumps({'prompt': 'A red ceramic teapot on a plain wooden table, studio photograph',
                       'width': 512, 'height': 512, 'steps': 2, 'seed': seed}).encode()
    request = urllib.request.Request(args.url + '/generate', data=body,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=1800) as response:
        assert response.headers.get_content_type() == 'image/png'
        path = root / 'outputs' / f'smoke-{seed}.png'
        path.write_bytes(response.read())
    with Image.open(path) as image:
        image.load()
        assert image.size == (512, 512), image.size
    after = get('/health')
    assert before['loaded_at'] == after['loaded_at'], 'Model was reloaded between requests!'
    print(f'PASS: {path.name}, 512x512, {time.monotonic() - started:.1f}s; model still resident.', flush=True)
print('LIVE SMOKE TEST PASSED: two generations using the same loaded model.')
