#!/usr/bin/env python3
"""Generate sample images and update the served gallery after each."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'samples'
OUT.mkdir(exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8021', help='Base URL of the running server.')
args = parser.parse_args()
SAMPLES = [
    ('alpine-morning', 'Alpine morning', 2101,
     'A photorealistic alpine lake in the Pacific Northwest at sunrise, a small red canoe near a weathered wooden dock, mist drifting over still turquoise water, towering pine trees and snow-capped mountains, warm golden light, rich natural textures, professional landscape photography, no text'),
    ('little-explorer', 'Little explorer', 2103,
     'This is an RGBA image with transparency. A charming small retro robot explorer holding a tiny leafy plant in a terracotta pot, mint green enamel body with copper joints, expressive round eyes, beautifully detailed 3D illustration, soft studio lighting, centered full-body composition. The image has alpha channel and the background is transparent.'),
]
index = []
for slug, title, seed, prompt in SAMPLES:
    path = OUT / (slug + '.png')
    payload = {'prompt': prompt, 'width': 1024, 'height': 1024, 'steps': 40, 'seed': seed}
    print(f'Generating {title}...', flush=True)
    started = time.monotonic()
    req = urllib.request.Request(args.url.rstrip('/') + '/generate', data=json.dumps(payload).encode(),
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=3600) as response:
        if response.headers.get_content_type() != 'image/png':
            raise RuntimeError('Expected PNG response')
        data = response.read()
        elapsed = float(response.headers['X-Generation-Seconds'])
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(data)
    with Image.open(temporary) as image:
        image.load()
        assert image.size == (1024,1024), image.size
        mode = image.mode
        alpha_range = image.getchannel('A').getextrema() if 'A' in image.getbands() else None
    temporary.replace(path)
    index.append({'title':title, 'file':path.name, **payload, 'generation_seconds':elapsed,
                  'mode':mode, 'alpha_range':alpha_range})
    tmp_index = OUT / 'index.tmp'
    tmp_index.write_text(json.dumps(index,indent=2) + '\n')
    tmp_index.replace(OUT / 'index.json')
    print(f'Saved {path.name}: {elapsed:.1f}s, {mode}, alpha={alpha_range}', flush=True)
print('SAMPLE GALLERY COMPLETE', flush=True)
