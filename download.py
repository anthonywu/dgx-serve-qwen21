#!/usr/bin/env python3
"""Resumable pinned download with complete upstream hash verification."""
import hashlib, json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = 'Qwen/Qwen-Image-2.1'
REVISION = 'b3179ad355be050328e483a9dfdd9e60cd62adfa'
DEST = ROOT / 'model'

def main():
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    total = sum(f.size for f in info.siblings)
    # Be conservative: stale/partial cache files do not reduce the required reserve.
    existing = sum(min((DEST / f.rfilename).stat().st_size, f.size)
                   for f in info.siblings if (DEST / f.rfilename).is_file())
    free = shutil.disk_usage(ROOT).free
    needed = max(0, total - existing) + 8 * 1024**3
    print(f'Model: {total / 1024**3:.2f} GiB; disk free: {free / 1024**3:.2f} GiB', flush=True)
    if free < needed:
        raise SystemExit('Insufficient disk space including 8 GiB reserve.')
    snapshot_download(REPO, revision=REVISION, local_dir=DEST, max_workers=3)
    records = []
    for f in info.siblings:
        p = DEST / f.rfilename
        if p.stat().st_size != f.size:
            raise RuntimeError(f'Size mismatch: {p}')
        h = hashlib.sha256() if f.lfs else hashlib.sha1()
        if not f.lfs:
            h.update(f'blob {f.size}\0'.encode())
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                h.update(block)
        expected = f.lfs.sha256 if f.lfs else f.blob_id
        if h.hexdigest() != expected:
            raise RuntimeError(f'Hash mismatch: {p}')
        records.append({'path': f.rfilename, 'size': f.size, 'hash': expected, 'mtime_ns': p.stat().st_mtime_ns})
        print(f'Verified {f.rfilename}', flush=True)
    receipt = {'repo': REPO, 'revision': REVISION, 'files': records}
    (ROOT / 'verified.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('DOWNLOAD AND VERIFICATION COMPLETE', flush=True)

if __name__ == '__main__':
    main()
