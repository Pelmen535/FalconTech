"""Verify every file listed in BUNDLE.sha256.json; no dependencies beyond Python."""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    root=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
    manifest=json.loads((root/'BUNDLE.sha256.json').read_text('utf-8'))
    bad=[]
    for name,entry in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():bad.append(name);continue
        if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:bad.append(name)
    print(json.dumps(dict(checked=len(manifest['files']),failed=bad),ensure_ascii=False))
    return int(bool(bad))


if __name__=='__main__':raise SystemExit(main())
