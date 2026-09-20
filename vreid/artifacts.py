"""Structural submission validation. This is NOT the organizer's scoring script."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

WEIGHT_EXTENSIONS = {'.pt', '.pth', '.bin', '.onnx', '.engine', '.plan',
                     '.safetensors', '.ckpt', '.trt', '.pb', '.tflite', '.npz'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path, header=None):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if header is not None and reader.fieldnames != header:
            raise ValueError(f'{path}: expected exact header {header}, got {reader.fieldnames}')
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f'{path}: malformed CSV row')
    return rows


def input_ids(path):
    rows = read_csv(path)
    ids = [r['image_id'] for r in rows]
    if not ids or any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'{path}: expected nonempty unique image_id')
    return ids


def validate_outputs(data, output, expected_dim=None):
    data, output = Path(data), Path(output)
    queries = input_ids(data / 'test_query.csv')
    gallery = input_ids(data / 'test_gallery.csv')
    if len(gallery) < 10:
        raise ValueError('gallery needs at least ten entries')
    qset, gset = set(queries), set(gallery)
    header = ['query_id'] + [f'gallery_id_{i}' for i in range(1, 11)]
    ranking = read_csv(output / 'submission.csv', header)
    if [r['query_id'] for r in ranking] != queries:
        raise ValueError('submission query coverage/order differs from input CSV')
    for row in ranking:
        candidates = [row[h] for h in header[1:]]
        if len(set(candidates)) != 10 or not set(candidates) <= gset:
            raise ValueError(f'invalid top10 for {row["query_id"]}')
    candidates = read_csv(output / 'candidates.csv', ['query_id', 'gallery_id', 'confidence'])
    seen = set()
    for row in candidates:
        if row['query_id'] not in qset or row['gallery_id'] not in gset:
            raise ValueError('candidate uses unknown image_id')
        if not np.isfinite(float(row['confidence'])):
            raise ValueError('nonfinite confidence')
        pair = (row['query_id'], row['gallery_id'])
        if pair in seen:
            raise ValueError('duplicate candidate pair')
        seen.add(pair)
    emb = np.load(output / 'embeddings.npy', allow_pickle=False, mmap_mode='r')
    if emb.ndim != 2 or emb.shape[0] != len(queries) + len(gallery) or emb.shape[1] < 1:
        raise ValueError('embedding shape must be (query+gallery, dimension)')
    if expected_dim is not None and emb.shape[1] != expected_dim:
        raise ValueError('embedding dimension differs from model')
    if emb.dtype != np.float32 or not np.isfinite(emb).all():
        raise ValueError('embeddings must be finite float32')
    if np.any(np.linalg.norm(emb, axis=1) <= 1e-8):
        raise ValueError('zero embedding')
    return {'structural_validation': 'passed', 'official_scoring': 'not_run',
            'n_query': len(queries), 'n_gallery': len(gallery), 'dimension': emb.shape[1],
            'refused': len(queries) - len({r['query_id'] for r in candidates}),
            'embedding_row_order': 'shape checked; provenance/CLI test required for semantics',
            'sha256': {p: sha256(output / p) for p in
                       ['submission.csv', 'candidates.csv', 'embeddings.npy']}}


def weight_inventory(root):
    paths = sorted(p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in WEIGHT_EXTENSIONS)
    entries = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                'sha256': sha256(p)} for p in paths]
    total = sum(x['bytes'] for x in entries)
    # Use a conservative decimal 2 GB ceiling; do not assume 2 GiB is accepted.
    if total > 2_000_000_000:
        raise ValueError(f'weight files exceed 2 GB: {total}')
    return {'files': entries, 'total_bytes': total, 'limit_bytes': 2_000_000_000}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dimension', type=int)
    ap.add_argument('--report')
    args = ap.parse_args()
    report = validate_outputs(args.data, args.out, args.dimension)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        path = Path(args.report); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
