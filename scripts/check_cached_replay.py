"""Пересобрать ранжирование по УЖЕ ЗАПИСАННЫМ эмбеддингам сдачи и сравнить с тем, что лежит.

Зачем. Сдача состоит из трёх файлов, и два из них выводимы из третьего: имея
`embeddings.npy` и рецепт, можно заново построить `submission.csv` и `candidates.csv`.
Если пересборка расходится с тем, что лежит рядом, значит файлы получены разным кодом или
разными настройками — и это надо увидеть до отправки, а не после.

Проверка дешёвая: сеть не запускается, кропы не читаются, GPU не нужен. Она ничего не
говорит о точности — меток здесь нет, только согласованность артефактов между собой.

    python scripts/check_cached_replay.py --submission submission --data Данные \\
        --release release --out results/cached_replay.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vreid.predict import load_recipe, rerank_chunk_size, sha256          # noqa: E402
from vreid.rerank import frame_block_mask, k_reciprocal_chunked           # noqa: E402
from vreid.submit import write_candidates, write_submission               # noqa: E402


def read_csv(path) -> list[dict]:
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('submission', 'data', 'release', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()

    source = Path(args.submission)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    query_keys = [row['image_id'] for row in read_csv(Path(args.data) / 'test_query.csv')]
    gallery_keys = [row['image_id'] for row in read_csv(Path(args.data) / 'test_gallery.csv')]

    embeddings = np.load(source / 'embeddings.npy', allow_pickle=False)
    query, gallery = embeddings[:len(query_keys)], embeddings[len(query_keys):]
    if len(gallery) != len(gallery_keys):
        raise ValueError('Число строк в embeddings.npy не совпадает с длинами входных CSV')

    recipe = load_recipe(Path(args.release))
    raw = np.stack([gallery @ row for row in query])
    blocked = frame_block_mask(query_keys, gallery_keys)
    raw[blocked] = -np.inf

    if recipe['kr']:
        chunk = rerank_chunk_size(len(query), len(gallery), 12)
        rank = -k_reciprocal_chunked(query, gallery, recipe['k1'], recipe['k2'],
                                     float(recipe.get('lam', 0.3)),
                                     q_keys=query_keys, g_keys=gallery_keys, chunk=chunk)
        rank[blocked] = -np.inf
    else:
        rank = raw.copy()

    selected = np.argsort(-rank, axis=1, kind='stable')[:, 0]
    conf = raw[np.arange(len(raw)), selected]
    write_submission(query_keys, gallery_keys, rank, out / 'submission.csv', exclude_self=False)
    write_candidates(query_keys, gallery_keys, rank, conf, recipe['threshold'],
                     out / 'candidates.csv', exclude_self=False, conf_sims=raw)

    old = read_csv(source / 'submission.csv')
    new = read_csv(out / 'submission.csv')
    if [row['query_id'] for row in old] != query_keys:
        raise ValueError('Порядок запросов в сдаче не совпадает с порядком входного CSV')

    old_candidates = {row['query_id']: row for row in read_csv(source / 'candidates.csv')}
    new_candidates = {row['query_id']: row for row in read_csv(out / 'candidates.csv')}

    info = dict(
        n_query=len(query_keys), n_gallery=len(gallery_keys),
        embeddings_sha256=sha256(source / 'embeddings.npy'),
        recipe_sha256=sha256(Path(args.release) / 'recipe.json'),
        top1_changes=sum(a['gallery_id_1'] != b['gallery_id_1'] for a, b in zip(old, new)),
        changed_rank_cells=sum(a[f'gallery_id_{i}'] != b[f'gallery_id_{i}']
                               for a, b in zip(old, new) for i in range(1, 11)),
        acceptance_changes=len(set(old_candidates) ^ set(new_candidates)),
        old_accepted=len(old_candidates), new_accepted=len(new_candidates),
        candidate_id_changes=sum(old_candidates[key]['gallery_id'] != new_candidates[key]['gallery_id']
                                 for key in set(old_candidates) & set(new_candidates)),
        scope=('Cached ranking regression only; no new extraction, accuracy '
               'or GPU performance measurement.'))

    (out / 'comparison.json').write_text(json.dumps(info, indent=2), 'utf-8')
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
