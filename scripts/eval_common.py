"""Общий код диагностик: ранжирование по кешу эмбеддингов, исходы отказа, запись отчёта.

Здесь не проводится никакой оценки точности по закрытому тесту — только пересчёт того, что
уже посчитано боевым путём, на сохранённых векторах. Всё, что отсюда берут скрипты
`refusal_stress.py`, `refusal_audit.py`, `gallery_size_effect.py` и `error_analysis.py`,
использует тот же рецепт и тот же ре-ранжировщик, что и конкурсный прогон.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vreid.predict import load_recipe, rerank_chunk_size          # noqa: E402
from vreid.rerank import frame_block_mask, k_reciprocal_chunked   # noqa: E402
from vreid.submit import write_candidates                         # noqa: E402


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: нечисло в отчёте — это молчаливая ложь о том, что что-то измерено.
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")


def load_pair(args):
    """Кеши запросов и галереи плюс рецепт. Проверки здесь, а не в вызывающем скрипте.

    Хэши кешей и рецепта возвращаются вместе с данными: отчёт без них нельзя привязать
    к конкретной модели, а «имя файла похоже на нужное» доказательством не является.
    """
    query_path = Path(args.query_npz) if args.query_npz else Path(args.run) / 'val_query_track.npz'
    gallery_path = (Path(args.gallery_npz) if args.gallery_npz
                    else Path(args.run) / 'val_gallery_track.npz')

    def read(path):
        with np.load(path, allow_pickle=False) as data:
            pack = {key: data[key] for key in ('emb', 'vids', 'cams', 'keys')}
        if not np.isfinite(pack['emb']).all():
            raise ValueError(f'{path}: в кеше есть NaN или inf')
        if any(len(pack[key]) != len(pack['emb']) for key in pack):
            raise ValueError(f'{path}: длины полей кеша не совпадают')
        if len(set(pack['keys'])) != len(pack['keys']):
            raise ValueError(f'{path}: повторяющиеся ключи эмбеддингов')
        return pack

    if Path(args.recipe).name != 'recipe.json':
        raise ValueError('Указывай recipe.json именно того релиза, который проверяешь')
    recipe = load_recipe(Path(args.recipe).parent)
    provenance = {
        'query_npz_sha256': digest(query_path),
        'gallery_npz_sha256': digest(gallery_path),
        'recipe_sha256': digest(args.recipe),
        'cache_provenance': ('Requires model SHA and extraction log from producer; '
                             'hashes alone do not prove decoder/model'),
    }
    return read(query_path), read(gallery_path), recipe, provenance


def score(query, gallery, recipe):
    """Ранжирование и уверенность ровно так, как их считает vreid/predict.py.

    Возвращает:
      raw      сырые косинусы [Q, G], запрещённые пары уже -inf;
      rank     порядок ранжирования (после k-reciprocal, если он включён в рецепте);
      conf     уверенность — сырой косинус ВЫБРАННОГО кандидата, а не максимум по строке;
      has      есть ли у запроса в галерее та же машина (основа режима отказа);
      correct  верен ли выбранный кандидат.
    """
    raw = np.stack([gallery['emb'] @ row for row in query['emb']])
    blocked = frame_block_mask(query['keys'], gallery['keys'])
    raw[blocked] = -np.inf

    if recipe['kr']:
        chunk = rerank_chunk_size(len(query['emb']), len(gallery['emb']), 12)
        distance = k_reciprocal_chunked(query['emb'], gallery['emb'],
                                        recipe['k1'], recipe['k2'], float(recipe.get('lam', .3)),
                                        chunk=chunk,
                                        q_keys=query['keys'], g_keys=gallery['keys'])
        rank = -distance
        rank[blocked] = -np.inf
    else:
        rank = raw.copy()

    selected = np.argsort(-rank, axis=1, kind='stable')[:, 0]
    conf = raw[np.arange(len(raw)), selected]
    has = np.array([np.any((gallery['vids'] == vid) & ~blocked[i])
                    for i, vid in enumerate(query['vids'])])
    correct = gallery['vids'][selected] == query['vids']
    return raw, rank, conf, has, correct


def counts(accept, has, correct) -> dict:
    """Матрица ошибок режима отказа по правилу организаторов (ответы 14/19/25).

    TP — принят верный топ-1; FP — принят неверный либо принят запрос без пары;
    FN — отказ при наличии пары; TN — отказ без пары. F1 и TNR считаются micro.
    """
    tp = int((accept & has & correct).sum())
    fp = int((accept & ~(has & correct)).sum())
    fn = int((~accept & has).sum())
    tn = int((~accept & ~has).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    tnr = tn / max(int((~has).sum()), 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, f1=f1, tnr=tnr,
                score=100 * (.7 * f1 + .3 * tnr), accept_rate=float(accept.mean()))


def emitted(query, gallery, raw, rank, conf, threshold, path):
    """Исходы, прочитанные из РЕАЛЬНО записанного candidates.csv, а не из намерения.

    Разница не косметическая: между решением и файлом лежит write_candidates со своими
    правилами (отказ = отсутствие строк, нечисловая уверенность = отказ, одна строка на
    запрос). Считать по массиву в памяти значит проверять не тот объект, который сдаётся.
    """
    write_candidates(query['keys'], gallery['keys'], rank, conf, threshold, path,
                     exclude_self=False, conf_sims=raw)
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))

    best = {}
    for row in rows:
        current = best.get(row['query_id'])
        if current is None or float(row['confidence']) > float(current['confidence']):
            best[row['query_id']] = row

    gallery_vid = dict(zip(gallery['keys'], gallery['vids']))
    accept = np.array([str(key) in best for key in query['keys']])
    correct = np.array([str(key) in best and gallery_vid[best[str(key)]['gallery_id']] == vid
                        for key, vid in zip(query['keys'], query['vids'])])
    return accept, correct
