"""Сколько стоит размер галереи, если менять ТОЛЬКО её размер.

Зачем. Наша валидация имеет галерею 688, закрытый тест — 750. Сравнивать mAP, снятые на
галереях разного размера, напрямую нельзя: чем больше отвлекающих, тем труднее задача.
Хочется знать величину эффекта.

Почему прошлый ответ был отозван. Первый эксперимент менял размер галереи и состав
запросов одновременно и потому мерил не то. Правильная схема одна: **запросы и все их
положительные кадры зафиксированы, добавляются только отвлекающие**. Именно она здесь и
реализована.

Ограничение, которое надо называть вслух: на наших данных отвлекающих почти нет — из 688
строк галереи лишь пять не являются положительными ни для одного запроса. Если пул
отвлекающих пуст, скрипт пишет `not_estimable` и не выдумывает эффект.

    python scripts/gallery_size_effect.py --run runs/hack/ft_soup_b336_fit \\
        --recipe release/recipe.json --sizes 683 685 688
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_common import counts, load_pair, save, score
from vreid.metrics import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', default='runs/hack/ft_soup_b336_fit')
    parser.add_argument('--query-npz')
    parser.add_argument('--gallery-npz')
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--query-ids-json',
                        help='заранее объявленный фиксированный набор запросов; по умолчанию все')
    parser.add_argument('--sizes', type=int, nargs='+', required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='results/gallery_size_effect_fixed.json')
    args = parser.parse_args()

    query, gallery, recipe, meta = load_pair(args)

    if args.query_ids_json:
        # Набор запросов объявляется заранее и отдельным файлом: выбирать его по тому,
        # где метрика удачно падает, — это подгонка, а не измерение.
        keys = json.loads(Path(args.query_ids_json).read_text('utf-8'))
        if len(keys) != len(set(keys)) or not set(keys) <= set(query['keys']):
            raise ValueError('Набор запросов должен быть без повторов и целиком лежать в кеше')
        keep = np.isin(query['keys'], keys)
        query = {name: values[keep] for name, values in query.items()}

    positives = np.flatnonzero(np.isin(gallery['vids'], query['vids']))
    distractors = np.flatnonzero(~np.isin(gallery['vids'], query['vids']))

    report = dict(**meta,
                  design='fixed_queries_all_positives_distractors_only',
                  threshold=recipe['threshold'],
                  fixed_query_ids=[str(k) for k in query['keys']],
                  positive_rows_retained=len(positives),
                  available_distractors=len(distractors),
                  rows=[])

    if not len(distractors):
        report.update(status='not_estimable',
                      reason=('All gallery rows are positives for the fixed query cohort; '
                              'no distractor pool. Do not invent a size effect.'))
        save(args.out, report)
        print('[gallery] отвлекающих нет: эффект не оценивается, и выдумывать его нельзя')
        return 0

    smallest = max(10, len(positives))
    if any(size < smallest or size > len(gallery['emb']) for size in args.sizes):
        raise ValueError(f'Размер галереи должен быть от {smallest} (все положительные остаются) '
                         f'до {len(gallery["emb"])}')

    for repeat in range(args.repeats):
        order = np.random.default_rng(args.seed + repeat).permutation(distractors)
        for size in args.sizes:
            index = np.sort(np.concatenate([positives, order[:size - len(positives)]]))
            subset = {name: values[index] for name, values in gallery.items()}
            _, rank, conf, has, correct = score(query, subset, recipe)
            jury = evaluate(rank, query['vids'], query['cams'],
                            subset['vids'], subset['cams'], cutoff=10)
            report['rows'].append(dict(
                seed=args.seed + repeat, gallery=size,
                mAP10=jury['mAP'], rank1=jury['rank1'],
                refusal=counts(conf >= recipe['threshold'], has, correct),
                n_query=len(query['emb'])))

    report['status'] = 'measured_fixed_cohort'
    save(args.out, report)
    for size in args.sizes:
        values = [row['mAP10'] for row in report['rows'] if row['gallery'] == size]
        print(f'[gallery] {size:5d} строк: mAP@10 {100 * float(np.mean(values)):.2f} '
              f'по {len(values)} повтор(ам)')
    print(f'[gallery] → {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
