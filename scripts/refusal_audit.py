"""Порог, выбранный на одной половине машин, проверяется на другой.

Зачем. Порог отказа в релизе заморожен, но выбран он был по валидации — той самой, на
которой мы отчитываемся. Вопрос «а не подогнан ли он» законен, и отвечать на него надо
измерением, а не словами.

Схема. Личности из валидации делятся пополам по `vehicle_id`: половина A и половина B.
На A перебираются все возможные пороги и выбирается лучший. Затем и замороженный порог
релиза, и выбранный на A применяются к обеим половинам. Если замороженный порог на B
ведёт себя так же, как на A, — он не подогнан под конкретные машины.

Честная оговорка, которая записывается прямо в отчёт: эти личности могли влиять на более
ранние решения о модели и пороге. Тогда B — это проверка после отбора, а не нетронутая
отложенная выборка. Мы не делаем вид, что это второе.

Исходы считаются не по массиву в памяти, а по РЕАЛЬНО записанному candidates.csv: между
решением и файлом лежит writer со своими правилами, и проверять надо тот объект, который
уходит жюри.

    python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_fit --recipe release/recipe.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval_common import counts, emitted, load_pair, save, score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', default='runs/hack/ft_soup_b336_fit')
    parser.add_argument('--query-npz')
    parser.add_argument('--gallery-npz')
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='results/refusal_audit_fixed.json')
    args = parser.parse_args()

    query, gallery, recipe, meta = load_pair(args)
    raw, rank, conf, has, correct = score(query, gallery, recipe)

    vehicle_ids = np.unique(query['vids'])
    np.random.default_rng(args.seed).shuffle(vehicle_ids)
    if len(vehicle_ids) < 2:
        raise ValueError('Нужно хотя бы две личности среди запросов')
    half = len(vehicle_ids) // 2
    in_a = np.isin(query['vids'], vehicle_ids[:half])
    in_b = ~in_a
    if not np.any(has[in_a]) or np.all(has[in_a]):
        raise ValueError('В половине A должны быть и запросы с парой, и запросы без пары')

    # Сетка порогов — все встречающиеся значения уверенности на A плюс одно значение выше
    # максимума: иначе вариант «отказать всем» недостижим и сравнение неполное.
    grid = np.append(np.unique(conf[in_a]),
                     np.nextafter(np.max(conf[in_a]), np.float32(np.inf)))
    best = max((dict(threshold=float(t), **counts(conf[in_a] >= t, has[in_a], correct[in_a]))
                for t in grid),
               key=lambda row: (row['score'], row['tnr'], row['threshold']))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, threshold in (('frozen_recipe', recipe['threshold']),
                            ('selected_on_A', best['threshold'])):
        csv_path = out.parent / f'{out.stem}_{name}_candidates.csv'
        accepted, actual = emitted(query, gallery, raw, rank, conf, threshold, csv_path)
        results[name] = dict(
            threshold=threshold,
            A=counts(accepted[in_a], has[in_a], actual[in_a]),
            B=counts(accepted[in_b], has[in_b], actual[in_b]),
            all=counts(accepted, has, actual))

    save(out, dict(
        **meta,
        seed=args.seed,
        n_query=len(conf),
        calibration_vehicle_ids=[int(v) for v in vehicle_ids[:half]],
        evaluation_vehicle_ids=[int(v) for v in vehicle_ids[half:]],
        results=results,
        threshold_grid_source='A only',
        history=('If these validation identities informed earlier model/threshold decisions, '
                 'B is a post-selection check, not untouched holdout.'),
        release_recipe_modified=False,
        official_scorer='not_run'))

    for name, block in results.items():
        print(f"[audit] {name:14s} порог {block['threshold']:.4f}  "
              f"A {block['A']['score']:.2f}  B {block['B']['score']:.2f}  "
              f"всё {block['all']['score']:.2f}")
    print(f'[audit] → {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
