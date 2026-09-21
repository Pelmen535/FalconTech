"""Прогнать НАШУ валидацию через ЭТАЛОННЫЙ scorer организаторов и сверить числа.

Зачем. До получения `evaluate.py` все наши метрики были реализованы по описанию из ТЗ и из
письменных ответов. Мы считали их правильными, но это было прочтение, а не проверка. Теперь
скрипт организаторов есть, и на закрытом тесте меток у нас по-прежнему нет — зато они есть
на нашей отложенной валидации. Значит можно сделать единственно честную вещь: собрать из
валидации набор ровно того формата, который ждёт `evaluate.py`, прогнать его и сравнить
полученные числа с нашими.

Совпали — наша реализация метрики подтверждена сторонним кодом. Разошлись — у нас есть
конкретное число, за которым надо идти, а не смутное подозрение.

Что делает скрипт:
  1. Берёт кеш эмбеддингов валидации (query/gallery, метки vehicle_id и camera_id).
  2. Считает ранжирование и отказ ТЕМ ЖЕ кодом, что и боевой прогон (vreid.predict через
     scripts/eval_common), с параметрами из замороженного рецепта.
  3. Пишет в отдельную папку: test_ground_truth.csv, test_query.csv, test_gallery.csv,
     embeddings.npy, submission.csv (без заголовка), candidates.csv.
  4. Запускает organizer/evaluate.py и разбирает его JSON.
  5. Сравнивает с нашими числами и печатает разницу по каждой метрике.

    python scripts/score_validation_with_official.py --run runs/hack/ft_soup_b336_fit \\
        --release release --out results/official_validation
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_common import counts, score                     # noqa: E402
from vreid.metrics import evaluate                        # noqa: E402
from vreid.predict import load_recipe                     # noqa: E402
from vreid.submit import save_embeddings, write_candidates, write_submission   # noqa: E402


def load_pack(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ('emb', 'vids', 'cams', 'keys')}


def write_rows(path: Path, header: list[str], rows) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/hack/ft_soup_b336_fit')
    parser.add_argument('--release', type=Path, default=ROOT / 'release')
    parser.add_argument('--scorer', type=Path, default=ROOT / 'organizer/evaluate.py')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/official_validation')
    args = parser.parse_args()

    if not args.scorer.is_file():
        raise SystemExit(f'нет {args.scorer}: положи эталонный evaluate.py организаторов')

    recipe = load_recipe(args.release)
    query = load_pack(args.run / 'val_query_track.npz')
    gallery = load_pack(args.run / 'val_gallery_track.npz')

    query_ids = [str(k) for k in query['keys']]
    gallery_ids = [str(k) for k in gallery['keys']]
    overlap = set(query_ids) & set(gallery_ids)
    if overlap:
        # Эталонный scorer индексирует ground truth по image_id, поэтому один и тот же
        # идентификатор не может быть и запросом, и галереей. В нашей валидации разбиение
        # по трекам это гарантирует; проверяем, а не надеемся.
        raise SystemExit(f'{len(overlap)} идентификаторов есть и в query, и в gallery — '
                         f'ground truth такого формата не выразит')

    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / 'test_ground_truth.csv', ['image_id', 'vehicle_id', 'camera_id', 'split'],
               [[key, int(vid), int(cam), 'query']
                for key, vid, cam in zip(query_ids, query['vids'], query['cams'])] +
               [[key, int(vid), int(cam), 'gallery']
                for key, vid, cam in zip(gallery_ids, gallery['vids'], gallery['cams'])])
    # bbox в этих CSV эталонному скрипту не нужен, но колонки должны быть на месте:
    # он читает их как обычный набор данных задачи.
    write_rows(args.out / 'test_query.csv', ['image_id', 'x', 'y', 'w', 'h'],
               [[key, 0, 0, 1, 1] for key in query_ids])
    write_rows(args.out / 'test_gallery.csv', ['image_id', 'x', 'y', 'w', 'h'],
               [[key, 0, 0, 1, 1] for key in gallery_ids])

    raw, rank, conf, has_match, correct = score(query, gallery, recipe)
    save_embeddings(np.concatenate([query['emb'], gallery['emb']]), args.out / 'embeddings.npy')
    write_submission(query_ids, gallery_ids, rank, args.out / 'submission.csv',
                     topk=int(recipe['topk']), exclude_self=False)
    write_candidates(query_ids, gallery_ids, rank, conf, float(recipe['threshold']),
                     args.out / 'candidates.csv', topk=int(recipe['candidates_topk']),
                     exclude_self=False, conf_sims=raw)

    ours = {
        'mAP@10': evaluate(rank, query['vids'], query['cams'],
                           gallery['vids'], gallery['cams'], cutoff=10),
        'refusal': counts(conf >= float(recipe['threshold']), has_match, correct),
    }

    command = [sys.executable, str(args.scorer),
               '--gt', str(args.out / 'test_ground_truth.csv'),
               '--submission', str(args.out / 'submission.csv'),
               '--candidates', str(args.out / 'candidates.csv'),
               '--embeddings', str(args.out / 'embeddings.npy'),
               '--query', str(args.out / 'test_query.csv'),
               '--gallery', str(args.out / 'test_gallery.csv'),
               '--json', str(args.out / 'official_report.json')]
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding='utf-8', errors='replace',
                            env={**os.environ, 'PYTHONUTF8': '1'})
    (args.out / 'official_stdout.txt').write_text(result.stdout or '', encoding='utf-8')
    (args.out / 'official_stderr.txt').write_text(result.stderr or '', encoding='utf-8')
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f'эталонный scorer вернул код {result.returncode}')

    official = json.loads((args.out / 'official_report.json').read_text(encoding='utf-8'))

    pairs = [
        ('mAP@10', official['ranking']['mAP@10'], ours['mAP@10']['mAP']),
        ('Rank-1', official['ranking']['Rank-1'], ours['mAP@10']['rank1']),
        ('Rank-5', official['ranking']['Rank-5'], ours['mAP@10']['rank5']),
        ('запросов в зачёте', official['ranking']['n_scored'], ours['mAP@10']['n_query_valid']),
        ('F1 отказа', official['candidates']['F1'], ours['refusal']['f1']),
        ('TNR отказа', official['candidates']['TNR'], ours['refusal']['tnr']),
        ('TP', official['candidates']['TP'], ours['refusal']['tp']),
        ('FP', official['candidates']['FP'], ours['refusal']['fp']),
        ('FN', official['candidates']['FN'], ours['refusal']['fn']),
        ('TN', official['candidates']['TN'], ours['refusal']['tn']),
    ]
    print(f"\n{'метрика':<22}{'эталонный scorer':>20}{'наш расчёт':>16}{'разница':>12}")
    print('-' * 70)
    worst = 0.0
    for name, theirs, mine in pairs:
        delta = float(theirs) - float(mine)
        worst = max(worst, abs(delta))
        print(f'{name:<22}{float(theirs):>20.6f}{float(mine):>16.6f}{delta:>12.2e}')

    jury_score = 100 * (0.7 * official['candidates']['F1'] + 0.3 * official['candidates']['TNR'])
    print(f"\nблок отказа по формуле организаторов 0.7*F1 + 0.3*TNR: {jury_score:.2f}")

    summary = {
        'scope': ('эталонный scorer на НАШЕЙ отложенной валидации; закрытый тест здесь '
                  'не оценивается и оценён быть не может'),
        'run': str(args.run), 'release': str(args.release),
        'scorer_sha256': __import__('hashlib').sha256(args.scorer.read_bytes()).hexdigest(),
        'official': official,
        'ours': {'mAP@10': ours['mAP@10']['mAP'], 'rank1': ours['mAP@10']['rank1'],
                 'rank5': ours['mAP@10']['rank5'], 'refusal': ours['refusal']},
        'max_abs_difference': worst,
        'refusal_block_score': jury_score,
        'verdict': ('наша реализация метрики совпала с эталонной' if worst < 1e-6 else
                    'РАСХОЖДЕНИЕ с эталонной реализацией — разбираться'),
    }
    (args.out / 'comparison.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n{summary['verdict']}  (максимальное расхождение {worst:.2e})")
    print(f'[official] → {args.out}')
    return 0 if worst < 1e-6 else 1


if __name__ == '__main__':
    raise SystemExit(main())
