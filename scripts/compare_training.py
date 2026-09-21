"""Сравнить два рецепта обучения по нескольким сидам и решить, стоит ли менять релиз.

Зачем отдельный скрипт, а не взгляд глазами. Разброс между сидами одного рецепта на наших
данных — 2.9 пункта (fit18: 71.63 / 71.24 / 68.73). При таком шуме один прогон не
доказывает ничего: выигрыш в полтора пункта у одиночной модели укладывается в случайность.
Решение о смене релиза должно приниматься по среднему нескольких сидов и по явному порогу,
заданному ЗАРАНЕЕ, а не подобранному под результат.

Числа берутся из log.json каждого прогона — это та же метрика жюри (mAP@10 с junk-фильтром
по vid+cam), которую печатает train.py. Сравнивать их можно только внутри одного разбиения,
поэтому скрипт сверяет эпоху 0: у одинакового сплита она обязана совпасть.

    python scripts/compare_training.py \\
        --baseline weights/fit18_s0 weights/fit18_s2 weights/fit18_s3 \\
        --candidate weights/fit_ls20_s0 weights/fit_ls20_s2 weights/fit_ls20_s3 \\
        --min-gain 1.5 --out results/training_upgrade.json

Код возврата 0 — прирост подтверждён, можно переобучать релиз; 1 — нет.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_run(run_dir: Path) -> dict:
    log_path = Path(run_dir) / 'log.json'
    if not log_path.is_file():
        raise SystemExit(f'нет {log_path}: прогон не завершён или путь неверен')
    data = json.loads(log_path.read_text(encoding='utf-8'))
    epochs = data.get('log') or []
    if not epochs:
        raise SystemExit(f'{log_path}: пустой журнал эпох')
    args = data.get('args', {})
    # Последняя эпоха, а не лучшая: релиз обучается фиксированным числом эпох без отбора,
    # и сравнивать надо то, что он реально получит.
    last = epochs[-1]
    zero = next((row for row in epochs if row.get('epoch') == 0), None)
    return {
        'run': str(run_dir),
        'epochs': args.get('epochs'),
        'arc_ls': args.get('arc_ls'),
        'seed': args.get('seed'),
        'distill_w': args.get('distill_w'),
        'no_select': bool(args.get('no_select')),
        'final_epoch': last.get('epoch'),
        'final_map10': round(100 * float(last['mAP']), 3),
        'epoch0_map10': None if zero is None else round(100 * float(zero['mAP']), 3),
        'curve': [round(100 * float(r['mAP']), 2) for r in epochs],
    }


def summarise(name: str, runs: list[dict]) -> dict:
    values = [r['final_map10'] for r in runs]
    mean = sum(values) / len(values)
    spread = max(values) - min(values)
    return {'name': name, 'runs': runs, 'n': len(runs),
            'mean_map10': round(mean, 3), 'best_map10': max(values),
            'worst_map10': min(values), 'spread': round(spread, 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', nargs='+', required=True, help='папки прогонов старого рецепта')
    parser.add_argument('--candidate', nargs='+', required=True, help='папки прогонов нового рецепта')
    parser.add_argument('--min-gain', type=float, default=1.5,
                        help='на сколько пунктов среднее должно вырасти, чтобы менять релиз')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/training_upgrade.json')
    args = parser.parse_args()

    base = summarise('baseline', [read_run(Path(p)) for p in args.baseline])
    cand = summarise('candidate', [read_run(Path(p)) for p in args.candidate])

    # Одинаковое разбиение — условие сравнимости. Эпоха 0 считается ещё не обученной
    # моделью, поэтому на одном сплите она обязана дать одно и то же число.
    zeros = {r['epoch0_map10'] for r in base['runs'] + cand['runs'] if r['epoch0_map10'] is not None}
    comparable = len(zeros) <= 1
    gain = round(cand['mean_map10'] - base['mean_map10'], 3)
    decision = comparable and gain >= args.min_gain

    report = {
        'baseline': base, 'candidate': cand,
        'gain_mean_map10': gain, 'min_gain_required': args.min_gain,
        'same_split': comparable,
        'epoch0_values': sorted(zeros),
        'decision': 'переобучать релиз' if decision else 'оставить прежний релиз',
        'why': ('среднее выросло больше порога, заданного заранее' if decision else
                ('прогоны считались на РАЗНЫХ разбиениях, сравнивать нельзя' if not comparable else
                 'прирост внутри разброса между сидами — на одном шуме релиз не меняют')),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    for block in (base, cand):
        values = ', '.join(f"{r['final_map10']:.2f}" for r in block['runs'])
        print(f"[compare] {block['name']:<10} среднее {block['mean_map10']:.2f}  "
              f"разброс {block['spread']:.2f}  ({values})")
    if not comparable:
        print(f'[compare] РАЗНЫЕ СПЛИТЫ: эпоха 0 даёт {sorted(zeros)} — числа несопоставимы')
    print(f"[compare] прирост {gain:+.2f} при пороге {args.min_gain:+.2f} → {report['decision']}")
    print(f'[compare] → {args.out}')
    return 0 if decision else 1


if __name__ == '__main__':
    raise SystemExit(main())
