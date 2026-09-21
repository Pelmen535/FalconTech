"""Доказать, что правка кода не изменила поведение — а не поверить в это.

Зачем. Чистка читаемости в `vreid/train.py` и `vreid/hack_cli.py` формально «ничего не
меняет», но именно этим кодом обучена сдаваемая модель и сняты отчётные метрики. Сказать
«я аккуратно переименовал» недостаточно: порядок обхода словаря, лишний вызов генератора
случайных чисел или переставленное слагаемое меняют числа молча.

Как. Скрипт прогоняет набор ДЕТЕРМИНИРОВАННЫХ команд на игрушечных данных и на кеше
эмбеддингов, снимает хэши их выходных файлов и складывает в JSON. До правки снимается
эталон, после правки — повторный снимок, и они сравниваются побайтово.

    python scripts/check_refactor_equivalence.py --record reports/equivalence_before.json
    ... правки ...
    python scripts/check_refactor_equivalence.py --compare reports/equivalence_before.json

Проверки идут на CPU и не трогают GPU: их можно запускать, пока идёт обучение.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_tree(root: Path, suffixes=('.json', '.csv', '.npy', '.npz', '.pt')) -> dict:
    """Хэши всех значимых файлов дерева. Имена относительные — чтобы снимок не зависел
    от того, в какой временной папке он снят."""
    if not root.exists():
        return {}
    out = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in suffixes:
            out[path.relative_to(root).as_posix()] = digest(path)
    return out


def tensor_digest(path: Path) -> str:
    """Хэш ПО ТЕНЗОРАМ чекпойнта, а не по файлу.

    Файл torch.save содержит zip с метаданными, и его байты могут отличаться при одинаковых
    весах. Сравнивать надо числа: обходим state_dict в отсортированном порядке ключей и
    подмешиваем сырые байты каждого тензора.
    """
    if not path.is_file():
        return 'missing'
    import numpy as np
    import torch

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    h = hashlib.sha256()
    for section in ('backbone', 'bnneck'):
        state = checkpoint.get(section) or {}
        for name in sorted(state):
            value = state[name]
            h.update(name.encode('utf-8'))
            h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


VOLATILE = re.compile(
    r'\d+[.,]\d+\s*(?:мс|ms|c\)|с\)|s\))'          # длительности
    r'|\d+[.,]\d+\s*batch/s'                          # скорость прогресс-бара
    r'|\d+:\d\d(?:<|,|\])'                            # таймеры прогресс-бара
    r'|\([\d.,]+\s*(?:c|с|s)\)'                        # «(5 c)»
)


def scrub(line: str) -> str | None:
    """Оставить от строки вывода только содержательное.

    Сравнивать вывод целиком нельзя: там время выполнения, скорость прогресс-бара и пути.
    Они меняются от запуска к запуску, и такая проверка кричала бы «поведение изменилось»
    на каждом прогоне. Убираем заведомо изменчивое, всё остальное — метрики, размеры,
    счётчики, пороги — сравниваем как есть.
    """
    if '\r' in line or 'it/s' in line or 'batch/s' in line or '%|' in line:
        return None
    if 'Warning' in line or 'warning' in line:
        return None
    # Строки замера времени целиком состоят из изменчивого — сравнивать в них нечего.
    if '/кроп всего' in line or 'чтение кадров' in line:
        return None
    cleaned = VOLATILE.sub('<t>', line).rstrip()
    return cleaned or None


def run(command: list[str], workdir: Path) -> dict:
    """Запустить команду проекта и вернуть код возврата плюс очищенный хвост вывода."""
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               CUDA_VISIBLE_DEVICES='', VREID_THROTTLE='0')
    result = subprocess.run([sys.executable, *command], cwd=workdir, env=env,
                            capture_output=True, text=True, encoding='utf-8', errors='replace')
    lines = [c for c in (scrub(line) for line in (result.stdout or '').splitlines()) if c]
    return {'returncode': result.returncode, 'stdout': lines[-30:]}


def toy_dataset(workdir: Path) -> None:
    root = workdir / 'data/toy_hack'
    if (root / 'train.csv').is_file():
        return
    run(['scripts/make_toy_hackathon.py', '--root', str(root), '--ids', '40', '--cams', '4'], workdir)


def checks(workdir: Path) -> dict:
    """Набор проверок. Каждая — детерминированная команда плюс хэши того, что она написала."""
    snapshot = {}

    # 1. Обучение целиком: два прохода по игрушечным данным на CPU с фиксированным сидом.
    #    Ловит любое изменение в порядке сэмплирования, в лоссах и в сохранении весов.
    out = workdir / 'runs/equivalence/train'
    shutil.rmtree(out, ignore_errors=True)
    # --pretrained false: веса инициализируются от сида, сеть не нужна, а проверка от этого
    # только строже — случайная инициализация одинаково чувствительна к порядку вызовов RNG.
    snapshot['train_toy'] = run([
        '-m', 'vreid.train', '--config', 'configs/toy_hack.yaml', '--backbone', 'dinov2_s',
        '--img-size', '70', '--epochs', '2', '--P', '3', '--K', '2', '--workers', '0',
        '--device', 'cpu', '--seed', '7', '--no-select', '--pretrained', 'false',
        '--out', str(out),
    ], workdir)
    snapshot['train_toy_files'] = digest_tree(out, suffixes=('.json',))
    snapshot['train_toy_weights'] = tensor_digest(out / 'best.pt')

    # 1b. Те же два прохода, но по другим веткам: camera-aware сэмплер, кросс-камерный
    #     triplet и режим «все id без валидации». Базовая проверка их не задевает, а именно
    #     этими ветками обучен релиз.
    out = workdir / 'runs/equivalence/train_camaware'
    shutil.rmtree(out, ignore_errors=True)
    snapshot['train_camaware'] = run([
        '-m', 'vreid.train', '--config', 'configs/toy_hack.yaml', '--backbone', 'dinov2_s',
        '--img-size', '70', '--epochs', '2', '--P', '3', '--K', '2', '--workers', '0',
        '--device', 'cpu', '--seed', '7', '--pretrained', 'false',
        '--cam-aware', '--cross-cam-triplet', '--val-frac', '0', '--out', str(out),
    ], workdir)
    snapshot['train_camaware_files'] = digest_tree(out, suffixes=('.json',))
    snapshot['train_camaware_weights'] = tensor_digest(out / 'best.pt')

    # 1c. Тексты --help: они задают порядок ключей в vars(args), а он попадает в log.json.
    for command in (['-m', 'vreid.train', '--help'],
                    ['-m', 'vreid.hack_cli', 'val', '--help'],
                    ['-m', 'vreid.hack_cli', 'submit', '--help']):
        snapshot['help_' + '_'.join(command[1:-1]).replace('.', '_')] = run(command, workdir)

    # 2. Валидация с dummy-бэкбоном: ловит изменения в протоколе, метриках и отчёте.
    snapshot['val_toy'] = run([
        '-m', 'vreid.hack_cli', 'val', '--config', 'configs/toy_hack.yaml',
        '--backbone', 'dummy', '--workers', '0',
    ], workdir)
    # Только отчёт игрушечного прогона: хэшировать весь results/ нельзя, там лежат
    # результаты настоящих замеров, которые меняются по своим причинам.
    toy_report = workdir / 'results/toy_hack_dummy_val.json'
    snapshot['val_toy_files'] = {toy_report.name: digest(toy_report)} if toy_report.is_file() else {}

    # 3. Сборка сдачи с dummy-бэкбоном: три файла должны совпасть побайтово.
    out = workdir / 'submission_equivalence'
    shutil.rmtree(out, ignore_errors=True)
    snapshot['submit_toy'] = run([
        '-m', 'vreid.hack_cli', 'submit', '--config', 'configs/toy_hack.yaml',
        '--backbone', 'dummy', '--workers', '0', '--out', str(out),
    ], workdir)
    snapshot['submit_toy_files'] = digest_tree(out)

    # 4. Постобработка на настоящих кешах: k-reciprocal, отказ, отчёт. GPU не нужен.
    cache = ROOT / 'runs/hack/ft_soup_b336_fit'
    if (cache / 'val_query_track.npz').is_file():
        report = workdir / 'results/equivalence_stress.json'
        snapshot['refusal_stress'] = run([
            'scripts/refusal_stress.py', '--run', str(cache),
            '--release', str(ROOT / 'release'), '--out', str(report),
        ], workdir)
        snapshot['refusal_stress_file'] = {report.name: digest(report)} if report.is_file() else {}

    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--record', type=Path, help='снять эталон в этот файл')
    parser.add_argument('--compare', type=Path, help='сравнить текущее состояние с эталоном')
    parser.add_argument('--workdir', type=Path, default=ROOT)
    args = parser.parse_args()
    if not args.record and not args.compare:
        parser.error('нужен либо --record, либо --compare')

    toy_dataset(args.workdir)
    snapshot = checks(args.workdir)

    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[equiv] эталон снят: {args.record}')
        for name in snapshot:
            if name.endswith('_files'):
                print(f'[equiv]   {name}: {len(snapshot[name])} файлов')
        return 0

    baseline = json.loads(args.compare.read_text(encoding='utf-8'))
    problems = []
    for key in sorted(set(baseline) | set(snapshot)):
        before, after = baseline.get(key), snapshot.get(key)
        if before == after:
            continue
        if isinstance(before, dict) and isinstance(after, dict) and key.endswith('_files'):
            for name in sorted(set(before) | set(after)):
                if before.get(name) != after.get(name):
                    problems.append(f'{key}: файл {name} изменился')
        else:
            problems.append(f'{key}: расхождение\n    было:  {before}\n    стало: {after}')

    if problems:
        print('[equiv] ПОВЕДЕНИЕ ИЗМЕНИЛОСЬ:')
        for line in problems:
            print('  -', line)
        return 1
    print(f'[equiv] поведение совпало по всем {len(snapshot)} проверкам')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
