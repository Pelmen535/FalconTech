"""Порог отказа для нового релиза — по записанному правилу, а не вписанным руками числом.

    python scripts/refusal_stress.py --run runs/hack/ft_soup_r392dw40f4_fit \\
        --out results/refusal_stress_r392dw40f4.json
    python scripts/choose_threshold.py --stress results/refusal_stress_r392dw40f4.json \\
        --out results/threshold_choice_r392dw40f4.json

ЗАЧЕМ. Порог 0.55 выбран под модели v1.0–v1.2. Ученик с фокусом на трудных парах увереннее
(медиана уверенности без дубликатов 0.60 против 0.51), и с тем же порогом он отказывает тем,
кому надо отвечать. Порог надо выбирать заново для каждой модели — и по одному правилу, чтобы
выбор не превращался в подгонку.

ПРАВИЛО. Максимум ожидаемого балла отказа 0.7·F1 + 0.3·TNR при вере `belief` в сценарий A
(у запроса с парой в галерее есть кадр той же машины с той же камеры) против B (такого кадра
нет), по сетке порогов с шагом 0.01 из refusal_stress.py. Вера 0.85 — потому что открытый тест
устроен как A: у v1.2 медиана уверенности лучшего ответа на нём 0.918, у 77% запросов она не
ниже 0.8 — как в сценарии A валидации (64%) и совсем не как в B (1.6%). Единица была бы
неосторожна: закрытый тест гарантирует только кросс-камерную пару (ответ 17).

Пишет обоснование в формате поля threshold_choice рецепта (scripts/export_release.py
--threshold-choice) и само число отдельным файлом: так его читает очередь PowerShell, не
завися от культуры чисел.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WHY = ("В валидации и в открытом тесте галерея собрана по одному кадру на трек «машина × "
       "камера», поэтому у запроса с парой почти всегда есть кадр той же машины с той же камеры "
       "(сценарий A). На открытом тесте это видно по уверенности v1.2: медиана лучшего ответа "
       "0.918, у 77% запросов не ниже 0.8 — как в сценарии A валидации (64%) и совсем не как "
       "без дубликатов (1.6%). Закрытый тест гарантирует только кросс-камерную пару (ответ 17), "
       "поэтому вера в A — 0.85, а не единица.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stress", type=Path, required=True, help="results/refusal_stress_<run>.json")
    ap.add_argument("--belief", type=float, default=0.85, help="вера в сценарий A")
    ap.add_argument("--previous", type=Path, default=Path("release/recipe.json"),
                    help="рецепт прежнего релиза: его порог записывается для сравнения")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-value", type=Path, default=None,
                    help="файл с одним числом порога; по умолчанию <out>.value.txt")
    a = ap.parse_args()

    stress = json.loads(a.stress.read_text(encoding="utf-8"))
    # только сетка 0.01: порог прежнего релиза refusal_stress добавляет в неё отдельной строкой,
    # и выбирать его просто потому, что он там есть, нельзя
    rows = [r for r in stress["rows"] if abs(r["threshold"] * 100 - round(r["threshold"] * 100)) < 1e-6]
    p = a.belief
    expect = lambda r: p * r["A"]["score"] + (1 - p) * r["B"]["score"]
    best = max(rows, key=expect)
    grid = [r["threshold"] for r in rows]
    if best["threshold"] in (min(grid), max(grid)):
        raise SystemExit(f"[threshold] максимум на краю сетки ({best['threshold']:.2f}) - "
                         f"расширь сетку в refusal_stress.py, а не бери край")
    optimum_a = max(rows, key=lambda r: r["A"]["score"])
    minimax = max(rows, key=lambda r: r["worst"])

    # при какой вере в A выбранный порог и минимакс равноценны: ниже неё лучше минимакс
    d_a = best["A"]["score"] - minimax["A"]["score"]
    d_b = minimax["B"]["score"] - best["B"]["score"]
    indifference = d_b / (d_a + d_b) if d_a + d_b > 0 else None

    previous = None
    if a.previous.is_file():
        prev = json.loads(a.previous.read_text(encoding="utf-8"))
        previous = prev.get("threshold")
    prev_row = next((r for r in stress["rows"] if previous is not None
                     and abs(r["threshold"] - previous) < 1e-9), None)

    pick = lambda r: {"with_duplicates": round(r["A"]["score"], 2),
                      "without_duplicates": round(r["B"]["score"], 2),
                      f"expected_at_belief_{p:g}": round(expect(r), 2)}
    table = {f"{r['threshold']:.2f}": pick(r) for r in (best, optimum_a, minimax)}
    if prev_row is not None:
        table[f"{prev_row['threshold']:.2f} (прежний)"] = pick(prev_row)

    choice = {
        "value": round(best["threshold"], 2),
        "selected_by": f"максимум ожидаемого балла отказа 0.7·F1 + 0.3·TNR при вере {p:g} в "
                       f"сценарий A, сетка 0.01 (scripts/choose_threshold.py)",
        "belief_in_A": p,
        "reason": WHY,
        "validation_optimum": round(optimum_a["threshold"], 2),
        "minimax": round(minimax["threshold"], 2),
        "refusal_score": {"note": "0.7*F1 + 0.3*TNR, отложенная валидация двойника релиза, "
                                  f"{stress.get('n_query')} запросов", **table},
        "indifference_probability": None if indifference is None else round(indifference, 3),
        "previous_threshold": previous,
        "reproduce": f"python scripts/choose_threshold.py --stress {a.stress.as_posix()} --belief {p:g}",
        "source": a.stress.as_posix(),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(choice, ensure_ascii=False, indent=2), encoding="utf-8")
    value_path = a.out_value or a.out.with_suffix(".value.txt")
    value_path.write_text(f"{choice['value']:.2f}\n", encoding="ascii")
    print(f"[threshold] порог {choice['value']:.2f} (вера в A {p:g}); оптимум A "
          f"{choice['validation_optimum']:.2f}, минимакс {choice['minimax']:.2f}, прежний {previous}")
    for k, v in table.items():
        print(f"[threshold]   {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
