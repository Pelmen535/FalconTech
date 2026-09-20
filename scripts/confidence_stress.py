"""Какая мера уверенности переживёт закрытый тест: сравнение под тем же стрессом, что и порог.

Вопрос. Сейчас уверенность — сырой косинус выбранного кандидата (`top1`). На нашей валидации
он лучший из всех, что мы пробовали. Но валидация устроена удобно: у каждого запроса с парой
в галерее лежит кадр той же машины с той же камеры, и косинус до него около 0.91. В закрытом
тесте гарантирована только кросс-камерная пара, и всё распределение уверенности сдвигается
вниз примерно на 0.4. Абсолютный порог по абсолютной величине — ровно то, что от такого
сдвига страдает сильнее всего.

Отсюда гипотеза: мера, устроенная как РАЗНОСТЬ внутри одного запроса, к общему сдвигу
устойчивее, чем сама величина. Проверяется это здесь, на тех же двух сценариях, что и порог:

  A — как сейчас: одно-камерные кадры в галерее есть;
  B — стресс: у каждого запроса убраны кадры своей машины со своей камеры; чужие машины
      с этой камеры остаются, то есть трудные негативы никуда не делись.

Все меры считаются по одному запросу и его галерее — других запросов не касаются (ответ 38).
Мера `bits` сюда не входит сознательно: она нормируется на размер галереи, который на
валидации 688, а на тесте 750, и потому непереносима.

    python scripts/confidence_stress.py --run runs/hack/ft_soup_b336_fit --out results/confidence_stress.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vreid.rerank import frame_block_mask   # noqa: E402


def confidences(raw: np.ndarray, top: int = 10) -> dict:
    """Меры уверенности по строке сходств одного запроса. Все — функции только этой строки."""
    order = np.argsort(-raw, axis=1, kind="stable")
    rows = np.arange(raw.shape[0])[:, None]
    ranked = raw[rows, order[:, :max(top, 2)]]
    first, second = ranked[:, 0], ranked[:, 1]
    rest = ranked[:, 1:]
    with np.errstate(invalid="ignore"):
        return {
            "top1": first,
            "margin": first - second,
            "gap_to_rest": first - np.nanmean(np.where(np.isfinite(rest), rest, np.nan), axis=1),
            "ratio": first / np.where(np.abs(second) < 1e-6, 1e-6, second),
        }, order[:, 0]


def outcome(confidence: np.ndarray, threshold: float, has: np.ndarray,
            correct: np.ndarray) -> dict:
    accepted = confidence >= threshold
    tp = int((accepted & has & correct).sum())
    fp = int((accepted & ~(has & correct)).sum())
    fn = int((~accepted & has).sum())
    tn = int((~accepted & ~has).sum())
    negatives = int((~has).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    tnr = tn / max(negatives, 1)
    return {"score": 100 * (0.7 * f1 + 0.3 * tnr), "f1": 100 * f1, "tnr": 100 * tnr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def sweep(values: np.ndarray, has: np.ndarray, correct: np.ndarray, points: int = 160):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    grid = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, points)))
    return [(float(t), outcome(values, float(t), has, correct)) for t in grid]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/hack/ft_soup_b336_fit")
    parser.add_argument("--beliefs", type=float, nargs="+", default=[0.7, 0.85, 0.95])
    parser.add_argument("--out", type=Path, default=ROOT / "results/confidence_stress.json")
    args = parser.parse_args()

    with np.load(args.run / "val_query_track.npz", allow_pickle=False) as data:
        query = {k: data[k] for k in ("emb", "vids", "cams", "keys")}
    with np.load(args.run / "val_gallery_track.npz", allow_pickle=False) as data:
        gallery = {k: data[k] for k in ("emb", "vids", "cams", "keys")}

    blocked = frame_block_mask(query["keys"], gallery["keys"])
    raw_a = query["emb"] @ gallery["emb"].T
    raw_a[blocked] = -np.inf
    same_vehicle = gallery["vids"][None, :] == query["vids"][:, None]
    same_camera = gallery["cams"][None, :] == query["cams"][:, None]
    raw_b = np.where(same_vehicle & same_camera, -np.inf, raw_a)

    has = {"A": same_vehicle.any(axis=1), "B": (same_vehicle & ~same_camera).any(axis=1)}
    measures, correct = {}, {}
    for name, raw in (("A", raw_a), ("B", raw_b)):
        measures[name], selected = confidences(raw)
        correct[name] = gallery["vids"][selected] == query["vids"]

    report = {"run": str(args.run), "n_query": int(len(query["keys"])),
              "note": "порог калибруется на A и применяется к B — так и происходит в жизни",
              "measures": {}}

    print(f"{'мера':>14}{'порог на A':>12}{'A':>8}{'B':>8}{'минимакс':>10}"
          + "".join(f"{'E p=' + str(p):>10}" for p in args.beliefs))
    print("-" * (14 + 12 + 8 + 8 + 10 + 10 * len(args.beliefs)))
    for name in measures["A"]:
        rows_a = sweep(measures["A"][name], has["A"], correct["A"])
        if not rows_a:
            continue
        # Порог выбирается ТОЛЬКО по сценарию A: закрытого теста у нас нет, и делать вид,
        # что мы можем настроиться на B, значило бы измерять несуществующее знание.
        threshold, best_a = max(rows_a, key=lambda row: row[1]["score"])
        best_b = outcome(measures["B"][name], threshold, has["B"], correct["B"])
        worst = min(best_a["score"], best_b["score"])
        expectations = {f"E{p}": p * best_a["score"] + (1 - p) * best_b["score"]
                        for p in args.beliefs}
        # Честное сравнение с нашим решением. Порог 0.55 в релизе выбран НЕ по максимуму A,
        # а по максимуму ожидания при вере p=0.7 в то, что дубликаты будут. Значит и другие
        # меры надо брать с их лучшим ожиданием, иначе мы сравниваем своё взвешенное решение
        # с чужим жадным и делаем вид, что победили.
        by_expectation = {}
        for belief in args.beliefs:
            scored = [(row[0], row[1]["score"],
                       outcome(measures["B"][name], row[0], has["B"], correct["B"])["score"])
                      for row in rows_a]
            point = max(scored, key=lambda item: belief * item[1] + (1 - belief) * item[2])
            by_expectation[f"E{belief}"] = {
                "threshold": point[0], "A": point[1], "B": point[2],
                "expectation": belief * point[1] + (1 - belief) * point[2]}

        report["measures"][name] = {
            "threshold_on_A": threshold, "A": best_a, "B": best_b, "worst": worst,
            **expectations,
            "tuned_by_expectation": by_expectation,
            "median_A": float(np.median(measures["A"][name][np.isfinite(measures["A"][name])])),
            "median_B": float(np.median(measures["B"][name][np.isfinite(measures["B"][name])])),
        }
        print(f"{name:>14}{threshold:12.4f}{best_a['score']:8.2f}{best_b['score']:8.2f}"
              f"{worst:10.2f}" + "".join(f"{expectations['E' + str(p)]:10.2f}" for p in args.beliefs))

    print(f"\n{'мера':>14}{'порог под E0.7':>16}{'A':>8}{'B':>8}{'ожидание':>11}")
    print("-" * 57)
    for name, block in report["measures"].items():
        tuned = block["tuned_by_expectation"]["E0.7"]
        print(f"{name:>14}{tuned['threshold']:16.4f}{tuned['A']:8.2f}{tuned['B']:8.2f}"
              f"{tuned['expectation']:11.2f}")
    ranking = sorted(report["measures"].items(),
                     key=lambda kv: -kv[1]["tuned_by_expectation"]["E0.7"]["expectation"])
    report["best_by_expectation_p0.7"] = ranking[0][0]
    report["current_in_release"] = "top1"
    print(f"\nлучшая по ожиданию при p=0.7 и своём пороге: {ranking[0][0]}; в релизе: top1")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[confidence] → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
