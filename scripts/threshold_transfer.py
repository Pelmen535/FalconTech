# -*- coding: utf-8 -*-
"""Переносится ли порог отказа с модели, на которой он калиброван, на релизную модель?

    python scripts/threshold_transfer.py --data Данные \
        --cal weights/soup_b336_val/best.pt --rel weights/soup_b336_all/best.pt \
        --release release --out results/threshold_transfer.json

ЗАЧЕМ. Порог 0.6011 выбран по валидации на модели soup_b336_val — она обучена на 80%
личностей, а 20% отложены, поэтому на них честно меряется и mAP, и режим отказа.
В релиз ушла soup_b336_all — та же рецептура, но обучена на ВСЕХ личностях. Прямо
померить её порог не на чем: валидационные личности она видела. Законное возражение:
«у двух моделей может быть разная шкала косинуса, и порог с одной на другую не переносится».

КАК ПРОВЕРЯЕМ, НЕ ТРОГАЯ ОТВЕТОВ. Организаторы подтвердили, что личности train и test
не пересекаются (ответ 7). Значит ОТКРЫТЫЙ тест — вне обучения для ОБЕИХ моделей, то есть
общая нейтральная площадка. Меряем на нём распределение уверенности (косинус топ-1 по сырым
векторам, ровно как в бою) у обеих моделей и сравниваем:
  * смещение распределений (медиана, квантили, корреляция по запросам);
  * долю отказов при одном и том же пороге;
  * порог для релизной модели, дающий ТУ ЖЕ долю отказов, что порог калибровки даёт
    у калибровочной модели, — и сколько решений от замены порога перевернётся.
Метки не используются вообще, ни одного ответа не читается, порог по итогу не подбирается
по тесту — это диагностика переноса, а не калибровка.

ЧТО ЗНАЧИТ РЕЗУЛЬТАТ. Это диагностика распределений и acceptance на открытом тесте.
Когда все query имеют пару, acceptance ограничивает recall сверху, но не определяет его:
принятый кандидат может быть неправильным. TNR на наборе без no-match не измеряется.
По этой диагностике нельзя автоматически менять production threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vreid.datasets import Split                       # noqa: E402
from vreid.extract import extract_split                # noqa: E402
from vreid.hackathon_data import read_annotations      # noqa: E402
from vreid.models import get_backbone                  # noqa: E402
from vreid.refusal import confidence_scores            # noqa: E402
from vreid.rerank import frame_block_mask              # noqa: E402


def top1_conf(ckpt: str, q: Split, g: Split, pad: float, fast: bool, bs: int, workers: int,
              tau: float | None) -> np.ndarray:
    bb = get_backbone("ft:" + ckpt)
    ex = lambda s: extract_split(s, bb, bs, workers, None, pad=pad, mask=None,
                                 tta_flip=False, fast_decode=fast)["emb"]
    qe, ge = ex(q), ex(g)
    block = frame_block_mask(list(q.keys), list(g.keys))
    raw = qe @ ge.T
    raw[block] = -np.inf
    sc = confidence_scores(raw, np.argsort(-raw, axis=1, kind="stable"), tau)
    return sc["top1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Данные")
    ap.add_argument("--cal", default="weights/soup_b336_val/best.pt", help="модель, на которой калиброван порог")
    ap.add_argument("--rel", default="weights/soup_b336_all/best.pt", help="модель, ушедшая в релиз")
    ap.add_argument("--release", default="release", help="откуда взять recipe.json (порог, pad, tau)")
    ap.add_argument("--val", default="results/hack_ft_soup_b336_val_val.json",
                    help="файл валидации, из которого взят порог (для recall на валидации)")
    ap.add_argument("--out", default="results/threshold_transfer.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    data = Path(a.data)
    rec = json.loads((Path(a.release) / "recipe.json").read_text(encoding="utf-8"))
    thr = float(rec["threshold"])
    images = data / "images"
    q = Split("query", read_annotations(data / "test_query.csv", images, None, "xywh"))
    g = Split("gallery", read_annotations(data / "test_gallery.csv", images, None, "xywh"))
    print(f"[transfer] открытый тест: запросов {len(q)}, галерея {len(g)}; порог из релиза {thr:.4f}")

    kw = dict(pad=float(rec.get("crop_pad", 0.05)), fast=bool(rec.get("fast_decode", True)),
              bs=a.batch_size, workers=a.workers, tau=rec.get("tau"))
    print(f"[transfer] модель калибровки: {a.cal}")
    c_cal = top1_conf(a.cal, q, g, **kw)
    print(f"[transfer] модель релиза:    {a.rel}")
    c_rel = top1_conf(a.rel, q, g, **kw)

    qs = [1, 5, 10, 20, 50, 80, 95]
    rows = []
    for name, c in (("калибровочная", c_cal), ("релизная", c_rel)):
        rows.append({"model": name, "mean": float(c.mean()), "median": float(np.median(c)),
                     "q": {str(p): float(np.percentile(c, p)) for p in qs},
                     "refuse_rate": float((c < thr).mean())})
    print(f"\n{'модель':<16}{'среднее':>9}{'медиана':>9}" + "".join(f"{'p' + str(p):>8}" for p in qs) + f"{'отказов':>10}")
    print("-" * (16 + 18 + 8 * len(qs) + 10))
    for r in rows:
        print(f"{r['model']:<16}{r['mean']:9.4f}{r['median']:9.4f}"
              + "".join(f"{r['q'][str(p)]:8.4f}" for p in qs)
              + f"{r['refuse_rate'] * 100:9.1f}%")

    # Сдвиг шкалы и согласованность решений
    shift_med = float(np.median(c_rel) - np.median(c_cal))
    pear = float(np.corrcoef(c_cal, c_rel)[0, 1])
    ranks = lambda x: np.argsort(np.argsort(x))
    spear = float(np.corrcoef(ranks(c_cal), ranks(c_rel))[0, 1])
    # Порог для релизной модели, дающий ту же ДОЛЮ отказов, что порог калибровки даёт у
    # калибровочной модели. Это не наш рабочий порог, а мера сдвига в единицах решений.
    rate_cal = float((c_cal < thr).mean())
    thr_matched = float(np.quantile(c_rel, rate_cal)) if rate_cal > 0 else float("-inf")
    flip = int(((c_rel < thr) != (c_rel < thr_matched)).sum())
    print(f"\nсдвиг медианы релизной модели к калибровочной: {shift_med:+.4f}")
    print(f"корреляция уверенностей по запросам: Пирсон {pear:.3f}, Спирмен {spear:.3f}")
    print(f"порог, дающий релизной модели ту же долю отказов ({rate_cal * 100:.1f}%): {thr_matched:.4f} "
          f"(отличие от рабочего {thr_matched - thr:+.4f}, перевернулось бы решений: {flip} из {len(c_rel)})")

    val_recall = None
    vp = Path(a.val)
    if vp.exists():
        v = json.loads(vp.read_text(encoding="utf-8"))
        ch = v["refusal"]["by_confidence"][rec["confidence"]]["chosen"]
        val_recall = float(ch["recall"])
        acc = float((c_rel >= thr).mean())
        print(f"\nВ открытом тесте у каждого запроса есть пара (ответ 17), поэтому доля принятых —")
        print(f"это coverage (верхняя граница recall при всех matched): релизная модель принимает {acc * 100:.1f}%, "
              f"а на валидации recall был {val_recall * 100:.1f}%.")
        print("Эти величины имеют разный смысл; разность не доказывает перенос recall или TNR.")

    out = {"threshold": thr, "n_query": int(len(q)), "n_gallery": int(len(g)),
           "cal_checkpoint": a.cal, "rel_checkpoint": a.rel, "rows": rows,
           "median_shift": shift_med, "pearson": pear, "spearman": spear,
           "rate_matched_threshold": thr_matched, "decisions_flipped": flip,
           "accept_rate_release_open_test": float((c_rel >= thr).mean()),
           "val_recall": val_recall, "recall_transfer_verified": False, "tnr_transfer_verified": False}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[transfer] → {a.out}")


if __name__ == "__main__":
    main()
