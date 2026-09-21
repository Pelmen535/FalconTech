"""Выборочная проверка: стоит ли запускать тяжёлый ре-ранкер только там, где первая ступень
не уверена.

Зачем спрашивали. Каскад даёт +4.3 п.п. mAP@10, но добавляет форвард ViT-L к каждому запросу.
Если бы тяжёлая проверка нужна была лишь меньшинству запросов, средняя и медианная цена
запроса почти не выросла бы — а жюри мерит именно медиану 300 прогонов. Это сняло бы
основной риск каскада (docs/SELF_REVIEW.md, таблица двух сценариев).

Ответ: не работает. При проверке 46% самых «неуверенных» запросов прирост 74.21 → 74.94,
то есть шестая часть от полного. Почти весь выигрыш приходит от запросов, где первая ступень
уверена — и уверенно ошибается: два одинаковых автомобиля дают высокий косинус к чужой машине,
запас между первым и вторым кандидатом большой, а ответ неверный. Запас уверенности не
отделяет исправимые ошибки от неисправимых, поэтому экономить на проверке не получается.

Порог здесь АБСОЛЮТНЫЙ (запас косинуса между первым и вторым кандидатом), а не квантиль по
тесту: квантиль связал бы запросы между собой, что запрещено (ответ 38).

    python scripts/cascade_selective.py --out results/cascade_selective.json
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vreid.cascade import heavy_scores, rescore, shortlist
from vreid.metrics import evaluate
from vreid.rerank import frame_block_mask, k_reciprocal

TOPK, ALPHA = 30, 0.9
THRESHOLDS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0)
REPORT = {"question": "хватит ли выборочной тяжёлой проверки вместо сплошной",
          "answer": "нет: при 46% проверенных запросов прирост шестая часть от полного",
          "runs": {}}


def pack(run, name):
    with np.load(ROOT / "runs/hack" / run / f"val_{name}_track.npz", allow_pickle=False) as d:
        return {k: d[k] for k in ("emb", "vids", "cams", "keys")}


def jury(rank, q, g):
    m = evaluate(rank, q["vids"], q["cams"], g["vids"], g["cams"], cutoff=10)
    return 100 * m["mAP"], 100 * m["rank1"]


for run in ("ft_soup_b336_fit", "ft_soup_b336_val"):
    q, g = pack(run, "query"), pack(run, "gallery")
    qh, gh = pack("ft_hack_dinov2_l_336_cam", "query"), pack("ft_hack_dinov2_l_336_cam", "gallery")
    blocked = frame_block_mask(q["keys"], g["keys"])

    raw = (q["emb"] @ g["emb"].T).astype(np.float32)
    raw[blocked] = -np.inf
    base = -k_reciprocal(q["emb"], g["emb"], k1=3, k2=2, lam=0.2, same_mask=None, shared=False,
                         isolate_queries=True)
    base[blocked] = -np.inf

    # запас первой ступени: разница сырых косинусов первого и второго кандидата её порядка
    order_base = np.argsort(-base, axis=1, kind="stable")
    rows = np.arange(len(q["emb"]))
    margin = raw[rows, order_base[:, 0]] - raw[rows, order_base[:, 1]]

    order = shortlist(base, TOPK)
    heavy = heavy_scores(qh["emb"], gh["emb"], order, blocked)
    full = rescore(base, order, heavy, ALPHA)

    print(f"--- {run}: без каскада {jury(base, q, g)[0]:.3f}, каскад везде {jury(full, q, g)[0]:.3f}")
    rows = []
    for thr in THRESHOLDS:
        use = margin < thr
        mixed = np.where(use[:, None], full, base)
        mAP, r1 = jury(mixed, q, g)
        rows.append({"margin_threshold": thr, "share_verified": round(100 * float(use.mean()), 2),
                     "mAP": round(mAP, 3), "rank1": round(r1, 3)})
        print(f"  порог запаса {thr:4.2f}: тяжёлая проверка у {100 * use.mean():5.1f}% запросов "
              f"→ mAP@10 {mAP:7.3f}  rank1 {r1:7.3f}")
    REPORT["runs"][run] = {"no_cascade": round(jury(base, q, g)[0], 3),
                           "cascade_everywhere": round(jury(full, q, g)[0], 3), "rows": rows}

import argparse
import json

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, default=ROOT / "results/cascade_selective.json")
args = parser.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[selective] отчёт → {args.out}")
