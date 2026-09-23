"""Ворота в очереди: лучше ли новая модель прежней настолько, чтобы тратить на неё следующие часы.

Зачем. Очередь «сильный учитель → ученики → релиз» идёт полсуток. Если учитель на 392
окажется не лучше прежнего, учить от него трёх учеников — это пять часов впустую; если ученики
не обгонят опубликованный релиз, собирать новый — ещё пять. Очередь спрашивает этот скрипт и
идёт дальше только при коде 0.

Меряет ровно то же, что scripts/cascade_verify.py (и что на двойниках совпало с эталонным
evaluate.py организаторов до сотых): mAP@10 по правилу жюри на кеше эмбеддингов отложенной
валидации. Режим raw - сырой косинус, для учителя: у него нет своей постобработки в релизе.
Режим kr - k-reciprocal 3/2/0.2, как в релизе, для учеников.

    python scripts/gate_better.py --candidate runs/hack/ft_l392cam_fit \\
        --baseline runs/hack/ft_l336cam_fit --mode raw --min-gain 0.3

Коды: 0 - лучше на min-gain и больше; 3 - не лучше; 1 - ошибка (нет кеша и т.п.).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vreid.metrics import evaluate                       # noqa: E402
from vreid.rerank import frame_block_mask, k_reciprocal  # noqa: E402


def score(run: Path, mode: str) -> tuple[float, float]:
    files = [run / "val_query_track.npz", run / "val_gallery_track.npz"]
    if not all(f.is_file() for f in files):
        raise SystemExit(f"[gate] нет кеша эмбеддингов в {run}")
    with np.load(files[0], allow_pickle=False) as data:
        q = {k: data[k] for k in ("emb", "vids", "cams", "keys")}
    with np.load(files[1], allow_pickle=False) as data:
        g = {k: data[k] for k in ("emb", "vids", "cams", "keys")}
    blocked = frame_block_mask(q["keys"], g["keys"])
    if mode == "raw":
        unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
        rank = (unit(q["emb"]) @ unit(g["emb"]).T).astype(np.float32)
    else:
        rank = -k_reciprocal(q["emb"], g["emb"], k1=3, k2=2, lam=0.2, same_mask=None,
                             shared=False, isolate_queries=True)
    rank[blocked] = -np.inf
    metric = evaluate(rank, q["vids"], q["cams"], g["vids"], g["cams"], cutoff=10)
    return 100 * metric["mAP"], 100 * metric["rank1"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw", "kr"), default="kr")
    parser.add_argument("--min-gain", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cand, base = ROOT / args.candidate, ROOT / args.baseline
    c_map, c_r1 = score(cand, args.mode)
    b_map, b_r1 = score(base, args.mode)
    gain = c_map - b_map
    passed = gain >= args.min_gain
    report = {"mode": args.mode, "candidate": str(args.candidate), "baseline": str(args.baseline),
              "candidate_map10": round(c_map, 3), "candidate_rank1": round(c_r1, 3),
              "baseline_map10": round(b_map, 3), "baseline_rank1": round(b_r1, 3),
              "gain": round(gain, 3), "min_gain": args.min_gain, "passed": passed}
    print(f"[gate] {args.mode}: {cand.name} {c_map:.2f} против {base.name} {b_map:.2f} -> "
          f"{gain:+.2f} (порог {args.min_gain:+.2f}) -> {'ДАЛЬШЕ' if passed else 'СТОП'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
