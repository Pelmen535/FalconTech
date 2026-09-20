"""Подбор параметров k-reciprocal с честным протоколом «выбрали на одном, проверили на другом».

Зачем. Ре-ранжирование не входит в замер латентности жюри (ответ 31), то есть параметры
k1/k2/lambda ничего не стоят по производительности. Прежняя сетка была грубой — 6/2, 10/3,
15/4, 20/6 — и остановилась на первой же строке. Это подозрительно похоже на «взяли край
диапазона», и стоит проверить, нет ли рядом лучшего.

Почему это не подгонка под ответ. Параметры выбираются на НАШЕЙ размеченной валидации,
а не на закрытом тесте; правила запрещают подбор по закрытым ответам, а не по своим данным
(ответ 38). Чтобы выбор не оказался подгонкой под конкретный сплит, сетка считается на двух
независимых кешах — двойника релиза (fit, личности модель не видела) и старой валидационной
модели (val) — и в отчёт идут обе колонки. Если лучшая точка на одном кеше проваливается на
другом, значит это шум, а не улучшение, и рецепт не меняется.

    python scripts/rerank_grid.py --out results/rerank_grid.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vreid.metrics import evaluate           # noqa: E402
from vreid.rerank import frame_block_mask, k_reciprocal   # noqa: E402


def load_pack(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("emb", "vids", "cams", "keys")}


def score_map10(pack_q: dict, pack_g: dict, rank: np.ndarray) -> dict:
    return evaluate(rank, pack_q["vids"], pack_q["cams"], pack_g["vids"], pack_g["cams"],
                    cutoff=10)


def run_grid(run_dir: Path, grid) -> dict:
    query = load_pack(run_dir / "val_query_track.npz")
    gallery = load_pack(run_dir / "val_gallery_track.npz")
    blocked = frame_block_mask(query["keys"], gallery["keys"])

    raw = np.stack([gallery["emb"] @ row for row in query["emb"]])
    raw[blocked] = -np.inf
    out = {"base": score_map10(query, gallery, raw)}

    for k1, k2, lam in grid:
        distance = k_reciprocal(query["emb"], gallery["emb"], k1=k1, k2=k2, lam=lam,
                                same_mask=None, shared=False, isolate_queries=True)
        rank = -distance
        rank[blocked] = -np.inf
        out[f"{k1}/{k2}/{lam:g}"] = score_map10(query, gallery, rank)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+",
                        default=[ROOT / "runs/hack/ft_soup_b336_fit",
                                 ROOT / "runs/hack/ft_soup_b336_val"])
    parser.add_argument("--k1", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 10, 12])
    parser.add_argument("--k2", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--lam", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    parser.add_argument("--out", type=Path, default=ROOT / "results/rerank_grid.json")
    args = parser.parse_args()

    grid = [(k1, k2, lam) for k1 in args.k1 for k2 in args.k2 for lam in args.lam if k2 <= k1]
    report = {"grid_size": len(grid), "runs": {}}
    for run_dir in args.runs:
        if not (run_dir / "val_query_track.npz").is_file():
            print(f"[grid] пропускаю {run_dir}: нет кеша")
            continue
        print(f"[grid] {run_dir.name}: {len(grid)} точек", flush=True)
        report["runs"][run_dir.name] = {name: round(100 * value["mAP"], 3)
                                        for name, value in run_grid(run_dir, grid).items()}

    names = list(report["runs"])
    if len(names) >= 2:
        primary, secondary = names[0], names[1]
        shared = [key for key in report["runs"][primary] if key in report["runs"][secondary]]
        best_primary = max(shared, key=lambda key: report["runs"][primary][key])
        best_secondary = max(shared, key=lambda key: report["runs"][secondary][key])
        current = "6/2/0.3"
        report["verdict"] = {
            "best_on_" + primary: [best_primary, report["runs"][primary][best_primary],
                                   report["runs"][secondary].get(best_primary)],
            "best_on_" + secondary: [best_secondary, report["runs"][secondary][best_secondary],
                                     report["runs"][primary].get(best_secondary)],
            "current_recipe": [current, report["runs"][primary].get(current),
                               report["runs"][secondary].get(current)],
            "agree": best_primary == best_secondary,
        }
        print(f"[grid] лучшее на {primary}: {best_primary} = "
              f"{report['runs'][primary][best_primary]}, на {secondary} даёт "
              f"{report['runs'][secondary].get(best_primary)}")
        print(f"[grid] лучшее на {secondary}: {best_secondary} = "
              f"{report['runs'][secondary][best_secondary]}, на {primary} даёт "
              f"{report['runs'][primary].get(best_secondary)}")
        print(f"[grid] в рецепте {current}: {report['runs'][primary].get(current)} / "
              f"{report['runs'][secondary].get(current)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[grid] → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
