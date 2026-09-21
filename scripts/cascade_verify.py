"""Каскад «лёгкая модель ищет — тяжёлая проверяет топ-K»: измерение выигрыша по метрике жюри.

Зачем. Балл за точность у нас самый слабый, а упирается он в размер бэкбона: ViT-L даёт на
той же валидации на 5-6 пунктов больше ViT-B, но втрое дороже на форварде, и сдавать его как
основную модель — минус около 12 баллов производительности (расчёт в docs/SELF_REVIEW.md).

Почему каскад — другое дело. Правила считают ре-ранжирование отдельно от инференса:
  * «latency_b1 включает чтение и декодирование, crop по BBox, preprocessing, forward,
    postprocessing и L2-нормализацию. Поиск по gallery и rerank НЕ входят» (ответы 31, 32);
  * лимит 2 ГБ относится к сумме весов «основного backbone, auxiliary-моделей,
    классификаторов, ensemble и reranker» (ответы 34, 37) — то есть отдельная модель-ре-ранкер
    правилами прямо предусмотрена;
  * «submission.csv может быть результатом per-query каскада: cosine shortlist,
    pairwise/local-feature проверка или per-query k-reciprocal» (ответ 28);
  * embeddings.npy у организаторов идёт только в справочный раздел, балл считается по
    submission.csv — расхождение порядка с косинусом по embeddings.npy штатно.

Что здесь считается. Шортлист строится ровно как в релизе (косинус ViT-B + k-reciprocal
3/2/0.2), затем топ-K пересортировывается по сходству ViT-L, и меряется mAP@10 жюри.
Протокол честный: вес слияния выбирается на одном кеше и проверяется на другом.

    python scripts/cascade_verify.py --out results/cascade_verify.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vreid.cascade import heavy_scores, rescore, shortlist  # noqa: E402
from vreid.metrics import evaluate                          # noqa: E402
from vreid.rerank import frame_block_mask, k_reciprocal     # noqa: E402

ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def load_pack(run_dir: Path, name: str) -> dict:
    with np.load(run_dir / f"val_{name}_track.npz", allow_pickle=False) as data:
        return {key: data[key] for key in ("emb", "vids", "cams", "keys")}


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def jury(rank: np.ndarray, q: dict, g: dict) -> dict:
    metric = evaluate(rank, q["vids"], q["cams"], g["vids"], g["cams"], cutoff=10)
    return {"mAP": round(100 * metric["mAP"], 3), "rank1": round(100 * metric["rank1"], 3),
            "rank5": round(100 * metric["rank5"], 3)}


def shortlist_recall(order: np.ndarray, q: dict, g: dict, blocked: np.ndarray) -> float:
    """Доля запросов с валидным позитивом, у которых позитив попал в шортлист.
    Это потолок каскада: чего нет в шортлисте, того не вернёт никакая проверка."""
    hit = total = 0
    for i in range(order.shape[0]):
        same = (g["vids"] == q["vids"][i]) & (g["cams"] != q["cams"][i]) & ~blocked[i]
        if not same.any():
            continue
        total += 1
        hit += bool(same[order[i]].any())
    return round(100 * hit / max(total, 1), 3)


def run_one(light_dir: Path, heavy_dir: Path, args) -> dict:
    q_light, g_light = load_pack(light_dir, "query"), load_pack(light_dir, "gallery")
    q_heavy, g_heavy = load_pack(heavy_dir, "query"), load_pack(heavy_dir, "gallery")
    if not (np.array_equal(q_light["keys"], q_heavy["keys"])
            and np.array_equal(g_light["keys"], g_heavy["keys"])):
        raise SystemExit(f"{light_dir.name} и {heavy_dir.name} сняты на разных сплитах — "
                         f"сравнивать их построчно нельзя")

    blocked = frame_block_mask(q_light["keys"], g_light["keys"])
    base = -k_reciprocal(q_light["emb"], g_light["emb"], k1=args.k1, k2=args.k2, lam=args.lam,
                         same_mask=None, shared=False, isolate_queries=True)
    base[blocked] = -np.inf

    order = shortlist(base, args.topk)
    heavy_full = (unit(q_heavy["emb"]) @ unit(g_heavy["emb"]).T).astype(np.float32)
    heavy_full[blocked] = -np.inf
    heavy = heavy_scores(q_heavy["emb"], g_heavy["emb"], order, blocked)

    out = {"topk": args.topk,
           "shortlist_recall": shortlist_recall(order, q_light, g_light, blocked),
           "light_only": jury(base, q_light, g_light),
           "heavy_only_full_gallery": jury(heavy_full, q_light, g_light),
           "mixed": {}}
    for alpha in ALPHAS:
        out["mixed"][f"{alpha:g}"] = jury(rescore(base, order, heavy, alpha), q_light, g_light)
    print(f"[cascade] {light_dir.name} + {heavy_dir.name}: recall@{args.topk} "
          f"{out['shortlist_recall']}, лёгкая {out['light_only']['mAP']}, "
          f"тяжёлая по всей галерее {out['heavy_only_full_gallery']['mAP']}", flush=True)
    for alpha, value in out["mixed"].items():
        print(f"    alpha={alpha:4s} mAP@10 {value['mAP']:7.3f}  rank1 {value['rank1']:7.3f}",
              flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--light", nargs="+", default=["runs/hack/ft_soup_b336_fit",
                                                       "runs/hack/ft_soup_b336_val"])
    parser.add_argument("--heavy", nargs="+", default=["runs/hack/ft_hack_dinov2_l_336_cam",
                                                       "runs/hack/ft_hack_dinov2_l_336"])
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--k1", type=int, default=3)
    parser.add_argument("--k2", type=int, default=2)
    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=ROOT / "results/cascade_verify.json")
    args = parser.parse_args()

    report = {"protocol": "метрика жюри mAP@10; alpha выбирается на одном кеше, проверяется на другом",
              "rules": "ответы 28/31/32/34/37: rerank вне латентности, веса ре-ранкера — в лимит 2 ГБ",
              "params": {"topk": args.topk, "rerank": [args.k1, args.k2, args.lam]},
              "pairs": {}}
    for light in args.light:
        for heavy in args.heavy:
            light_dir, heavy_dir = ROOT / light, ROOT / heavy
            if not (light_dir / "val_query_track.npz").is_file():
                print(f"[cascade] пропускаю {light}: нет кеша")
                continue
            if not (heavy_dir / "val_query_track.npz").is_file():
                print(f"[cascade] пропускаю {heavy}: нет кеша")
                continue
            report["pairs"][f"{light_dir.name}|{heavy_dir.name}"] = run_one(light_dir, heavy_dir, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cascade] отчёт → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
