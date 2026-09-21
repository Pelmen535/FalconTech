"""Локальная (патчевая) проверка кандидатов поверх релизного ранжирования.

Зачем. Глобальный вектор путает одинаковые машины одного цвета; различают их детали —
наклейка, диск, вмятина. `vreid/local_match.py` умеет считать, сколько патчей совпало у пары
кропов с согласованным сдвигом по сетке, но в релизе это не включено и никогда не было
измерено на метрике жюри.

Почему это законно и бесплатно. Ответ 28 прямо разрешает per-query каскад «cosine shortlist →
pairwise/local-feature проверка»: решение по запросу принимается по нему одному, другие
запросы на него не влияют. В замер латентности ре-ранжирование не входит (ответ 31), так что
по производительности это ноль.

Протокол честный, как в scripts/rerank_grid.py: beta подбирается на одном кеше и проверяется
на другом. Если лучшая beta на одном проваливается на другом — это шум, и рецепт не меняется.

    python scripts/local_verify.py --out results/local_verify.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vreid.hackathon_data import build_local_validation          # noqa: E402
from vreid.local_match import LocalFeatures, fuse, local_scores  # noqa: E402
from vreid.metrics import evaluate                               # noqa: E402
from vreid.models import get_backbone                            # noqa: E402
from vreid.rerank import frame_block_mask, k_reciprocal          # noqa: E402

BETAS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)


def load_pack(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("emb", "vids", "cams", "keys")}


def jury(rank: np.ndarray, q: dict, g: dict) -> dict:
    return evaluate(rank, q["vids"], q["cams"], g["vids"], g["cams"], cutoff=10)


def valid_order(rank: np.ndarray, topk: int) -> np.ndarray:
    """Топ-K индексов галереи; -1 там, где кандидат закрыт маской (см. hack_cli._valid_order)."""
    order = np.argsort(-rank, axis=1)[:, :topk]
    rows = np.arange(rank.shape[0])[:, None]
    return np.where(np.isfinite(rank[rows, order]), order, -1)


def tokens_for(run_dir: Path, weights: str, splits: dict, args):
    """Токены патчей для query и gallery; кладутся рядом с кешем эмбеддингов.

    Базис PCA подгоняется на ПЕРВОМ сплите (local_match.py) — поэтому query всегда первым,
    а галерея проецируется в тот же базис. Переставить вызовы = другие токены."""
    q_cache = run_dir / f"val_query_local{args.size}.npz"
    g_cache = run_dir / f"val_gallery_local{args.size}.npz"
    if q_cache.is_file() and g_cache.is_file():
        with np.load(q_cache, allow_pickle=False) as data:
            q_tok, grid = data["tokens"], data["grid"]
        with np.load(g_cache, allow_pickle=False) as data:
            g_tok = data["tokens"]
        return q_tok, g_tok, grid
    feats = LocalFeatures(get_backbone(weights, device=args.device), size=args.size,
                          pca_dim=args.pca)
    got_q = feats.extract(splits["query"], q_cache, pad=args.pad,
                          batch_size=args.batch_size, workers=args.workers)
    got_g = feats.extract(splits["gallery"], g_cache, pad=args.pad,
                          batch_size=args.batch_size, workers=args.workers)
    return got_q["tokens"], got_g["tokens"], got_q["grid"]


def run_one(run_dir: Path, weights: str, splits: dict, args) -> dict:
    query = load_pack(run_dir / "val_query_track.npz")
    gallery = load_pack(run_dir / "val_gallery_track.npz")
    blocked = frame_block_mask(query["keys"], gallery["keys"])

    cosine = (query["emb"] @ gallery["emb"].T).astype(np.float32)
    cosine[blocked] = -np.inf
    distance = k_reciprocal(query["emb"], gallery["emb"], k1=args.k1, k2=args.k2, lam=args.lam,
                            same_mask=None, shared=False, isolate_queries=True)
    reranked = -distance
    reranked[blocked] = -np.inf

    q_tok, g_tok, grid = tokens_for(run_dir, weights, splits, args)
    n_tokens = int(grid[0] * grid[1])

    out = {"n_query": int(len(query["vids"])), "n_gallery": int(len(gallery["vids"])),
           "grid": [int(grid[0]), int(grid[1])], "bases": {}, "inliers": {}}
    for base_name, base in (("cosine", cosine), ("kr", reranked)):
        order = valid_order(base, args.topk)
        inliers = local_scores(q_tok, g_tok, order, grid, topk=args.topk, min_sim=args.min_sim)
        row = {}
        for beta in BETAS:
            metric = jury(fuse(base, order, inliers, beta, n_tokens), query, gallery)
            row[f"{beta:g}"] = {"mAP": round(100 * metric["mAP"], 3),
                                "rank1": round(100 * metric["rank1"], 3)}
        out["bases"][base_name] = row
        out["inliers"][base_name] = {"mean_top1": float(inliers[:, 0].mean()),
                                     "mean_rest": float(inliers[:, 1:].mean()),
                                     "share_zero": float((inliers == 0).mean())}
        print(f"[local] {run_dir.name} / {base_name}: " +
              "  ".join(f"b={b} {row[b]['mAP']:.2f}" for b in row), flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["runs/hack/ft_soup_b336_fit",
                                                      "runs/hack/ft_soup_b336_val"])
    parser.add_argument("--weights", nargs="+", default=["ft:weights/soup_b336_fit/best.pt",
                                                         "ft:weights/soup_b336_val/best.pt"])
    parser.add_argument("--config", default="configs/hackathon.yaml")
    parser.add_argument("--size", type=int, default=336)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--pca", type=int, default=64)
    parser.add_argument("--min-sim", type=float, default=0.6)
    parser.add_argument("--pad", type=float, default=None, help="по умолчанию из конфига")
    parser.add_argument("--k1", type=int, default=3)
    parser.add_argument("--k2", type=int, default=2)
    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results/local_verify.json")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.pad is None:
        args.pad = float(cfg.get("crop", {}).get("pad", 0.05))
    splits = build_local_validation(cfg["dataset"])

    report = {"protocol": "метрика жюри mAP@10, отбор beta на одном кеше и проверка на другом",
              "params": {"topk": args.topk, "size": args.size, "pca": args.pca,
                         "min_sim": args.min_sim, "pad": args.pad,
                         "rerank": [args.k1, args.k2, args.lam]},
              "runs": {}}
    for run, weights in zip(args.runs, args.weights):
        run_dir = ROOT / run
        if not (run_dir / "val_query_track.npz").is_file():
            print(f"[local] пропускаю {run}: нет кеша эмбеддингов")
            continue
        report["runs"][run_dir.name] = run_one(run_dir, weights, splits, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[local] отчёт → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
