"""CLI стенда.

  python -m vreid.cli extract --config configs/carla.yaml --backbone dinov2_s
  python -m vreid.cli eval    --config configs/carla.yaml --backbone dinov2_s
  python -m vreid.cli run     --config configs/carla.yaml --backbone dinov2_s   # extract + eval

Результаты: runs/<dataset>/<backbone>/{query,gallery}.npz и results/<dataset>_<backbone>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from . import bits as bitsmod
from .datasets import load_dataset
from .extract import extract_split, load_npz
from .index import GalleryIndex
from .metrics import evaluate, format_metrics
from .models import get_backbone


def _run_dir(cfg: dict, backbone: str) -> Path:
    return Path(cfg.get("runs_dir", "runs")) / cfg["name"] / backbone


def cmd_extract(cfg: dict, backbone_name: str, splits_wanted, batch_size: int, workers: int, device):
    splits = load_dataset(cfg["dataset"])
    bb = get_backbone(backbone_name, device=device)
    out_dir = _run_dir(cfg, backbone_name)
    for name in splits_wanted:
        if name not in splits:
            continue
        extract_split(splits[name], bb, batch_size=batch_size, num_workers=workers,
                      out=out_dir / f"{name}.npz")
    return out_dir


def cmd_eval(cfg: dict, backbone_name: str, recall: float = 0.95) -> dict:
    out_dir = _run_dir(cfg, backbone_name)
    q, g = load_npz(out_dir / "query.npz"), load_npz(out_dir / "gallery.npz")
    index = GalleryIndex(g["emb"], g["vids"], g["cams"])
    t0 = time.time()
    sims = index.all_sims(q["emb"])
    t_search = time.time() - t0

    res = {"dataset": cfg["name"], "backbone": backbone_name, "dim": int(q["emb"].shape[1]),
           "search_time_s": round(t_search, 3)}
    res["standard"] = evaluate(sims, q["vids"], q["cams"], g["vids"], g["cams"])
    res["cross_camera"] = evaluate(sims, q["vids"], q["cams"], g["vids"], g["cams"], cross_camera_only=True)
    print(format_metrics(res["standard"], "standard   "))
    print(format_metrics(res["cross_camera"], "cross-cam  "))

    try:
        cal = bitsmod.calibrate_tau(sims, q["vids"], q["cams"], g["vids"], g["cams"], recall=recall)
        res["bits"] = {"calibration": cal,
                       "table": bitsmod.bits_calibration_table(sims, q["vids"], q["cams"],
                                                               g["vids"], g["cams"], cal["tau"]),
                       "max_bits": float(np.log2(index.n))}
        print(f"bits: tau={cal['tau']:.3f} (recall {recall:.0%}), max {res['bits']['max_bits']:.1f} бит при галерее {index.n}")
        for row in res["bits"]["table"]:
            print(f"   {row['bits_lo']:5.1f}–{row['bits_hi']:5.1f} бит  n={row['n']:5d}  rank-1 = {row['rank1_acc'] * 100:5.1f}%")
    except RuntimeError as e:
        print(f"[bits] пропущено: {e}")

    _save(cfg, backbone_name, res)
    return res


def _save(cfg, backbone_name, res, suffix=""):
    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{cfg['name']}_{backbone_name}{suffix}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[save] → {path}")


def cmd_compress(cfg: dict, backbone_name: str, variants=None) -> dict:
    """Кривая размер слепка → качество на уже извлечённых эмбеддингах."""
    from .compress import DEFAULT_VARIANTS, evaluate_variants, format_table
    out_dir = _run_dir(cfg, backbone_name)
    q, g = load_npz(out_dir / "query.npz"), load_npz(out_dir / "gallery.npz")
    fit = load_npz(out_dir / "train.npz")["emb"] if (out_dir / "train.npz").exists() else g["emb"]
    rows = evaluate_variants(q["emb"], g["emb"], q["vids"], q["cams"], g["vids"], g["cams"],
                             fit_emb=fit, variants=variants or DEFAULT_VARIANTS)
    print(format_table(rows))
    res = {"dataset": cfg["name"], "backbone": backbone_name, "fit_on": "train" if (out_dir / "train.npz").exists() else "gallery",
           "variants": rows}
    _save(cfg, backbone_name, res, suffix="_compress")
    return res


def _bench_slug(name: str) -> str:
    """ft:release/model.pt → ft_release_model_pt: в имени файла Windows не терпит ':' и '/'."""
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_").replace(".", "_")


def cmd_bench(backbone_name: str, threads: int, device: str | None, batch: int = 16):
    from .bench import bench_backbone, format_bench
    bb = get_backbone(backbone_name, device=device)
    r = bench_backbone(bb, threads=threads, batch=batch)
    print(format_bench(r))
    Path("results").mkdir(exist_ok=True)
    with open(Path("results") / f"bench_{_bench_slug(backbone_name)}_t{threads}.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    return r


def main(argv=None):
    p = argparse.ArgumentParser(prog="vreid")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "eval", "run", "compress"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--backbone", default="dinov2_s")
        s.add_argument("--batch-size", type=int, default=64)
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--device", default=None)
        s.add_argument("--splits", nargs="+", default=["query", "gallery"])
        s.add_argument("--recall", type=float, default=0.95, help="recall для калибровки радиуса битов")
        s.add_argument("--variants", nargs="+", default=None, help="кодеки для compress, напр. pca128_i8 pq16")
    b = sub.add_parser("bench", help="латентность batch=1 и пропускная способность")
    b.add_argument("--backbone", default="dinov2_s")
    b.add_argument("--threads", type=int, default=4)
    b.add_argument("--device", default=None, help="cuda | cpu; по умолчанию cuda, если доступна")
    b.add_argument("--batch", type=int, default=16)
    a = p.parse_args(argv)
    if a.cmd == "bench":
        cmd_bench(a.backbone, a.threads, a.device, a.batch)
        return
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if a.cmd in ("extract", "run"):
        cmd_extract(cfg, a.backbone, a.splits, a.batch_size, a.workers, a.device)
    if a.cmd in ("eval", "run"):
        cmd_eval(cfg, a.backbone, recall=a.recall)
    if a.cmd in ("compress", "run"):
        cmd_compress(cfg, a.backbone, a.variants)


if __name__ == "__main__":
    main()
