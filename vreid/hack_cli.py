"""CLI для данных организаторов.

  python -m vreid.hack_cli val    --config configs/hackathon.yaml --backbone dinov2_b
      локальная open-set валидация из train.csv: mAP/Rank-k (кросс-камерно по псевдокамере),
      биты, режим отказа (F1/TNR/mINP/PR-AUC), выбор порога → results/hack_<backbone>[_flip]_val.json

  python -m vreid.hack_cli submit --config configs/hackathon.yaml --backbone dinov2_b
      test.csv → embeddings.npy, submission.csv, candidates.csv (порог берётся из val-json)

  python -m vreid.hack_cli inspect-cams --config configs/hackathon.yaml
      сохраняет примеры кадров по псевдокамерам, чтобы проверить порог кластеризации глазами
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from . import bits as bitsmod
from .extract import extract_split, load_npz
from .hackathon_data import build_local_validation, load_test
from .index import GalleryIndex
from .metrics import evaluate, format_metrics
from .models import get_backbone
from .refusal import confidence_scores, evaluate_refusal, format_refusal, rank_gallery


def _slug(backbone: str) -> str:
    """ft:weights/hack_dinov2_b_224/best.pt → ft_hack_dinov2_b_224 (для имён папок и файлов)."""
    if backbone.startswith("ft:"):
        return "ft_" + Path(backbone[3:]).parent.name
    return backbone.replace("/", "_").replace(":", "_")


def _run_dir(cfg, backbone):
    return Path(cfg.get("runs_dir", "runs")) / cfg["name"] / _slug(backbone)


def _results(cfg):
    p = Path(cfg.get("results_dir", "results"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ablate_postprocess(q, g, has, sims) -> dict:
    """Строки абляций: базовый косинус / DBA-уточнение векторов / k-reciprocal / оба."""
    from .rerank import dba, frame_block_mask, k_reciprocal
    m = has
    ev = lambda s: evaluate(s[m], q["vids"][m], q["cams"][m], g["vids"], g["cams"], cutoff=10)
    keys = np.concatenate([q["keys"], g["keys"]])
    block = frame_block_mask(keys, keys)
    allv = np.concatenate([q["emb"], g["emb"]])
    Q = len(q["emb"])
    rows = [("base", ev(sims))]
    best_k, best_map = 0, rows[0][1]["mAP"]
    for k in (1, 2, 3, 5):
        ref = dba(allv, k=k, block_mask=block)
        r = ev(ref[:Q] @ ref[Q:].T)
        rows.append((f"dba k={k}", r))
        if r["mAP"] > best_map:
            best_k, best_map = k, r["mAP"]
    kr_grid = ((6, 2), (10, 3), (15, 4), (20, 6))
    for k1, k2 in kr_grid:
        rows.append((f"k-recip {k1}/{k2}", ev(-k_reciprocal(q["emb"], g["emb"], k1, k2, 0.3, same_mask=block))))
    if best_k:
        ref = dba(allv, k=best_k, block_mask=block)
        for k1, k2 in kr_grid:
            rows.append((f"dba k={best_k} + k-recip {k1}/{k2}",
                         ev(-k_reciprocal(ref[:Q], ref[Q:], k1, k2, 0.3, same_mask=block))))
    best = max(rows, key=lambda t: t[1]["mAP"])
    print("постобработка (метрика жюри):")
    for name, r in rows:
        mark = "  ← лучшее" if name == best[0] else ""
        print(f"   {name:<28} mAP={r['mAP'] * 100:5.1f}%  rank1={r['rank1'] * 100:5.1f}%  rank5={r['rank5'] * 100:5.1f}%{mark}")
    print(f"   (разница < ~0.7 п.п. при {int(m.sum())} запросах — шум; выбирай самое простое из лучших)")
    return {"rows": [{"name": n, **r} for n, r in rows], "best_dba_k": best_k, "best": best[0]}


def _ablate_local(cfg, bb, d, q, g, has, sims, out, post, batch_size, workers) -> dict:
    """Локальное сопоставление патчей для топ-K: извлечение токенов, подбор beta, строки абляций."""
    from .local_match import LocalFeatures, local_scores, tune_beta, fuse
    from .refusal import rank_gallery
    crop = cfg.get("crop", {})
    size, topk = int(post.get("local_size", 336)), int(post.get("local_topk", 30))
    lf = LocalFeatures(bb, size=size, pca_dim=int(post.get("local_pca", 64)))
    tag = "" if d.get("protocol", "id") == "id" else f"_{d['protocol']}"
    qp, gp = out / f"val_query{tag}_local{size}.npz", out / f"val_gallery{tag}_local{size}.npz"
    if qp.exists() and gp.exists() and post.get("reuse_local", True):
        qt, gt = np.load(qp), np.load(gp)
        q_tok, g_tok, grid = qt["tokens"], gt["tokens"], qt["grid"]
    else:
        qd = lf.extract(d["query"], qp, crop.get("pad", 0.08), None, batch_size, workers)
        gd = lf.extract(d["gallery"], gp, crop.get("pad", 0.08), None, batch_size, workers)
        q_tok, g_tok, grid = qd["tokens"], gd["tokens"], qd["grid"]
    order, sm = rank_gallery(sims, q["cams"], g["cams"], cross_camera_only=True)
    order = np.where(np.isfinite(np.take_along_axis(sm, order, 1)), order, -1)
    ls = local_scores(q_tok, g_tok, order, grid, topk=topk, min_sim=float(post.get("local_min_sim", 0.6)))
    T = int(grid[0] * grid[1])
    beta, best = tune_beta(sims, order, ls, T, q["vids"], q["cams"], g["vids"], g["cams"], has)
    base = evaluate(sims[has], q["vids"][has], q["cams"][has], g["vids"], g["cams"], cutoff=10)
    print(f"локальное сопоставление (top-{topk}, {size}px, сетка {grid[0]}x{grid[1]}): "
          f"base mAP={base['mAP'] * 100:.1f}% → +local(beta={beta}) mAP={best['mAP'] * 100:.1f}% rank1={best['rank1'] * 100:.1f}%")
    return {"beta": beta, "topk": topk, "size": size, "min_sim": float(post.get("local_min_sim", 0.6)),
            "base": base, "fused": best}


def _val_json(cfg, backbone_name: str, tta_flip: bool = False, fast_decode: bool = False) -> Path:
    """Калибровка порога зависит от того, как считались эмбеддинги — держим отдельные файлы.

    TTA и быстрый декод дают ДРУГИЕ векторы, а значит и другую шкалу уверенности. Без разделения
    имён прогон с --fast-decode тихо затирал бы тот самый файл, из которого export_release.py
    берёт порог релиза — и релиз уехал бы с порогом от другого эксперимента."""
    tag = ("_flip" if tta_flip else "") + ("_fast" if fast_decode else "")
    return _results(cfg) / f"{cfg['name']}_{_slug(backbone_name)}{tag}_val.json"


def _val_files(d: dict, out: Path) -> tuple[Path, Path]:
    """Имена кэша val-эмбеддингов: у протокола track свои файлы (другое разбиение query/gallery)."""
    tag = "" if d.get("protocol", "id") == "id" else f"_{d['protocol']}"
    return out / f"val_query{tag}.npz", out / f"val_gallery{tag}.npz"


def cmd_val(cfg, backbone_name, batch_size, workers, device, reuse, post=None):
    post = post or {}
    d = build_local_validation(cfg["dataset"])
    out = _run_dir(cfg, backbone_name)
    out.mkdir(parents=True, exist_ok=True)
    d["query"].to_csv(out / "val_query.csv")
    d["gallery"].to_csv(out / "val_gallery.csv")
    d["fit"].to_csv(out / "fit.csv")
    np.save(out / "val_has_match.npy", d["has_match"])
    q_npz, g_npz = _val_files(d, out)
    if post.get("tta_flip"):
        q_npz, g_npz = q_npz.with_name(q_npz.stem + "_flip.npz"), g_npz.with_name(g_npz.stem + "_flip.npz")
    if post.get("fast_decode"):
        q_npz, g_npz = q_npz.with_name(q_npz.stem + "_fast.npz"), g_npz.with_name(g_npz.stem + "_fast.npz")
    if post.get("mask_plate"):
        tag = f"_mask-{post['mask_plate']}"
        q_npz, g_npz = q_npz.with_name(q_npz.stem + tag + ".npz"), g_npz.with_name(g_npz.stem + tag + ".npz")

    crop = cfg.get("crop", {})
    # --mask-plate: абляция «модель опирается на номер или нет». Зона номера ищется детектором
    # (vreid/plate.py), а не фиксированным прямоугольником, иначе при косом ракурсе маска мимо
    # пластины и закрывает бампер — и падение качества уже не про номер.
    #   det/auto — закрыть зону номера; up/rand — КОНТРОЛЬ той же площади в другом месте;
    #   band — старый грубый прямоугольник (для сравнения с прежним замером).
    mask_mode = post.get("mask_plate") or None
    mask = None
    if mask_mode:
        from .plate import load_cache
        cache_path = cfg.get("plate_cache", "results/plate_boxes.json")
        cache = load_cache(cache_path)
        n_found = sum(1 for v in cache.values() if v)
        if mask_mode in ("det", "auto", "up", "rand") and n_found == 0:
            raise SystemExit(
                f"в кеше {cache_path} нет ни одного бокса ({len(cache)} записей) — "
                f"абляция была бы пустой. Пересобери: python scripts/build_plate_cache.py "
                f"(и проверь, что стоит opencv: pip install opencv-python-headless)")
        mask = {"cache": cache, "mode": mask_mode}
        print(f"[val] маска «{mask_mode}», боксов в кеше {n_found}/{len(cache)}")
    bb = None
    if reuse and q_npz.exists() and g_npz.exists():
        q, g = load_npz(q_npz), load_npz(g_npz)
        if len(q["vids"]) != len(d["query"]) or len(g["vids"]) != len(d["gallery"]):
            raise SystemExit("--reuse: кэш эмбеддингов не совпадает с текущим разбиением (сменился протокол/seed?) — запусти без --reuse")
    else:
        bb = get_backbone(backbone_name, device=device)
        q = extract_split(d["query"], bb, batch_size, workers, q_npz,
                          pad=crop.get("pad", 0.08), mask=mask,
                          tta_flip=bool(post.get("tta_flip")),
                          fast_decode=bool(post.get("fast_decode")))
        g = extract_split(d["gallery"], bb, batch_size, workers, g_npz,
                          pad=crop.get("pad", 0.08), mask=mask,
                          tta_flip=bool(post.get("tta_flip")),
                          fast_decode=bool(post.get("fast_decode")))
    has = d["has_match"]

    index = GalleryIndex(g["emb"], g["vids"], g["cams"])
    sims = index.all_sims(q["emb"])
    # метрики ранжирования — только по запросам с парой
    m = has
    res = {"dataset": cfg["name"], "backbone": backbone_name, "dim": int(q["emb"].shape[1]),
           "protocol": d["protocol"], "refusal_mask_cam": d["refusal_mask_cam"],
           "tta_flip": bool(post.get("tta_flip")), "fast_decode": bool(post.get("fast_decode")),
           "mask_plate": mask_mode,
           "n_query": int(len(has)), "n_query_no_match": int((~has).sum()), "n_gallery": int(index.n)}
    # Метрика жюри — это удаление только пар «тот же vehicle_id И та же camera_id» (ответы 5 и 38).
    # Строгий cross-camera режим дополнительно выбрасывает ЧУЖИЕ машины с камеры запроса, то есть
    # самые трудные негативы, и завышает mAP на 1-3 пункта. Держим обе строки, но заголовок
    # «метрика жюри» стоит на правильной.
    res["jury"] = evaluate(sims[m], q["vids"][m], q["cams"][m], g["vids"], g["cams"], cutoff=10)
    res["standard"] = res["jury"]                      # совместимость со старыми файлами
    res["cross_camera_strict"] = evaluate(sims[m], q["vids"][m], q["cams"][m], g["vids"], g["cams"], cross_camera_only=True, cutoff=10)
    res["cross_camera"] = res["cross_camera_strict"]   # совместимость со старыми файлами
    print(format_metrics(res["jury"], f"[{d['protocol']}] МЕТРИКА ЖЮРИ (убрано только vid+cam)"))
    print(format_metrics(res["cross_camera_strict"], "строгий cross-cam (диагностика, завышает)"))

    cal = bitsmod.calibrate_tau(sims[m], q["vids"][m], q["cams"][m], g["vids"], g["cams"], recall=0.95)
    res["bits"] = {"calibration": cal,
                   "table": bitsmod.bits_calibration_table(sims[m], q["vids"][m], q["cams"][m], g["vids"], g["cams"], cal["tau"])}
    print(f"bits: tau={cal['tau']:.3f}, max {np.log2(index.n):.1f} бит при галерее {index.n}")
    for row in res["bits"]["table"]:
        print(f"   {row['bits_lo']:5.1f}–{row['bits_hi']:5.1f} бит  n={row['n']:5d}  rank-1 = {row['rank1_acc'] * 100:5.1f}%")

    if post.get("rerank"):
        res["postprocess"] = _ablate_postprocess(q, g, has, sims)
    if post.get("local"):
        if bb is None:
            bb = get_backbone(backbone_name, device=device)
        res["local"] = _ablate_local(cfg, bb, d, q, g, has, sims, out, post, batch_size, workers)

    ref = evaluate_refusal(sims, q["vids"], q["cams"], g["vids"], g["cams"], has, tau=cal["tau"],
                           min_tnr=(cfg.get("refusal") or {}).get("min_tnr"),
                           cross_camera_only=d["refusal_mask_cam"])
    print(format_refusal(ref) + ("" if d["refusal_mask_cam"] else "   (камера запроса не маскируется — как на тесте)"))
    # кривые в JSON держим, но не печатаем
    res["refusal"] = {k: v for k, v in ref.items() if k != "by_confidence"}
    res["refusal"]["by_confidence"] = {n: {"chosen": e["chosen"], "pr_auc": e["pr_auc"]} for n, e in ref["by_confidence"].items()}
    res["refusal"]["curves"] = {n: e["curve"] for n, e in ref["by_confidence"].items()}

    path = _val_json(cfg, backbone_name, bool(post.get("tta_flip")), bool(post.get("fast_decode")))
    if mask_mode:
        path = path.with_name(f"{path.stem}_mask-{mask_mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[val] → {path}")
    return res


def cmd_submit(cfg, backbone_name, batch_size, workers, device, out_dir, threshold, confidence, post=None):
    from .rerank import dba, frame_block_mask, k_reciprocal
    from .submit import save_embeddings, write_candidates, write_submission
    post = post or {}
    tests = load_test(cfg["dataset"])
    run = _run_dir(cfg, backbone_name)
    crop = cfg.get("crop", {})
    bb = get_backbone(backbone_name, device=device)
    tta = bool(post.get("tta_flip"))
    ex = lambda split: extract_split(split, bb, batch_size, workers,
                                     run / f"test_{split.name}{'_flip' if tta else ''}.npz",
                                     pad=crop.get("pad", 0.08), mask=None, tta_flip=tta)
    if "query" in tests:                       # явные query / gallery (формат организаторов)
        tq, tg = ex(tests["query"]), ex(tests["gallery"])
        shared = False
    else:                                      # один test.csv: каждая запись — запрос ко всем остальным
        tq = tg = ex(tests["test"])
        shared = True
    q_keys, g_keys = list(tq["keys"]), list(tg["keys"])

    val_json = _val_json(cfg, backbone_name, tta)
    tau, v = None, None
    if val_json.exists():
        with open(val_json, encoding="utf-8") as f:
            v = json.load(f)
        tau = v["bits"]["calibration"]["tau"]
    if threshold is None or confidence is None:
        if v is None:
            raise SystemExit(f"нет {val_json}: запусти val{' --tta-flip' if tta else ''} "
                             f"(калибровка порога должна быть с тем же TTA), или задай --threshold и --confidence")
        confidence = confidence or v["refusal"]["best_confidence"]
        threshold = threshold if threshold is not None else v["refusal"]["by_confidence"][confidence]["chosen"]["threshold"]
        print(f"[submit] порог из val (протокол {v.get('protocol', 'id')}"
              f"{', TTA flip' if v.get('tta_flip') else ''}): уверенность={confidence}, threshold={threshold:.4f}")
        if v.get("refusal_mask_cam", True):
            print("[submit] ВНИМАНИЕ: порог калиброван с маской своей камеры, а на тесте камер нет — "
                  "ожидай заниженную долю отказов. Лучше protocol: track в конфиге и заново val.")

    q_emb, g_emb = tq["emb"], tg["emb"]
    # пары, которые нельзя считать соседями: та же запись и тот же кадр
    if shared:
        block = frame_block_mask(q_keys, q_keys); np.fill_diagonal(block, True)
    else:
        allk = q_keys + g_keys
        block = frame_block_mask(allk, allk); np.fill_diagonal(block, True)
    qg_block = block if shared else block[: len(q_keys), len(q_keys):]

    # УВЕРЕННОСТЬ ДЛЯ ОТКАЗА — строго по сырым векторам: порог калиброван на val без DBA,
    # а DBA поднимает сходства и иначе почти ничего не отказывается.
    raw_sims = q_emb @ g_emb.T
    raw_sims[qg_block] = -np.inf
    raw_order = np.argsort(-raw_sims, axis=1)
    scores = confidence_scores(raw_sims, raw_order, tau)
    if "bits" in scores:
        scores["top1+bits"] = scores["top1"] + 0.02 * scores["bits"]
    conf = scores[confidence]
    top1 = scores["top1"]
    print(f"[submit] top-1 сходство на тесте: медиана {np.median(top1):.3f}, 10%={np.quantile(top1, .1):.3f}, "
          f"90%={np.quantile(top1, .9):.3f}; порог {threshold:.3f} → отказов ожидается {(conf < threshold).mean() * 100:.1f}%")
    if v is not None and "refusal" in v:
        ar = v["refusal"]["by_confidence"].get(confidence, {}).get("chosen", {}).get("accept_rate")
        if ar is not None:
            print(f"[submit] для сравнения на val при этом пороге принималось {ar * 100:.1f}% запросов")

    if post.get("dba"):
        allv = dba(np.concatenate([q_emb, g_emb]) if not shared else q_emb, k=int(post["dba"]), block_mask=block)
        if shared:
            q_emb = g_emb = allv
        else:
            q_emb, g_emb = allv[: len(q_emb)], allv[len(q_emb):]
        print(f"[submit] DBA k={post['dba']}: векторы уточнены по взаимным соседям")
    sims = q_emb @ g_emb.T
    sims[qg_block] = -np.inf
    order = np.argsort(-sims, axis=1)

    rank_sims = sims
    if post.get("local") and v is not None and "local" in v:
        from .local_match import LocalFeatures, local_scores, fuse
        L = v["local"]
        lf = LocalFeatures(bb, size=int(L["size"]), pca_dim=64)
        lq = lf.extract(tests.get("query", tests.get("test")), run / f"test_query_local{L['size']}.npz",
                        crop.get("pad", 0.08), None, batch_size, workers)
        lg = lq if shared else lf.extract(tests["gallery"], run / f"test_gallery_local{L['size']}.npz",
                                          crop.get("pad", 0.08), None, batch_size, workers)
        order_top = np.where(np.isfinite(np.take_along_axis(sims, order, 1)), order, -1)
        ls = local_scores(lq["tokens"], lg["tokens"], order_top, lq["grid"], topk=int(L["topk"]), min_sim=float(L["min_sim"]))
        rank_sims = fuse(sims, order_top, ls, float(L["beta"]), int(lq["grid"][0] * lq["grid"][1]))
        print(f"[submit] локальное сопоставление: top-{L['topk']}, beta={L['beta']}")
    if post.get("kr"):
        k1, k2 = int(post.get("k1", 20)), int(post.get("k2", 6))
        if shared:
            d = k_reciprocal(q_emb, q_emb, k1, k2, 0.3, same_mask=block, shared=True)
            d[block] = np.inf
        else:
            d = k_reciprocal(q_emb, g_emb, k1, k2, 0.3, same_mask=block)
            d[~np.isfinite(sims)] = np.inf
        rank_sims = -d
        print(f"[submit] k-reciprocal {k1}/{k2} применено к submission.csv")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_out = q_emb if shared else np.concatenate([q_emb, g_emb])   # порядок: query, затем gallery
    save_embeddings(emb_out, out_dir / "embeddings.npy")
    write_submission(q_keys, g_keys, rank_sims, out_dir / "submission.csv", topk=10, exclude_self=shared)
    write_candidates(q_keys, g_keys, rank_sims, conf, threshold, out_dir / "candidates.csv",
                     topk=int(post.get("cand_topk", 1)), exclude_self=shared, conf_sims=raw_sims,
                     per_candidate_min_sim=post.get("cand_min"))


def cmd_ens(cfg, backbones: list[str], weights: list[float] | None = None):
    """Ансамбль по уже посчитанным val-эмбеддингам нескольких моделей: конкатенация нормированных
    векторов (с весами) → те же метрики, постобработка и режим отказа. Ничего не пересчитывает."""
    d = build_local_validation(cfg["dataset"])
    has = d["has_match"]
    qs, gs = [], []
    for i, b in enumerate(backbones):
        run = _run_dir(cfg, b)
        q_npz, g_npz = _val_files(d, run)
        if not q_npz.exists():
            raise SystemExit(f"нет {q_npz}: сначала прогони val для {b} с текущим протоколом")
        q, g = load_npz(q_npz), load_npz(g_npz)
        w = float(weights[i]) if weights else 1.0
        qs.append(q["emb"] * w); gs.append(g["emb"] * w)
        print(f"[ens] {b}: D={q['emb'].shape[1]}, вес {w}")
        if i == 0:
            q0, g0 = q, g
    q_emb = np.concatenate(qs, axis=1); g_emb = np.concatenate(gs, axis=1)
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True); g_emb /= np.linalg.norm(g_emb, axis=1, keepdims=True)
    q = {**q0, "emb": q_emb.astype(np.float32)}; g = {**g0, "emb": g_emb.astype(np.float32)}
    sims = q_emb @ g_emb.T
    m = has
    name = "ens_" + "+".join(_slug(b) for b in backbones)
    res = {"dataset": cfg["name"], "backbone": name, "members": backbones, "weights": weights, "dim": int(q_emb.shape[1]),
           "protocol": d["protocol"], "refusal_mask_cam": d["refusal_mask_cam"]}
    res["cross_camera"] = evaluate(sims[m], q["vids"][m], q["cams"][m], g["vids"], g["cams"], cross_camera_only=True, cutoff=10)
    print(format_metrics(res["cross_camera"], f"ансамбль {len(backbones)} моделей (cross-cam)"))
    res["postprocess"] = _ablate_postprocess(q, g, has, sims)
    cal = bitsmod.calibrate_tau(sims[m], q["vids"][m], q["cams"][m], g["vids"], g["cams"], recall=0.95)
    res["bits"] = {"calibration": cal}
    ref = evaluate_refusal(sims, q["vids"], q["cams"], g["vids"], g["cams"], has, tau=cal["tau"],
                           min_tnr=(cfg.get("refusal") or {}).get("min_tnr"),
                           cross_camera_only=d["refusal_mask_cam"])
    print(format_refusal(ref))
    res["refusal"] = {k: v for k, v in ref.items() if k != "by_confidence"}
    res["refusal"]["by_confidence"] = {n: {"chosen": e["chosen"], "pr_auc": e["pr_auc"]} for n, e in ref["by_confidence"].items()}
    out = _run_dir(cfg, name); out.mkdir(parents=True, exist_ok=True)
    oq, og = _val_files(d, out)
    np.savez(oq, **q); np.savez(og, **g)
    path = _results(cfg) / f"{cfg['name']}_{name}_val.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[ens] → {path}")
    return res


def cmd_inspect_cams(cfg, n_per_cam=4):
    from PIL import Image
    from .hackathon_data import assign_pseudo_cameras, read_annotations
    ds = cfg["dataset"]
    root = Path(ds["root"])
    recs = read_annotations(root / ds.get("train_csv", "train.csv"), root / ds.get("images_dir", "images"),
                            ds.get("cols"), ds.get("bbox_format", "xywh"))
    recs = assign_pseudo_cameras(recs, ds.get("pseudo_camera_threshold", 0.35))
    by_cam = {}
    for r in recs:
        by_cam.setdefault(r.cam, set()).add(r.path)
    out = _results(cfg) / "pseudo_cameras"
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for cam, paths in sorted(by_cam.items()):
        paths = sorted(paths)
        lines.append(f"cam {cam}: {len(paths)} кадров")
        thumbs = []
        for p in paths[:n_per_cam]:
            with Image.open(p) as im:
                im.thumbnail((320, 240)); thumbs.append(im.copy())
        sheet = Image.new("RGB", (320 * len(thumbs), 240), (0, 0, 0))
        for i, th in enumerate(thumbs):
            sheet.paste(th, (320 * i, 0))
        sheet.save(out / f"cam_{cam:04d}.jpg", quality=80)
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:30]))
    print(f"[inspect-cams] листы по камерам → {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="vreid.hack_cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("ens", help="ансамбль по готовым val-эмбеддингам: --backbones A B [C]")
    e.add_argument("--config", required=True)
    e.add_argument("--backbones", nargs="+", required=True)
    e.add_argument("--weights", nargs="+", type=float, default=None)
    for name in ("val", "submit", "inspect-cams"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--backbone", default="dinov2_b")
        s.add_argument("--batch-size", type=int, default=64)
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--device", default=None)
        s.add_argument("--reuse", action="store_true", help="val: не пересчитывать эмбеддинги")
        s.add_argument("--out", default="submission", help="submit: папка артефактов")
        s.add_argument("--threshold", type=float, default=None)
        s.add_argument("--confidence", default=None, help="top1 | margin | bits | top1+bits")
        s.add_argument("--rerank", action="store_true", help="val: абляции DBA / k-reciprocal")
        s.add_argument("--local", action="store_true", help="локальное сопоставление патчей для top-K")
        s.add_argument("--local-size", type=int, default=336)
        s.add_argument("--local-topk", type=int, default=30)
        s.add_argument("--local-min-sim", type=float, default=0.6)
        s.add_argument("--dba", type=int, default=0, help="submit: уточнить векторы по k взаимным соседям")
        s.add_argument("--kr", action="store_true", help="submit: k-reciprocal для submission.csv")
        s.add_argument("--k1", type=int, default=20, help="submit: k1 для k-reciprocal (см. таблицу val --rerank)")
        s.add_argument("--k2", type=int, default=6, help="submit: k2 для k-reciprocal")
        s.add_argument("--cand-min", type=float, default=None,
                       help="не писать в candidates.csv кандидатов слабее этого косинуса")
        s.add_argument("--cand-topk", type=int, default=1,
                       help="строк на принятый запрос в candidates.csv (в метрику идёт только топ-1)")
        s.add_argument("--mask-plate", nargs="?", const="det", default=None,
                       choices=["det", "auto", "band", "up", "rand"],
                       help="закрасить зону номера: det — только найденную детектором, auto — плюс "
                            "запасная полоса там, где не нашли; up/rand — КОНТРОЛЬ той же площади в "
                            "другом месте (отделяет «потеряли номер» от «потеряли пиксели»); "
                            "band — старый грубый прямоугольник")
        s.add_argument("--fast-decode", action="store_true",
                       help="ускоренное декодирование JPEG (как в боевом инференсе)")
        s.add_argument("--tta-flip", action="store_true",
                       help="усреднить вектор кропа с вектором его зеркала (инференс ×2 по времени)")
    a = p.parse_args(argv)
    if a.cmd == "ens":
        with open(a.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cmd_ens(cfg, a.backbones, a.weights)
        return
    post = {"rerank": getattr(a, "rerank", False), "local": getattr(a, "local", False),
            "local_size": getattr(a, "local_size", 336), "local_topk": getattr(a, "local_topk", 30),
            "local_min_sim": getattr(a, "local_min_sim", 0.6), "dba": getattr(a, "dba", 0), "kr": getattr(a, "kr", False),
                "k1": getattr(a, "k1", 20), "k2": getattr(a, "k2", 6),
                "tta_flip": getattr(a, "tta_flip", False), "fast_decode": getattr(a, "fast_decode", False),
                "mask_plate": getattr(a, "mask_plate", None),
                "cand_min": getattr(a, "cand_min", None),
                "cand_topk": getattr(a, "cand_topk", 1)}
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if a.cmd == "val":
        cmd_val(cfg, a.backbone, a.batch_size, a.workers, a.device, a.reuse, post)
    elif a.cmd == "submit":
        cmd_submit(cfg, a.backbone, a.batch_size, a.workers, a.device, a.out, a.threshold, a.confidence, post)
    elif a.cmd == "inspect-cams":
        cmd_inspect_cams(cfg)


if __name__ == "__main__":
    main()
