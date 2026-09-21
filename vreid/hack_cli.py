"""CLI для данных организаторов.

  python -m vreid.hack_cli val    --config configs/hackathon.yaml --backbone dinov2_b
      локальная open-set валидация из train.csv: mAP/Rank-k (кросс-камерно по псевдокамере),
      биты, режим отказа (F1/TNR/mINP/PR-AUC), выбор порога
      → results/<cfg.name>_<backbone>[_flip][_fast][_mask-<режим>]_val.json
      (имя зависит от режима: у разных режимов своя шкала уверенности, см. _val_json)

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

# Размер миниатюры в диагностических листах по псевдокамерам (inspect-cams).
_THUMB = (320, 240)

# Вес исходного косинуса против жаккардова расстояния в k-reciprocal (аргумент lam).
# Это дефолт rerank.k_reciprocal, оставлен намеренно: абляция мерит k1/k2, а не lam.
KR_LAMBDA = 0.3


def _slug(backbone: str) -> str:
    """ft:weights/hack_dinov2_b_224/best.pt → ft_hack_dinov2_b_224 (для имён папок и файлов)."""
    if backbone.startswith("ft:"):
        return "ft_" + Path(backbone[3:]).parent.name
    return backbone.replace("/", "_").replace(":", "_")


def _run_dir(cfg: dict, backbone: str) -> Path:
    """Каталог прогона runs/<cfg.name>/<slug бэкбона> для кэша эмбеддингов и CSV разбиения.

    Каталог НЕ создаёт — это делает вызывающий (cmd_val, cmd_ens)."""
    return Path(cfg.get("runs_dir", "runs")) / cfg["name"] / _slug(backbone)


def _results(cfg: dict) -> Path:
    """Каталог для отчётных JSON (cfg.results_dir, по умолчанию results/).

    СОЗДАЁТ каталог — вызывающему mkdir не нужен."""
    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _load_config(path: str) -> dict:
    """Читает YAML-конфиг. Кодировка задана явно: в конфигах есть русские комментарии,
    а дефолтная кодировка Windows их ломает."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _valid_order(sims, order) -> np.ndarray:
    """Помечает -1 позиции топа, закрытые маской (в sims там не-конечное значение).

    -1 — метка «кандидата нет»: позиция закрыта маской своей камеры/кадра. local_scores такие
    позиции пропускает (j < 0), поэтому подставлять сюда реальный индекс нельзя — иначе в топ
    попадёт заведомо запрещённая пара."""
    return np.where(np.isfinite(np.take_along_axis(sims, order, 1)), order, -1)


def _dump_result_json(res: dict, path: Path, label: str) -> None:
    """Пишет отчётный JSON и печатает путь. json.dump сохраняет порядок вставки ключей —
    на него опираются scripts/export_release.py и побайтовое сравнение отчётов."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[{label}] → {path}")


def _ablate_postprocess(q, g, has_match, sims) -> dict:
    """Строки абляций: базовый косинус / DBA-уточнение векторов / k-reciprocal / оба.

    q, g — словари из load_npz (emb/vids/cams/keys). has_match — маска запросов, у которых
    в галерее есть пара; метрики считаются только по ним. sims — сходства ДО постобработки.

    Возвращает {rows, best, best_dba_k}, где best_dba_k = 0 означает «DBA не дала прироста»,
    а не «k=0» (нулевого k в сетке нет)."""
    from .rerank import dba, frame_block_mask, k_reciprocal

    def metrics_for(sim_matrix):
        """Метрики жюри по матрице сходств: считаем только по запросам с парой в галерее."""
        return evaluate(sim_matrix[has_match], q["vids"][has_match], q["cams"][has_match],
                        g["vids"], g["cams"], cutoff=10)

    keys = np.concatenate([q["keys"], g["keys"]])
    block = frame_block_mask(keys, keys)
    all_emb = np.concatenate([q["emb"], g["emb"]])
    n_query = len(q["emb"])
    rows = [("base", metrics_for(sims))]
    # best_k=0 — сентинел «DBA хуже базы» для проверки `if best_k:` ниже (k=0 в сетке не
    # перебирается). best_map стартует с mAP базы, сравнение строгое `>`, поэтому при ничьей
    # побеждают база и меньшее k. max ниже возвращает ПЕРВЫЙ максимум — это и фиксирует выбор
    # при равных mAP.
    best_k, best_map = 0, rows[0][1]["mAP"]
    for k in (1, 2, 3, 5):
        refined = dba(all_emb, k=k, block_mask=block)
        metrics = metrics_for(refined[:n_query] @ refined[n_query:].T)
        rows.append((f"dba k={k}", metrics))
        if metrics["mAP"] > best_map:
            best_k, best_map = k, metrics["mAP"]
    # KR_LAMBDA — это lam из rerank.k_reciprocal (вес исходного косинуса против жаккардова
    # расстояния); дефолт оставлен намеренно, чтобы абляция мерила k1/k2, а не lam.
    # Минус перед вызовом обязателен: k_reciprocal отдаёт РАССТОЯНИЯ, а метрика ждёт сходства.
    kr_grid = ((6, 2), (10, 3), (15, 4), (20, 6))
    for k1, k2 in kr_grid:
        rows.append((f"k-recip {k1}/{k2}",
                     metrics_for(-k_reciprocal(q["emb"], g["emb"], k1, k2,
                                               KR_LAMBDA, same_mask=block))))
    if best_k:
        refined = dba(all_emb, k=best_k, block_mask=block)
        for k1, k2 in kr_grid:
            rows.append((f"dba k={best_k} + k-recip {k1}/{k2}",
                         metrics_for(-k_reciprocal(refined[:n_query], refined[n_query:],
                                                   k1, k2, KR_LAMBDA, same_mask=block))))
    best = max(rows, key=lambda t: t[1]["mAP"])
    print("постобработка (метрика жюри):")
    for name, metrics in rows:
        mark = "  ← лучшее" if name == best[0] else ""
        print(f"   {name:<28} mAP={metrics['mAP'] * 100:5.1f}%  "
              f"rank1={metrics['rank1'] * 100:5.1f}%  rank5={metrics['rank5'] * 100:5.1f}%{mark}")
    print(f"   (разница < ~0.7 п.п. при {int(has_match.sum())} запросах — шум; "
          f"выбирай самое простое из лучших)")
    return {"rows": [{"name": n, **metrics} for n, metrics in rows],
            "best_dba_k": best_k, "best": best[0]}


def _ablate_local(cfg, backbone, val_split, q, g, has_match, sims, out, post,
                  batch_size, workers) -> dict:
    """Локальное сопоставление патчей для топ-K: извлечение токенов, подбор beta, строки абляций.

    Возвращает dict, который попадает в val-JSON как res["local"] и читается cmd_submit:
    beta (подобранный вес локальных инлайеров), size/topk/min_sim (параметры извлечения
    токенов — на тесте должны совпадать с val, иначе beta не переносится), base/fused —
    метрики до и после слияния. Имена полей менять нельзя."""
    from .local_match import LocalFeatures, local_scores, tune_beta
    crop = cfg.get("crop", {})
    size, topk = int(post.get("local_size", 336)), int(post.get("local_topk", 30))
    local_feats = LocalFeatures(backbone, size=size, pca_dim=int(post.get("local_pca", 64)))
    tag = _protocol_tag(val_split)
    query_npz = out / f"val_query{tag}_local{size}.npz"
    gallery_npz = out / f"val_gallery{tag}_local{size}.npz"
    if query_npz.exists() and gallery_npz.exists() and post.get("reuse_local", True):
        q_cached, g_cached = np.load(query_npz), np.load(gallery_npz)
        q_tok, g_tok, grid = q_cached["tokens"], g_cached["tokens"], q_cached["grid"]
    else:
        # Порядок важен: LocalFeatures подгоняет PCA на ПЕРВОМ сплите (local_match.py:95-99,
        # `if self.P is None: self.fit_pca(...)`) — базис берётся по query, галерея проецируется
        # в него же. Переставить вызовы, завести отдельный LocalFeatures на сплит или
        # распараллелить = другие токены → другая beta → другой submission. PCA может
        # подогнаться и посреди цикла, поэтому базис зависит ещё и от batch_size.
        q_fresh = local_feats.extract(val_split["query"], query_npz, pad=crop.get("pad", 0.08),
                                      mask_frac=None, batch_size=batch_size, workers=workers)
        g_fresh = local_feats.extract(val_split["gallery"], gallery_npz, pad=crop.get("pad", 0.08),
                                      mask_frac=None, batch_size=batch_size, workers=workers)
        q_tok, g_tok, grid = q_fresh["tokens"], g_fresh["tokens"], q_fresh["grid"]
    order, masked_sims = rank_gallery(sims, q["cams"], g["cams"], cross_camera_only=True)
    # -1 в топе = «кандидата нет», позиция закрыта маской камеры (см. _valid_order)
    order = _valid_order(masked_sims, order)
    local_sims = local_scores(q_tok, g_tok, order, grid, topk=topk,
                              min_sim=float(post.get("local_min_sim", 0.6)))
    n_tokens = int(grid[0] * grid[1])
    beta, best = tune_beta(sims, order, local_sims, n_tokens, q["vids"], q["cams"],
                           g["vids"], g["cams"], has_match)
    # base — метрика жюри (cross_camera_only=False), а tune_beta подбирает beta по строгому
    # cross-cam (local_match.py). Числа в печати ниже НЕ сопоставимы напрямую; приводить к
    # одному протоколу — отдельная задача с перезамером beta и прогоном
    # scripts/check_refactor_equivalence.py, а не косметика.
    base = evaluate(sims[has_match], q["vids"][has_match], q["cams"][has_match],
                    g["vids"], g["cams"], cutoff=10)
    print(f"локальное сопоставление (top-{topk}, {size}px, сетка {grid[0]}x{grid[1]}): "
          f"base mAP={base['mAP'] * 100:.1f}% → +local(beta={beta}) "
          f"mAP={best['mAP'] * 100:.1f}% rank1={best['rank1'] * 100:.1f}%")
    return {"beta": beta, "topk": topk, "size": size, "min_sim": float(post.get("local_min_sim", 0.6)),
            "base": base, "fused": best}


def _val_json(cfg, backbone_name: str, tta_flip: bool = False, fast_decode: bool = False) -> Path:
    """Калибровка порога зависит от того, как считались эмбеддинги — держим отдельные файлы.

    TTA и быстрый декод дают ДРУГИЕ векторы, а значит и другую шкалу уверенности. Без разделения
    имён прогон с --fast-decode тихо затирал бы тот самый файл, из которого export_release.py
    берёт порог релиза — и релиз уехал бы с порогом от другого эксперимента."""
    tag = ("_flip" if tta_flip else "") + ("_fast" if fast_decode else "")
    return _results(cfg) / f"{cfg['name']}_{_slug(backbone_name)}{tag}_val.json"


def _protocol_tag(split: dict) -> str:
    """Тег протокола в именах кэш-файлов: у протокола id тега нет (исторические имена без
    суффикса), у остальных — «_<protocol>», чтобы разные разбиения не затирали друг друга."""
    return "" if split.get("protocol", "id") == "id" else f"_{split['protocol']}"


def _val_files(split: dict, out: Path) -> tuple[Path, Path]:
    """Имена кэша val-эмбеддингов: у протокола track свои файлы (другое разбиение query/gallery)."""
    tag = _protocol_tag(split)
    return out / f"val_query{tag}.npz", out / f"val_gallery{tag}.npz"


def _tagged(path: Path, tag: str) -> Path:
    """Дописывает тег режима в имя npz-кэша: val_query.npz + «_flip» → val_query_flip.npz.

    Теги накладываются цепочкой, и порядок наложения определяет итоговое имя файла."""
    return path.with_name(path.stem + tag + ".npz")


def _val_cache_paths(split: dict, out: Path, post: dict) -> tuple[Path, Path]:
    """Пути npz-кэша val-эмбеддингов с учётом режима.

    Теги накладываются цепочкой flip → fast → mask — именно этот порядок задаёт итоговое имя,
    и по нему прогон одного режима не затирает кэш другого."""
    q_npz, g_npz = _val_files(split, out)
    if post.get("tta_flip"):
        q_npz, g_npz = _tagged(q_npz, "_flip"), _tagged(g_npz, "_flip")
    if post.get("fast_decode"):
        q_npz, g_npz = _tagged(q_npz, "_fast"), _tagged(g_npz, "_fast")
    if post.get("mask_plate"):
        tag = f"_mask-{post['mask_plate']}"
        q_npz, g_npz = _tagged(q_npz, tag), _tagged(g_npz, tag)
    return q_npz, g_npz


def _load_plate_mask(cfg, post) -> tuple[str | None, dict | None]:
    """Готовит описание маски номера для extract_split: (режим, {cache, mode}).

    Без --mask-plate возвращает (None, None). Пустой кэш боксов — это SystemExit здесь же,
    до создания бэкбона: иначе абляция молча померила бы «ничего не закрыли»."""
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
    return mask_mode, mask


def cmd_val(cfg, backbone_name, batch_size, workers, device, reuse, post=None):
    """Локальная open-set валидация: ранжирование, биты, режим отказа и выбор порога.

    Из cfg берутся: dataset (описание данных), name (имя прогона), runs_dir, results_dir,
    crop.pad, refusal.min_tnr, plate_cache. Из post понимаются ключи tta_flip, fast_decode,
    mask_plate, rerank, local и local_size/local_topk/local_min_sim/local_pca/reuse_local.

    Пишет в runs/<name>/<slug бэкбона>/: val_query.csv, val_gallery.csv, fit.csv,
    val_has_match.npy и npz-кэши эмбеддингов (плюс npz токенов при --local). Отчёт кладёт
    в results/ и возвращает его же словарём res.

    Этот JSON — источник порога для submit и для scripts/export_release.py, поэтому имя файла
    зависит от режима (см. _val_json): у TTA, быстрого декода и масок своя шкала уверенности.
    """
    post = post or {}
    val_split = build_local_validation(cfg["dataset"])
    out = _run_dir(cfg, backbone_name)
    out.mkdir(parents=True, exist_ok=True)
    # Имена без тега протокола намеренно: их читают scripts/error_analysis.py (154-155) и
    # scripts/make_manifest.py (57-58). CSV и val_has_match.npy пишутся ДО проверки --reuse,
    # чтобы после падения в папке лежало текущее разбиение, а не прошлое.
    val_split["query"].to_csv(out / "val_query.csv")
    val_split["gallery"].to_csv(out / "val_gallery.csv")
    val_split["fit"].to_csv(out / "fit.csv")
    np.save(out / "val_has_match.npy", val_split["has_match"])
    q_npz, g_npz = _val_cache_paths(val_split, out, post)

    crop = cfg.get("crop", {})
    # --mask-plate: абляция «модель опирается на номер или нет». Зона номера ищется детектором
    # (vreid/plate.py), а не фиксированным прямоугольником, иначе при косом ракурсе маска мимо
    # пластины и закрывает бампер — и падение качества уже не про номер.
    #   det/auto — закрыть зону номера; up/rand — КОНТРОЛЬ той же площади в другом месте;
    #   band — старый грубый прямоугольник (для сравнения с прежним замером).
    mask_mode, mask = _load_plate_mask(cfg, post)
    backbone = None
    if reuse and q_npz.exists() and g_npz.exists():
        q, g = load_npz(q_npz), load_npz(g_npz)
        if len(q["vids"]) != len(val_split["query"]) or len(g["vids"]) != len(val_split["gallery"]):
            raise SystemExit("--reuse: кэш эмбеддингов не совпадает с текущим разбиением "
                             "(сменился протокол/seed?) — запусти без --reuse")
    else:
        backbone = get_backbone(backbone_name, device=device)
        q = extract_split(val_split["query"], backbone, batch_size, workers, q_npz,
                          pad=crop.get("pad", 0.08), mask=mask,
                          tta_flip=bool(post.get("tta_flip")),
                          fast_decode=bool(post.get("fast_decode")))
        g = extract_split(val_split["gallery"], backbone, batch_size, workers, g_npz,
                          pad=crop.get("pad", 0.08), mask=mask,
                          tta_flip=bool(post.get("tta_flip")),
                          fast_decode=bool(post.get("fast_decode")))
    has_match = val_split["has_match"]

    index = GalleryIndex(g["emb"], g["vids"], g["cams"])
    sims = index.all_sims(q["emb"])
    # метрики ранжирования — только по запросам с парой
    res = {"dataset": cfg["name"], "backbone": backbone_name, "dim": int(q["emb"].shape[1]),
           "protocol": val_split["protocol"], "refusal_mask_cam": val_split["refusal_mask_cam"],
           "tta_flip": bool(post.get("tta_flip")), "fast_decode": bool(post.get("fast_decode")),
           "mask_plate": mask_mode,
           "n_query": int(len(has_match)), "n_query_no_match": int((~has_match).sum()),
           "n_gallery": int(index.n)}
    # Метрика жюри — это удаление только пар «тот же vehicle_id И та же camera_id» (ответы 5 и 38).
    # Строгий cross-camera режим дополнительно выбрасывает ЧУЖИЕ машины с камеры запроса, то есть
    # самые трудные негативы, и завышает mAP на 1-3 пункта. Держим обе строки, но заголовок
    # «метрика жюри» стоит на правильной.
    # cutoff=10 — это mAP@10 организаторов; столько же кандидатов пишется в submission.csv
    # (write_submission вызывается с topk=10).
    res["jury"] = evaluate(sims[has_match], q["vids"][has_match], q["cams"][has_match],
                           g["vids"], g["cams"], cutoff=10)
    # Алиасы для старых читателей val-JSON: scripts/export_release.py читает v["cross_camera"]
    # без фолбэка (строки 108, 126), scripts/ablation_report.py и scripts/scorecard.py
    # используют "standard" как запасной ключ.
    # ВНИМАНИЕ: в файлах прежних поколений (см. 1809/handoff/vreid/hack_cli.py:186) ключ
    # cross_camera считался БЕЗ cross_camera_only и хранил метрику жюри — сравнивать цифры
    # между поколениями JSON по этому ключу нельзя, только по "jury".
    res["standard"] = res["jury"]                      # алиас, см. комментарий выше
    res["cross_camera_strict"] = evaluate(sims[has_match], q["vids"][has_match],
                                          q["cams"][has_match], g["vids"], g["cams"],
                                          cross_camera_only=True, cutoff=10)
    res["cross_camera"] = res["cross_camera_strict"]   # алиас, см. комментарий выше
    print(format_metrics(res["jury"],
                         f"[{val_split['protocol']}] МЕТРИКА ЖЮРИ (убрано только vid+cam)"))
    print(format_metrics(res["cross_camera_strict"], "строгий cross-cam (диагностика, завышает)"))

    # recall=0.95 — рабочая точка радиуса: истинное кросс-камерное совпадение попадает в радиус
    # у 95% запросов. Ниже — радиус сужается и биты растут, но чаще теряем правильный ответ;
    # выше — радиус размывается и биты перестают различать запросы.
    cal = bitsmod.calibrate_tau(sims[has_match], q["vids"][has_match], q["cams"][has_match],
                                g["vids"], g["cams"], recall=0.95)
    res["bits"] = {"calibration": cal,
                   "table": bitsmod.bits_calibration_table(sims[has_match], q["vids"][has_match],
                                                           q["cams"][has_match], g["vids"],
                                                           g["cams"], cal["tau"])}
    print(f"bits: tau={cal['tau']:.3f}, max {np.log2(index.n):.1f} бит при галерее {index.n}")
    for row in res["bits"]["table"]:
        print(f"   {row['bits_lo']:5.1f}–{row['bits_hi']:5.1f} бит  n={row['n']:5d}  "
              f"rank-1 = {row['rank1_acc'] * 100:5.1f}%")

    if post.get("rerank"):
        res["postprocess"] = _ablate_postprocess(q, g, has_match, sims)
    if post.get("local"):
        if backbone is None:
            backbone = get_backbone(backbone_name, device=device)
        res["local"] = _ablate_local(cfg, backbone, val_split, q, g, has_match, sims, out, post,
                                     batch_size, workers)

    ref = evaluate_refusal(sims, q["vids"], q["cams"], g["vids"], g["cams"], has_match,
                           tau=cal["tau"],
                           min_tnr=(cfg.get("refusal") or {}).get("min_tnr"),
                           cross_camera_only=val_split["refusal_mask_cam"])
    print(format_refusal(ref) + ("" if val_split["refusal_mask_cam"]
                                 else "   (камера запроса не маскируется — как на тесте)"))
    # кривые в JSON держим, но не печатаем
    # by_confidence намеренно выбрасывается и дописывается ПОСЛЕ остальных ключей: json.dump
    # сохраняет порядок вставки, поэтому свёртка в {**ref, "by_confidence": ...} оставила бы
    # ключ на его исходной позиции и val-json перестал бы быть побайтово тем же. Значения тут
    # не копируются — chosen/curve это те же объекты, что внутри ref.
    res["refusal"] = {k: v for k, v in ref.items() if k != "by_confidence"}
    res["refusal"]["by_confidence"] = {
        conf_name: {"chosen": entry["chosen"], "pr_auc": entry["pr_auc"]}
        for conf_name, entry in ref["by_confidence"].items()}
    res["refusal"]["curves"] = {conf_name: entry["curve"]
                                for conf_name, entry in ref["by_confidence"].items()}

    path = _val_json(cfg, backbone_name, bool(post.get("tta_flip")), bool(post.get("fast_decode")))
    if mask_mode:
        path = path.with_name(f"{path.stem}_mask-{mask_mode}.json")
    _dump_result_json(res, path, "val")
    return res


def cmd_submit(cfg, backbone_name, batch_size, workers, device, out_dir, threshold, confidence, post=None):
    """Сдача по test-данным: три артефакта в out_dir.

    embeddings.npy — векторы в порядке query, затем gallery; submission.csv — top-10 кандидатов
    на запрос; candidates.csv — принятые по порогу запросы (в метрику идёт только топ-1).

    Два формата входа: явные query/gallery (формат организаторов) и один test.csv, где каждая
    запись одновременно и запрос, и галерея (shared=True, самосовпадения блокируются).

    Порог, тип уверенности и параметры локального сопоставления берутся из val-json ТОГО ЖЕ
    бэкбона и ТОГО ЖЕ TTA — иначе шкала уверенности не та. Без такого файла обязательны
    --threshold и --confidence.
    """
    from .rerank import dba, frame_block_mask, k_reciprocal
    from .submit import save_embeddings, write_candidates, write_submission
    post = post or {}
    tests = load_test(cfg["dataset"])
    run = _run_dir(cfg, backbone_name)
    crop = cfg.get("crop", {})
    backbone = get_backbone(backbone_name, device=device)
    tta = bool(post.get("tta_flip"))
    def extract_for_split(split):
        """Эмбеддинги одного сплита теста с кэшем в run/test_<имя>[_flip].npz."""
        return extract_split(split, backbone, batch_size, workers,
                             run / f"test_{split.name}{'_flip' if tta else ''}.npz",
                             pad=crop.get("pad", 0.08), mask=None, tta_flip=tta)

    if "query" in tests:                       # явные query / gallery (формат организаторов)
        tq, tg = extract_for_split(tests["query"]), extract_for_split(tests["gallery"])
        shared = False
    else:                                      # один test.csv: каждая запись — запрос ко всем остальным
        tq = tg = extract_for_split(tests["test"])
        shared = True
    q_keys, g_keys = list(tq["keys"]), list(tg["keys"])

    # Намеренно только tta: submit всегда читает НЕ-fast, НЕ-mask валидацию — это калибровка
    # релиза (см. docstring _val_json). Флаг --fast-decode на submit не действует, экстрактор
    # выше его тоже не передаёт; менять — только обе точки сразу и с перезамером порога.
    val_json = _val_json(cfg, backbone_name, tta)
    tau, val_res = None, None
    if val_json.exists():
        with open(val_json, encoding="utf-8") as f:
            val_res = json.load(f)
        tau = val_res["bits"]["calibration"]["tau"]
    if threshold is None or confidence is None:
        if val_res is None:
            raise SystemExit(f"нет {val_json}: запусти val{' --tta-flip' if tta else ''} "
                             f"(калибровка порога должна быть с тем же TTA), или задай --threshold и --confidence")
        # Сначала confidence: он индексирует by_confidence строкой ниже, перестановка даст
        # KeyError: None при запуске без --confidence. threshold проверяется через
        # `is not None` намеренно — 0.0 валидный порог, `or` его бы молча выбросил и подставил
        # порог из val.
        confidence = confidence or val_res["refusal"]["best_confidence"]
        threshold = (threshold if threshold is not None
                     else val_res["refusal"]["by_confidence"][confidence]["chosen"]["threshold"])
        print(f"[submit] порог из val (протокол {val_res.get('protocol', 'id')}"
              f"{', TTA flip' if val_res.get('tta_flip') else ''}): "
              f"уверенность={confidence}, threshold={threshold:.4f}")
        if val_res.get("refusal_mask_cam", True):
            print("[submit] ВНИМАНИЕ: порог калиброван с маской своей камеры, а на тесте камер нет — "
                  "ожидай заниженную долю отказов. Лучше protocol: track в конфиге и заново val.")

    q_emb, g_emb = tq["emb"], tg["emb"]
    # пары, которые нельзя считать соседями: та же запись и тот же кадр
    # block намеренно по ОБЪЕДИНЁННОМУ множеству (Q+G)²: такую маску ждут dba(block_mask=) и
    # k_reciprocal(same_mask=) ниже. qg_block — только вид на правый верхний угол для
    # маскирования sims. Заменить на frame_block_mask(q_keys, g_keys) нельзя: qg_block численно
    # совпадёт и тесты не покраснеют, а DBA и k-reciprocal получат маску неверной формы.
    if shared:
        block = frame_block_mask(q_keys, q_keys)
        np.fill_diagonal(block, True)
    else:
        allk = q_keys + g_keys
        block = frame_block_mask(allk, allk)
        np.fill_diagonal(block, True)
    qg_block = block if shared else block[: len(q_keys), len(q_keys):]

    # УВЕРЕННОСТЬ ДЛЯ ОТКАЗА — строго по сырым векторам: порог калиброван на val без DBA,
    # а DBA поднимает сходства и иначе почти ничего не отказывается.
    # raw_sims мутируется на месте намеренно: -inf здесь означает «нет в галерее» при счёте
    # bits (refusal.confidence_scores: n_g = isfinite(...).sum(axis=1)) и одновременно
    # «принудительный отказ» в write_candidates(conf_sims=raw_sims) ниже. Замена на копию
    # (np.where(qg_block, -inf, raw_sims)) или пересчёт raw_sims перед write_candidates
    # изменит и биты, и candidates.csv.
    raw_sims = q_emb @ g_emb.T
    raw_sims[qg_block] = -np.inf
    # из raw_order читаются только top1/top2, поэтому ничьи тут не важны; но править одну
    # строку argsort без другой (см. order ниже) нельзя
    raw_order = np.argsort(-raw_sims, axis=1)
    scores = confidence_scores(raw_sims, raw_order, tau)
    # Дубль формулы из refusal.evaluate_refusal (refusal.py:119-120): там этим же выражением
    # на val строится скор, для которого калибруется порог, здесь его надо повторить БИТ В БИТ —
    # иначе threshold из val не про этот скор. Веса 0.02 обязаны меняться только вдвоём;
    # расхождение тихо сдвинет долю отказов, без падения mAP и без ошибки. Масштаб подобран так,
    # чтобы биты только доразделяли близкие top1, а не переупорядочивали их.
    # Удалять эти две строки нельзя: confidence_scores ключ top1+bits не считает, и
    # `scores[confidence]` ниже упадёт KeyError ровно в обычном случае, когда val выбрал
    # top1+bits.
    if "bits" in scores:
        scores["top1+bits"] = scores["top1"] + 0.02 * scores["bits"]
    conf = scores[confidence]
    top1 = scores["top1"]
    print(f"[submit] top-1 сходство на тесте: медиана {np.median(top1):.3f}, "
          f"10%={np.quantile(top1, .1):.3f}, "
          f"90%={np.quantile(top1, .9):.3f}; порог {threshold:.3f} → "
          f"отказов ожидается {(conf < threshold).mean() * 100:.1f}%")
    if val_res is not None and "refusal" in val_res:
        ar = (val_res["refusal"]["by_confidence"]
              .get(confidence, {}).get("chosen", {}).get("accept_rate"))
        if ar is not None:
            print(f"[submit] для сравнения на val при этом пороге принималось {ar * 100:.1f}% запросов")

    if post.get("dba"):
        stacked = q_emb if shared else np.concatenate([q_emb, g_emb])
        refined = dba(stacked, k=int(post["dba"]), block_mask=block)
        if shared:
            q_emb = g_emb = refined
        else:
            q_emb, g_emb = refined[: len(q_emb)], refined[len(q_emb):]
        print(f"[submit] DBA k={post['dba']}: векторы уточнены по взаимным соседям")
    sims = q_emb @ g_emb.T
    sims[qg_block] = -np.inf
    # Сортировка здесь намеренно дефолтная, в отличие от metrics.py/refusal.py/submit.py: этим
    # order выбираются кандидаты для локального сопоставления, и переход на kind="stable"
    # переставил бы ничьи и изменил submission.csv при --local. Переписывать через
    # np.argsort(sims)[:, ::-1] тоже нельзя — это переворачивает ничьи гарантированно.
    # Любое изменение — только с замером scripts/check_refactor_equivalence.py.
    order = np.argsort(-sims, axis=1)

    rank_sims = sims
    if post.get("local") and val_res is not None and "local" in val_res:
        from .local_match import LocalFeatures, local_scores, fuse
        local_cfg = val_res["local"]
        local_feats = LocalFeatures(backbone, size=int(local_cfg["size"]), pca_dim=64)
        # Порядок важен: LocalFeatures подгоняет PCA на ПЕРВОМ сплите (local_match.py:95-99,
        # `if self.P is None: self.fit_pca(...)`) — базис берётся по query, галерея проецируется
        # в него же. Переставить вызовы, завести отдельный LocalFeatures на сплит или
        # распараллелить = другие токены → другая beta → другой submission. PCA может
        # подогнаться и посреди цикла, поэтому базис зависит ещё и от batch_size.
        q_local = local_feats.extract(tests.get("query", tests.get("test")),
                                      run / f"test_query_local{local_cfg['size']}.npz",
                                      pad=crop.get("pad", 0.08), mask_frac=None,
                                      batch_size=batch_size, workers=workers)
        g_local = q_local if shared else local_feats.extract(
            tests["gallery"], run / f"test_gallery_local{local_cfg['size']}.npz",
            pad=crop.get("pad", 0.08), mask_frac=None, batch_size=batch_size, workers=workers)
        # -1 в топе = «кандидата нет», позиция закрыта маской кадра/записи (см. _valid_order)
        order_top = _valid_order(sims, order)
        local_sims = local_scores(q_local["tokens"], g_local["tokens"], order_top, q_local["grid"],
                                  topk=int(local_cfg["topk"]), min_sim=float(local_cfg["min_sim"]))
        rank_sims = fuse(sims, order_top, local_sims, float(local_cfg["beta"]),
                         int(q_local["grid"][0] * q_local["grid"][1]))
        print(f"[submit] локальное сопоставление: top-{local_cfg['topk']}, "
              f"beta={local_cfg['beta']}")
    if post.get("kr"):
        k1, k2 = int(post.get("k1", 20)), int(post.get("k2", 6))
        if shared:
            dist = k_reciprocal(q_emb, q_emb, k1, k2, KR_LAMBDA, same_mask=block, shared=True)
            dist[block] = np.inf
        else:
            dist = k_reciprocal(q_emb, g_emb, k1, k2, KR_LAMBDA, same_mask=block)
            dist[~np.isfinite(sims)] = np.inf
        rank_sims = -dist          # k_reciprocal отдаёт расстояния → знак
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
    векторов (с весами) → те же метрики, постобработка и режим отказа. Ничего не пересчитывает.

    ВНИМАНИЕ про отчётную цифру: здесь считается ТОЛЬКО строгий cross-camera
    (cross_camera_only=True), он завышает mAP на 1-3 п.п. относительно метрики жюри. Ключа
    "jury" в ens-JSON нет, поэтому scripts/export_release.py (строка 106,
    `v.get("jury") or v["cross_camera"]`) подставит эту завышенную цифру как val_mAP_jury —
    для отчётных чисел брать val, а не ens.

    Побочные эффекты: пишет runs/<name>/ens_.../val_*.npz и results/<name>_ens_..._val.json,
    возвращает res."""
    val_split = build_local_validation(cfg["dataset"])
    has_match = val_split["has_match"]
    query_parts, gallery_parts = [], []
    for i, backbone in enumerate(backbones):
        run = _run_dir(cfg, backbone)
        q_npz, g_npz = _val_files(val_split, run)
        if not q_npz.exists():
            raise SystemExit(f"нет {q_npz}: сначала прогони val для {backbone} "
                             f"с текущим протоколом")
        q, g = load_npz(q_npz), load_npz(g_npz)
        weight = float(weights[i]) if weights else 1.0
        query_parts.append(q["emb"] * weight)
        gallery_parts.append(g["emb"] * weight)
        print(f"[ens] {backbone}: D={q['emb'].shape[1]}, вес {weight}")
        if i == 0:
            # vids/cams/keys у всех участников ансамбля одни и те же — берём у первого
            q_meta, g_meta = q, g
    q_emb = np.concatenate(query_parts, axis=1)
    g_emb = np.concatenate(gallery_parts, axis=1)
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
    g_emb /= np.linalg.norm(g_emb, axis=1, keepdims=True)
    # Именно {**q_meta, "emb": ...}: q_meta задаёт набор и ПОРЯДОК ключей для np.savez ниже,
    # а emb перекрывается ансамблевым вектором. Обратный порядок ({"emb": ..., **q_meta}) тихо
    # вернул бы эмбеддинг первого бэкбона — с правильной формой файла и без исключения.
    q = {**q_meta, "emb": q_emb.astype(np.float32)}
    g = {**g_meta, "emb": g_emb.astype(np.float32)}
    sims = q_emb @ g_emb.T
    name = "ens_" + "+".join(_slug(backbone) for backbone in backbones)
    res = {"dataset": cfg["name"], "backbone": name, "members": backbones, "weights": weights,
           "dim": int(q_emb.shape[1]),
           "protocol": val_split["protocol"], "refusal_mask_cam": val_split["refusal_mask_cam"]}
    res["cross_camera"] = evaluate(sims[has_match], q["vids"][has_match], q["cams"][has_match],
                                   g["vids"], g["cams"], cross_camera_only=True, cutoff=10)
    print(format_metrics(res["cross_camera"], f"ансамбль {len(backbones)} моделей (cross-cam)"))
    res["postprocess"] = _ablate_postprocess(q, g, has_match, sims)
    cal = bitsmod.calibrate_tau(sims[has_match], q["vids"][has_match], q["cams"][has_match],
                                g["vids"], g["cams"], recall=0.95)
    res["bits"] = {"calibration": cal}
    ref = evaluate_refusal(sims, q["vids"], q["cams"], g["vids"], g["cams"], has_match,
                           tau=cal["tau"],
                           min_tnr=(cfg.get("refusal") or {}).get("min_tnr"),
                           cross_camera_only=val_split["refusal_mask_cam"])
    print(format_refusal(ref))
    res["refusal"] = {k: v for k, v in ref.items() if k != "by_confidence"}
    res["refusal"]["by_confidence"] = {
        conf_name: {"chosen": entry["chosen"], "pr_auc": entry["pr_auc"]}
        for conf_name, entry in ref["by_confidence"].items()}
    out = _run_dir(cfg, name)
    out.mkdir(parents=True, exist_ok=True)
    out_query_npz, out_gallery_npz = _val_files(val_split, out)
    np.savez(out_query_npz, **q)
    np.savez(out_gallery_npz, **g)
    path = _results(cfg) / f"{cfg['name']}_{name}_val.json"
    _dump_result_json(res, path, "ens")
    return res


def _contact_sheet(paths, n_per_cam):
    """Склеивает первые n_per_cam кадров камеры в одну горизонтальную полосу миниатюр."""
    from PIL import Image
    thumbs = []
    for img_path in paths[:n_per_cam]:
        with Image.open(img_path) as im:
            im.thumbnail(_THUMB)
            thumbs.append(im.copy())
    sheet = Image.new("RGB", (_THUMB[0] * len(thumbs), _THUMB[1]), (0, 0, 0))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, (_THUMB[0] * i, 0))
    return sheet


def cmd_inspect_cams(cfg, n_per_cam=4):
    # импорт PIL здесь намеренно: падать «нет PIL» надо сразу, а не после долгого чтения
    # аннотаций; сам Image используется в _contact_sheet, который импортирует его отдельно
    from PIL import Image  # noqa: F401
    from .hackathon_data import assign_pseudo_cameras, read_annotations
    ds = cfg["dataset"]
    root = Path(ds["root"])
    recs = read_annotations(root / ds.get("train_csv", "train.csv"), root / ds.get("images_dir", "images"),
                            ds.get("cols"), ds.get("bbox_format", "xywh"))
    recs = assign_pseudo_cameras(recs, ds.get("pseudo_camera_threshold", 0.35))
    by_cam = {}
    for rec in recs:
        by_cam.setdefault(rec.cam, set()).add(rec.path)
    out = _results(cfg) / "pseudo_cameras"
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for cam, paths in sorted(by_cam.items()):
        sorted_paths = sorted(paths)
        lines.append(f"cam {cam}: {len(sorted_paths)} кадров")
        sheet = _contact_sheet(sorted_paths, n_per_cam)
        sheet.save(out / f"cam_{cam:04d}.jpg", quality=80)
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:30]))
    print(f"[inspect-cams] листы по камерам → {out}")


def _build_parser() -> argparse.ArgumentParser:
    """Парсер CLI: подкоманда ens стоит особняком, остальные три собираются одним циклом.

    Порядок add_argument определяет порядок флагов в --help, поэтому переставлять их нельзя.
    """
    parser = argparse.ArgumentParser(prog="vreid.hack_cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ens_parser = sub.add_parser("ens",
                                help="ансамбль по готовым val-эмбеддингам: --backbones A B [C]")
    ens_parser.add_argument("--config", required=True)
    ens_parser.add_argument("--backbones", nargs="+", required=True)
    ens_parser.add_argument("--weights", nargs="+", type=float, default=None)
    # Общий набор флагов для трёх команд сделан одним циклом ради единообразия — лишние флаги
    # команда просто игнорирует, поэтому область действия указана в help каждого флага.
    for name in ("val", "submit", "inspect-cams"):
        sub_parser = sub.add_parser(name)
        sub_parser.add_argument("--config", required=True)
        sub_parser.add_argument("--backbone", default="dinov2_b")
        sub_parser.add_argument("--batch-size", type=int, default=64)
        sub_parser.add_argument("--workers", type=int, default=4)
        sub_parser.add_argument("--device", default=None)
        sub_parser.add_argument("--reuse", action="store_true",
                                help="val: не пересчитывать эмбеддинги")
        sub_parser.add_argument("--out", default="submission", help="submit: папка артефактов")
        sub_parser.add_argument("--threshold", type=float, default=None)
        sub_parser.add_argument("--confidence", default=None,
                                help="top1 | margin | bits | top1+bits")
        sub_parser.add_argument("--rerank", action="store_true",
                                help="val: абляции DBA / k-reciprocal")
        sub_parser.add_argument("--local", action="store_true",
                                help="локальное сопоставление патчей для top-K")
        sub_parser.add_argument("--local-size", type=int, default=336,
                                help="val: сторона кропа для токенов; в submit берётся из val-JSON")
        sub_parser.add_argument("--local-topk", type=int, default=30,
                                help="val: сколько кандидатов пересопоставлять; "
                                     "в submit берётся из val-JSON")
        sub_parser.add_argument("--local-min-sim", type=float, default=0.6,
                                help="val: порог сходства патчей для инлайера; "
                                     "в submit берётся из val-JSON")
        sub_parser.add_argument("--dba", type=int, default=0,
                                help="submit: уточнить векторы по k взаимным соседям")
        sub_parser.add_argument("--kr", action="store_true",
                                help="submit: k-reciprocal для submission.csv")
        sub_parser.add_argument("--k1", type=int, default=20,
                                help="submit: k1 для k-reciprocal (см. таблицу val --rerank)")
        sub_parser.add_argument("--k2", type=int, default=6, help="submit: k2 для k-reciprocal")
        sub_parser.add_argument("--cand-min", type=float, default=None,
                                help="не писать в candidates.csv кандидатов слабее этого косинуса")
        sub_parser.add_argument(
            "--cand-topk", type=int, default=1,
            help="строк на принятый запрос в candidates.csv (в метрику идёт только топ-1)")
        sub_parser.add_argument(
            "--mask-plate", nargs="?", const="det", default=None,
            choices=["det", "auto", "band", "up", "rand"],
            help="val: закрасить зону номера; в submit игнорируется. det — только найденную "
                 "детектором, auto — плюс запасная полоса там, где не нашли; up/rand — "
                 "КОНТРОЛЬ той же площади в другом месте (отделяет «потеряли номер» от "
                 "«потеряли пиксели»); band — старый грубый прямоугольник")
        sub_parser.add_argument(
            "--fast-decode", action="store_true",
            help="val: ускоренное декодирование JPEG (как в боевом инференсе); "
                 "в submit игнорируется")
        sub_parser.add_argument(
            "--tta-flip", action="store_true",
            help="усреднить вектор кропа с вектором его зеркала (инференс ×2 по времени)")
    return parser


def _post_from_args(args) -> dict:
    """Собирает словарь post — параметры постобработки, общие для val и submit.

    getattr со значениями по умолчанию — страховка на случай вызова main() с урезанным
    парсером; для ens выход происходит выше, поэтому здесь атрибуты всегда есть.
    ВАЖНО: дефолты дублируют add_argument — при правке менять в обоих местах."""
    return {"rerank": getattr(args, "rerank", False), "local": getattr(args, "local", False),
            "local_size": getattr(args, "local_size", 336),
            "local_topk": getattr(args, "local_topk", 30),
            "local_min_sim": getattr(args, "local_min_sim", 0.6),
            "dba": getattr(args, "dba", 0), "kr": getattr(args, "kr", False),
            "k1": getattr(args, "k1", 20), "k2": getattr(args, "k2", 6),
            "tta_flip": getattr(args, "tta_flip", False),
            "fast_decode": getattr(args, "fast_decode", False),
            "mask_plate": getattr(args, "mask_plate", None),
            "cand_min": getattr(args, "cand_min", None),
            "cand_topk": getattr(args, "cand_topk", 1)}


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "ens":
        cfg = _load_config(args.config)
        cmd_ens(cfg, args.backbones, args.weights)
        return
    post = _post_from_args(args)
    cfg = _load_config(args.config)
    if args.cmd == "val":
        cmd_val(cfg, args.backbone, args.batch_size, args.workers, args.device, args.reuse, post)
    elif args.cmd == "submit":
        cmd_submit(cfg, args.backbone, args.batch_size, args.workers, args.device, args.out,
                   args.threshold, args.confidence, post)
    elif args.cmd == "inspect-cams":
        cmd_inspect_cams(cfg)


if __name__ == "__main__":
    main()
