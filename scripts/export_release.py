"""Сборка релиза для сдачи и для бэкенда: веса fp16 + замороженный рецепт.

    python scripts/export_release.py --weights weights/hack_dinov2_l_336/best.pt \
        --val results/hack_ft_hack_dinov2_l_336_flip_val.json \
        --kr --k1 6 --k2 2 --threshold 0.55

Делает release/model.pt (fp16, вдвое меньше — у нас 1.21 ГБ → ~0.61 ГБ при лимите 2 ГБ)
и release/recipe.json: порог отказа, вид уверенности, tau для битов, постобработка.
Дальше инференс запускается одной командой и без конфигов: python -m vreid.predict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="weights/<run>/best.pt")
    ap.add_argument("--val", required=True, help="results/..._val.json, из которого берём порог")
    ap.add_argument("--out", default="release")
    ap.add_argument("--confidence", default="top1",
                    help="вид уверенности; auto — взять лучшую по val (осторожно, см. комментарий)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--dba", type=int, default=0)
    ap.add_argument("--kr", action="store_true")
    ap.add_argument("--k1", type=int, default=10)
    ap.add_argument("--k2", type=int, default=3)
    ap.add_argument("--crop-pad", type=float, default=0.05)
    ap.add_argument("--cand-min", type=float, default=None,
                    help="порог косинуса для попадания в candidates.csv (не влияет на submission.csv)")
    ap.add_argument("--cand-topk", type=int, default=1,
                    help="строк на принятый запрос в candidates.csv; в метрику идёт только топ-1")
    ap.add_argument("--refuse-rate", type=float, default=None,
                    help="перенести на тест ДОЛЮ отказов вместо порога. Работает только если доля "
                         "запросов без пары на тесте такая же, как на валидации — у нас это не так "
                         "(на валидации 20%% по построению, на тесте ~3%% по замеру), поэтому "
                         "по умолчанию выключено.")
    ap.add_argument("--rate-threshold", action="store_true",
                    help="взять долю отказов из рабочей точки валидации (1 - accept_rate)")
    ap.add_argument("--no-fast-decode", action="store_true",
                    help="отключить ускоренное декодирование JPEG (по умолчанию включено)")
    ap.add_argument("--fp32", action="store_true", help="не переводить веса в fp16")
    a = ap.parse_args()
    if a.dba != 0 or a.refuse_rate is not None or a.rate_threshold:
        ap.error('Competition export forbids DBA and test-query quantile thresholds')
    if a.confidence != 'top1' or a.cand_topk != 1 or a.cand_min is not None:
        ap.error('Competition export requires selected-candidate cosine and one candidate')

    import torch

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(a.val, encoding="utf-8") as f:
        v = json.load(f)
    if v.get("tta_flip", False) != a.tta_flip:
        raise SystemExit(f"{a.val} посчитан с tta_flip={v.get('tta_flip')}, а релиз собирается с "
                         f"tta_flip={a.tta_flip} — порог будет не от тех эмбеддингов")
    # ПОЧЕМУ ПО УМОЛЧАНИЮ top1, А НЕ «лучшая по val».
    # bits = -log2(кандидатов в радиусе / размер галереи) явно зависит от размера галереи, а он
    # у валидации (688) и у теста (750) разный; замер gallery_size_effect.py показывает, что
    # только от размера галереи оптимальный порог ездит на порядки больше, чем от всего остального.
    # Чистый косинус top1 такой зависимости не имеет и переносится (измерено:
    # threshold_transfer.json — сдвиг медианы +0.0025, перевернулось бы 1 решение из 1110).
    # Автовыбор однажды уже молча подсунул top1+bits в релиз-кандидат: разница на val была
    # 0.04 балла, а риск на тесте — весь блок отказа. Нужен автовыбор — пиши --confidence auto явно.
    conf = v["refusal"]["best_confidence"] if a.confidence == "auto" else a.confidence
    if conf not in v["refusal"]["by_confidence"]:
        raise SystemExit(f"в {a.val} нет уверенности «{conf}»; есть: "
                         f"{', '.join(v['refusal']['by_confidence'])}")
    if conf != "top1":
        print(f"[release] ВНИМАНИЕ: уверенность «{conf}», а не top1. Если в неё входят bits — "
              f"порог будет зависеть от размера галереи, а он на тесте другой (750 против 688)")
    thr = a.threshold if a.threshold is not None else v["refusal"]["by_confidence"][conf]["chosen"]["threshold"]
    chosen = v["refusal"]["by_confidence"][conf]["chosen"]

    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    if not a.fp32:
        for part in ("backbone", "bnneck"):
            ck[part] = {k: (t.half() if t.is_floating_point() else t) for k, t in ck[part].items()}
    torch.save(ck, out / "model.pt")
    mb = (out / "model.pt").stat().st_size / 1024 ** 2

    recipe = {"threshold": float(thr), "confidence": conf,
              "tau": float(v["bits"]["calibration"]["tau"]),
              "tta_flip": bool(a.tta_flip), "dba": int(a.dba), "kr": bool(a.kr),
              "k1": int(a.k1), "k2": int(a.k2),
              "crop_pad": float(a.crop_pad), "mask_plate": False, "topk": 10,
              "candidate_min_sim": (None if a.cand_min is None else float(a.cand_min)),
              "candidates_topk": int(a.cand_topk), "fast_decode": not a.no_fast_decode,
              # Переносится АБСОЛЮТНЫЙ порог, не доля отказов.
              # Доля не переносится на ОТКРЫТЫЙ тест: там у каждого запроса есть пара (ответ 17),
              # то есть 0% open-set против 20% на валидации, и перенос доли заставлял отказывать
              # 24% вместо 8%. В ЗАКРЫТОМ тесте open-set доля ~20% — ровно как на валидации,
              # поэтому там оба способа близки; абсолютный выбран потому, что он не требует знать
              # эту долю заранее и не связывает запросы друг с другом (квантиль — это статистика
              # по всему тесту сразу, а запросы обязаны оставаться независимыми, ответ 38).
              "refuse_rate": (a.refuse_rate if a.refuse_rate is not None
                              else (round(1.0 - float(chosen["accept_rate"]), 4)
                                    if a.rate_threshold else None)),
              "source": {"weights": a.weights, "val": a.val, "protocol": v.get("protocol"),
                         # ГЛАВНОЕ число — метрика жюри (убраны только пары vid+cam, mAP@10, ответ 11).
                         # cross_camera в старых JSON — строгий режим, он завышает на 1-3 пункта;
                         # держим оба, чтобы никто не сравнивал разное как одинаковое.
                         "val_mAP_jury": (v.get("jury") or v["cross_camera"])["mAP"],
                         "val_rank1_jury": (v.get("jury") or v["cross_camera"])["rank1"],
                         "val_mAP_cross_camera_strict": v["cross_camera"]["mAP"],
                         "val_mINP": v["refusal"].get("mINP"),
                         "val_n_query": v.get("n_query"), "val_n_no_match": v.get("n_query_no_match"),
                         "val_n_gallery": v.get("n_gallery"),
                         "val_refusal_mask_cam": v.get("refusal_mask_cam"),
                         "val_refusal": {k: chosen[k] for k in ("f1", "precision", "recall", "tnr")}}}
    with open(out / "recipe.json", "w", encoding="utf-8") as f:
        json.dump(recipe, f, ensure_ascii=False, indent=2)

    print(f"[release] model.pt {mb:.0f} МБ ({'fp32' if a.fp32 else 'fp16'}), лимит ТЗ — 2048 МБ")
    print(f"[release] frozen threshold {thr:.4f}; evaluate emitted CSV with scripts/refusal_audit.py")
    print(f"[release] source validation point {chosen['threshold']:.4f}: F1={chosen['f1'] * 100:.1f}%, TNR={chosen['tnr'] * 100:.1f}% (not a measurement of this exported recipe)")
    rr = recipe["refuse_rate"]
    print(f"[release] отказов в рабочей точке: {(1 - chosen['accept_rate']) * 100:.1f}% "
          + (f"→ на тесте порог возьмётся как квантиль {rr * 100:.1f}%" if rr is not None
             else "(используется абсолютный порог)"))
    j = v.get("jury") or v["cross_camera"]
    print(f"[release] val mAP@10 по правилу жюри {j['mAP'] * 100:.1f}% "
          f"(строгий cross-cam для сравнения {v['cross_camera']['mAP'] * 100:.1f}%, протокол {v.get('protocol')})")
    if v.get("refusal_mask_cam"):
        print("[release] ВНИМАНИЕ: порог калиброван с маской своей камеры, а на тесте камер нет — "
              "пересобери val с protocol: track")
    print(f"[release] → {out}/model.pt, {out}/recipe.json")


if __name__ == "__main__":
    main()
