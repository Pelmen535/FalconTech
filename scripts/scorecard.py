# -*- coding: utf-8 -*-
"""Сводная карточка ядра: что у нас по каждому блоку оценки, из каких файлов взято.

    python scripts/scorecard.py --release release --submission submission

Собирает в одну таблицу всё, что жюри смотрит по ядру, и рядом — файл, из которого число
взято. Если файла нет, строка так и говорит «нет замера», а не подставляет старое значение:
подставленное старое число один раз уже уехало в README и жило там сутки.

Веса блоков по ответам организаторов: ранжирование 45%, отказ 10%, производительность 20%
(10 латентность + 10 пропускная). Оставшиеся 25% — защита и продукт, здесь не считаются.
Как именно mAP переводится в баллы, организаторы не публиковали, поэтому в колонке «баллы»
для ранжирования стоит линейное допущение и оно помечено как допущение.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="release")
    ap.add_argument("--submission", default="submission_soup")
    ap.add_argument("--results", default="results")
    ap.add_argument("--val", default=None, help="по умолчанию берётся из recipe.source.val")
    a = ap.parse_args()

    R, S, RES = Path(a.release), Path(a.submission), Path(a.results)
    rec = load(R / "recipe.json") or {}
    src = rec.get("source", {})
    val = load(Path(a.val or src.get("val", "")))
    bench = load(RES / "bench_full.json")
    run = load(S / "run_info.json")
    tr = load(RES / "threshold_transfer.json")
    gs = load(RES / "gallery_size_effect.json")

    import hashlib
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    model_hash=digest(R/'model.pt') if (R/'model.pt').is_file() else None
    recipe_hash=digest(R/'recipe.json') if (R/'recipe.json').is_file() else None
    if bench and (bench.get('model_sha256')!=model_hash or bench.get('recipe_sha256')!=recipe_hash):
        print('Benchmark does not match exact model/recipe hashes; excluded from current scorecard.')
        bench=None
    print('Local diagnostic card; no official total score or A5000 result is asserted.')
    rows = []
    add = lambda *r: rows.append(r)

    if val:
        j = val.get("jury") or val.get("cross_camera")
        add("Ранжирование", "mAP@10 без постобработки", f"{j['mAP'] * 100:.1f}",
            f"{45 * j['mAP']:.1f} / 45 (допущение: линейно)", str(Path(src.get("val", ""))))
        post = val.get("postprocess") or {}
        pr = post.get("rows") if isinstance(post, dict) else (post if isinstance(post, list) else [])
        # DBA и query expansion запрещены ответом 38 — их строки в балл идти не могут,
        # но показываем лучшую из них отдельной строкой: чтобы никто не сравнивал наши числа
        # с чужими, где DBA не выключен, и чтобы разница была видна на защите.
        ok = [r for r in pr if "dba" not in str(r.get("name", "")).lower()]
        bad = [r for r in pr if "dba" in str(r.get("name", "")).lower()]
        pick = lambda rs: max(rs, key=lambda r: r.get("mAP", 0)) if rs else None
        # Строка берётся ПО РЕЦЕПТУ, а не как максимум таблицы: иначе после смены рецепта
        # отчёт показал бы чужой, лучший исследовательский вариант и подписал его «в релизе».
        want = (f"k-recip {rec.get('k1')}/{rec.get('k2')}" if rec.get("kr") else "base")
        # Сначала ищем точку с ТОЧНЫМИ параметрами рецепта в результатах подбора: таблица
        # абляций считает k-reciprocal с фиксированной lambda 0.3, а в рецепте она своя,
        # и строка из таблицы описывала бы другой вариант, чем тот, что сдаётся.
        b = None
        if rec.get("kr"):
            # Таблица постобработки снимается редко и содержит только ту сетку, что была
            # тогда. Точка текущего рецепта может лежать в результате подбора параметров —
            # это то же самое измерение на том же кеше, просто в другом файле.
            grid = load(RES / "rerank_grid.json") or load(RES / "rerank_grid_fine.json")
            key = f"{rec.get('k1')}/{rec.get('k2')}/{float(rec.get('lam', 0.3)):g}"
            source_run = Path(src.get("weights", "")).parent.name
            for run_name, values in ((grid or {}).get("runs") or {}).items():
                if key in values and (source_run in run_name or "fit" in run_name):
                    b = {"name": f"k-recip {key}", "mAP": values[key] / 100,
                         "from_grid": run_name}
                    break
        if b is None:
            b = next((r for r in ok if r.get("name") == want), None)
        if b is None:
            add("Ранжирование", f"НЕ НАШЁЛ строки «{want}» в таблице постобработки",
                "—", "пересчитай val для этого рецепта", "тот же файл")
        if b and b.get("name") != "base":
            where = ("results/rerank_grid.json, " + b["from_grid"]) if b.get("from_grid") \
                else "тот же файл, блок postprocess"
            add("Ранжирование", f"mAP@10 с постобработкой ({b['name']}, validation; перенос не измерен)",
                f"{b['mAP'] * 100:.1f}", f"{45 * b['mAP']:.1f} / 45 (допущение: линейно)", where)
        f = pick(bad)
        if f:
            add("Ранжирование", f"— для сравнения: {f['name']} ЗАПРЕЩЕНО (ответ 38)",
                f"{f['mAP'] * 100:.1f}", "не идёт в балл", "тот же файл, блок postprocess")
        # Каскад меряется отдельным скриптом на паре кешей, а не в val-отчёте одной модели:
        # в нём участвуют две модели сразу. Берём ЧЕСТНЫЙ замер — с ре-ранкером, который
        # не выбирал лучшую эпоху по этой же валидации.
        if rec.get("cascade"):
            casc = load(RES / "cascade_verify_honest.json") or load(RES / "cascade_verify.json")
            alpha = f"{float(rec.get('cascade_alpha', 0.9)):g}"
            pair = next(iter((casc or {}).get("pairs", {}).items()), None)
            point = (pair[1]["mixed"].get(alpha) if pair else None)
            if point:
                add("Ранжирование",
                    f"mAP@10 с каскадом ViT-L по топ-{rec.get('cascade_topk')} "
                    f"(alpha={alpha}, validation; перенос не измерен)",
                    f"{point['mAP']:.1f}", f"{45 * point['mAP'] / 100:.1f} / 45 (допущение: линейно)",
                    f"results/cascade_verify_honest.json, {pair[0]}")
            else:
                add("Ранжирование", "рецепт включает каскад, но замера для этой alpha нет",
                    "—", "прогони scripts/cascade_verify.py", "results/cascade_verify_honest.json")
        # Блок отказа считается ДЛЯ ФАКТИЧЕСКОГО порога релиза, а не для точки максимума
        # из val-отчёта. Порог сознательно смещён ниже оптимума (см. threshold_choice в рецепте),
        # и подставлять сюда 97.2 от старого порога значило бы отчитываться не за то, что сдаём.
        thr = float(rec.get("threshold", 0))
        # Очереди пишут refusal_stress_fit.json (двойник релиза). Старое имя оставлено
        # запасным: подставлять числа из файла, который никто не обновляет, нельзя.
        stress_path = (RES / "refusal_stress_fit.json") if (RES / "refusal_stress_fit.json").is_file() \
            else (RES / "refusal_stress.json")
        st = load(stress_path)
        row = None
        if st:
            row = min(st["rows"], key=lambda r: abs(r["threshold"] - thr))
            if abs(row["threshold"] - thr) > 1e-6:
                row = None
        if row:
            A, B = row["A"], row["B"]
            add("Отказ", f"0.7*F1 + 0.3*TNR при пороге {thr:g} (F1 {A['f1']:.1f}, TNR {A['tnr']:.1f})",
                f"{A['score']:.1f}", "диагностика, не балл релиза", str(stress_path))
            add("Отказ", "— то же без одно-камерных дубликатов (стресс)",
                f"{B['score']:.1f}", "stress, не балл релиза", "тот же файл")
        else:
            ch = val["refusal"]["by_confidence"][rec.get("confidence", val["refusal"]["best_confidence"])]["chosen"]
            add("Отказ", f"НЕТ замера для порога {thr:g}; точка максимума val — {ch['threshold']:.4f}",
                "—", "прогони scripts/refusal_stress.py", "тот же файл")
        add("Отказ", "mINP (справочно, в балл не входит)",
            f"{val['refusal'].get('mINP', float('nan')) * 100:.1f}", "—", "тот же файл")
    else:
        add("Ранжирование", "нет файла валидации", "—", "—", str(src.get("val", "?")))

    if bench:
        lat, fps = bench["latency_ms_median"], bench["best_fps"]
        add("Производительность", f"латентность batch=1, медиана ({bench.get('n', '?')} прогонов)",
            f"{lat:.1f} мс", f"{10 * bench['latency_score']:.1f}/10, проекция на этом устройстве", "results/bench_full.json")
        add("Производительность", f"лучший FPS (batch {bench.get('best_batch')})",
            f"{fps:.1f}", f"{10 * bench['throughput_score']:.1f}/10, проекция на этом устройстве", "results/bench_full.json")
        # Замеряемая жюри латентность не включает rerank (ответы 31/32). Вторая строка —
        # настоящая цена запроса с форвардом ре-ранкера. В балл по правилам не идёт, но
        # показывать только первую значило бы скрывать половину картины.
        if bench.get("latency_ms_end_to_end"):
            add("Производительность", "— настоящая цена запроса с форвардом ре-ранкера",
                f"{bench['latency_ms_end_to_end']:.1f} мс",
                f"по правилам не в балл; будь иначе — {10 * bench['latency_score_if_reranker_counted']:.1f}/10",
                "results/bench_full.json")
        for k, label in (("weights_mb", "суммарный размер весов, МБ (лимит 2048)"),
                         ("weights_load_seconds", "время загрузки весов, с"),
                         ("peak_vram_mb", "пик VRAM, МБ"),
                         ("determinism_bitwise", "два прогона совпали бит в бит")):
            if k in bench and bench[k] is not None:
                add("Производительность", label, str(bench[k]), "—", "results/bench_full.json")
    else:
        add("Производительность", "нет замера", "—", "—", "results/bench_full.json")

    if run:
        add("Сдача", f"отказов на открытом тесте", f"{run['refused']} из {run['n_query']} "
            f"({run['refused'] / max(run['n_query'], 1) * 100:.1f}%)", "—", f"{S}/run_info.json")
        add("Сдача", "размерность вектора", str(run["dim"]), "—", f"{S}/run_info.json")
        add("Сдача", "весь прогон, с", f"{run['seconds_total']:.1f}", "—", f"{S}/run_info.json")
    if tr:
        add("Перенос порога", "сдвиг медианы уверенности релизной модели",
            f"{tr['median_shift']:+.4f}", "—", "results/threshold_transfer.json")
        add("Перенос порога", "решений перевернулось бы при выравнивании доли",
            f"{tr['decisions_flipped']} из {tr['n_query']}", "—", "results/threshold_transfer.json")
    if gs and gs.get("design") == "fixed_queries_all_positives_distractors_only" and gs.get("rows"):
        # Each size can have several seeded repetitions; compare means in percentage points.
        from collections import defaultdict
        grouped = defaultdict(list)
        for row in gs['rows']:
            grouped[row['gallery']].append(float(row['mAP10']) * 100)
        low, high = min(grouped), max(grouped)
        lo, hi = (sum(grouped[n]) / len(grouped[n]) for n in (low, high))
        add("Размер галереи", f"mAP@10 при галерее {low} против {high} (среднее повторов)",
            f"{lo:.1f} против {hi:.1f} ({hi - lo:+.1f} п.п.)", "—",
            "results/gallery_size_effect.json")

    w = [max(len(str(r[i])) for r in rows + [("блок", "что", "значение", "баллы", "откуда")]) for i in range(5)]
    head = ("блок", "что меряется", "значение", "баллы", "источник")
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
    print("  ".join("-" * w[i] for i in range(5)))
    for r in rows:
        print("  ".join(str(r[i]).ljust(w[i]) for i in range(5)))
    print("\nБлоки: ранжирование 45%, отказ 10%, производительность 20%. Защита и продукт (25%) не здесь.")


if __name__ == "__main__":
    main()
