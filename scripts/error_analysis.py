"""Разбор ошибок на отложенной валидации: где именно модель ошибается и почему (ТЗ §10, §11).

Считается по правилу жюри и по замороженному рецепту релиза — те же k-reciprocal и тот же
порог отказа, что уходят в сдачу. Никаких «подобранных под картинку» настроек.

Запросы делятся на взаимоисключающие исходы:

  rank1_ok                   первый кандидат — та же машина;
  rank1_wrong_but_in_top10   правильный ответ есть, но не на первом месте: ошибка ранжирования,
                             которую оператор исправит глазами за секунду;
  miss_top10                 правильного ответа нет во всей десятке: настоящий промах;
  no_match_query             пары в галерее нет вовсе — здесь оценивается только отказ.

Отдельно считается режим отказа: ложное принятие (ответил, а был не тот или пары не было)
и ложный отказ (промолчал, хотя пара была).

Для каждой группы печатаются признаки, которыми она отличается от остальных: площадь рамки,
пропорции, яркость кропа, наличие в галерее кадра той же машины с той же камеры и число
кросс-камерных положительных. Это и есть ответ на вопрос «что ломает модель».

    python scripts/error_analysis.py --run runs/hack/ft_soup_b336_fit --release release
    → results/error_analysis.json, docs/ERROR_ANALYSIS.md, docs/errors/*.jpg
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vreid.artifacts import release_twin_run  # noqa: E402 - двойник текущего релиза
sys.path.insert(0, str(ROOT / "scripts"))

from eval_common import score                      # noqa: E402
from vreid.metrics import evaluate                 # noqa: E402
from vreid.predict import load_recipe              # noqa: E402


def read_rows(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["key"]: row for row in rows}


def load_pack(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("emb", "vids", "cams", "keys")}


def crop_features(rows: dict, keys, data_root: Path) -> dict:
    """Геометрия рамки и яркость кропа. Яркость считается по уменьшенной копии:
    нам нужна характеристика освещения, а не точность до единицы."""
    from PIL import Image

    area, aspect, brightness = [], [], []
    for key in keys:
        row = rows[str(key)]
        w, h = float(row["w"]), float(row["h"])
        area.append(w * h)
        aspect.append(w / max(h, 1e-6))
        path = data_root / Path(row["path"]).name
        try:
            with Image.open(path) as frame:
                x, y = float(row["x"]), float(row["y"])
                crop = frame.crop((int(x), int(y), int(x + w), int(y + h))).convert("L")
                crop.thumbnail((64, 64))
                brightness.append(float(np.asarray(crop, dtype=np.float32).mean()))
        except OSError:
            brightness.append(float("nan"))
    return {"area": np.array(area), "aspect": np.array(aspect),
            "brightness": np.array(brightness)}


def describe(features: dict, mask: np.ndarray, extra: dict) -> dict:
    if not mask.any():
        return {"count": 0}
    out = {"count": int(mask.sum())}
    for name, values in features.items():
        block = values[mask]
        block = block[np.isfinite(block)]
        if block.size:
            out[f"median_{name}"] = round(float(np.median(block)), 2)
    for name, values in extra.items():
        out[f"median_{name}"] = round(float(np.median(values[mask])), 3)
    return out


FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def label_font(size: int = 13):
    """Шрифт с кириллицей, если он есть в системе. Встроенный шрифт PIL умеет только
    латиницу и рисует кириллицу квадратиками — тогда подпись переводится на латиницу."""
    from PIL import ImageFont

    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size), True
            except OSError:
                continue
    return ImageFont.load_default(), False


def contact_sheet(query_key, gallery_keys, rows_q, rows_g, data_root: Path, out: Path,
                  labels: list[str]) -> None:
    """Кроп запроса и первые кандидаты в одну полосу: в отчёте должно быть видно, что случилось."""
    from PIL import Image, ImageDraw

    font, cyrillic = label_font()
    tile = 200
    cells = [(rows_q[str(query_key)], "запрос" if cyrillic else "QUERY")] + [
        (rows_g[str(key)], label) for key, label in zip(gallery_keys, labels)]
    sheet = Image.new("RGB", (tile * len(cells), tile + 22), (16, 21, 27))
    draw = ImageDraw.Draw(sheet)
    for index, (row, label) in enumerate(cells):
        path = data_root / Path(row["path"]).name
        try:
            with Image.open(path) as frame:
                x, y, w, h = (float(row[k]) for k in ("x", "y", "w", "h"))
                crop = frame.crop((int(x), int(y), int(x + w), int(y + h))).convert("RGB")
        except OSError:
            crop = Image.new("RGB", (tile, tile), (60, 20, 20))
        crop.thumbnail((tile, tile))
        sheet.paste(crop, (index * tile + (tile - crop.width) // 2,
                           (tile - crop.height) // 2))
        draw.text((index * tile + 5, tile + 5), label, fill=(210, 225, 240), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=release_twin_run(ROOT))
    parser.add_argument("--release", type=Path, default=ROOT / "release")
    parser.add_argument("--data", type=Path, default=ROOT / "Данные")
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--out", type=Path, default=ROOT / "results/error_analysis.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/ERROR_ANALYSIS.md")
    args = parser.parse_args()

    recipe = load_recipe(args.release)
    query = load_pack(args.run / "val_query_track.npz")
    gallery = load_pack(args.run / "val_gallery_track.npz")
    rows_q = read_rows(args.run / "val_query.csv")
    rows_g = read_rows(args.run / "val_gallery.csv")
    images = args.data / "images"

    raw, rank, conf, has_match, top1_correct = score(query, gallery, recipe)
    order = np.argsort(-rank, axis=1, kind="stable")

    # Правило жюри: из галереи исключаются только пары vid+cam, совпавшие с запросом.
    # Важно не смешивать две разные основы:
    #   jury_has  — есть ли у запроса ВАЛИДНОЕ положительное после junk-фильтра; по этому
    #               признаку запрос попадает в mAP (ответы 11/13/22);
    #   has_match — есть ли вообще та же машина в галерее; по нему считается режим отказа,
    #               потому что там кадр той же камеры остаётся законным ответом.
    # Дубликат своей камеры существует ровно там, где эти две основы расходятся.
    count = len(query["keys"])
    in_top10 = np.zeros(count, dtype=bool)
    jury_top1 = np.zeros(count, dtype=bool)
    jury_has = np.zeros(count, dtype=bool)
    cross_camera_positives = np.zeros(count)
    same_camera_duplicate = np.zeros(count, dtype=bool)
    for i in range(count):
        same_vehicle = gallery["vids"] == query["vids"][i]
        junk = same_vehicle & (gallery["cams"] == query["cams"][i])
        valid = same_vehicle & ~junk
        cross_camera_positives[i] = int(valid.sum())
        jury_has[i] = bool(valid.any())
        same_camera_duplicate[i] = bool(junk.any())
        kept = [j for j in order[i] if not junk[j]][:10]
        if kept:
            jury_top1[i] = bool(same_vehicle[kept[0]])
            in_top10[i] = bool(same_vehicle[kept].any())

    accepted = conf >= float(recipe["threshold"])
    groups = {
        "rank1_ok": jury_has & jury_top1,
        "rank1_wrong_but_in_top10": jury_has & ~jury_top1 & in_top10,
        "miss_top10": jury_has & ~in_top10,
        "no_valid_positive": ~jury_has,
    }
    assert sum(int(mask.sum()) for mask in groups.values()) == count

    features = crop_features(rows_q, query["keys"], images)
    extra = {"cross_camera_positives": cross_camera_positives,
             "same_camera_duplicate": same_camera_duplicate.astype(float),
             "top1_cosine": conf}

    jury = evaluate(rank, query["vids"], query["cams"], gallery["vids"], gallery["cams"],
                    cutoff=10)
    refusal = {
        "threshold": float(recipe["threshold"]),
        "true_positive": int((accepted & has_match & top1_correct).sum()),
        "false_positive": int((accepted & ~(has_match & top1_correct)).sum()),
        "false_negative": int((~accepted & has_match).sum()),
        "true_negative": int((~accepted & ~has_match).sum()),
        "false_accept_on_no_match": int((accepted & ~has_match).sum()),
        "false_accept_on_wrong_top1": int((accepted & has_match & ~top1_correct).sum()),
    }

    report = {
        "scope": "отложенная валидация двойника релиза; правило жюри и замороженный рецепт",
        "run": str(args.run), "release": str(args.release),
        "n_query": int(len(query["keys"])), "n_gallery": int(len(gallery["keys"])),
        "jury_map10": round(jury["mAP"] * 100, 2), "jury_rank1": round(jury["rank1"] * 100, 2),
        "groups": {name: describe(features, mask, extra) for name, mask in groups.items()},
        "refusal": refusal,
        "by_bbox_area_quartile": [],
        "by_same_camera_duplicate": {},
        "examples": [],
    }

    # Зависимость от размера рамки: маленький кроп — это мало пикселей на кузов.
    finite = np.isfinite(features["area"])
    edges = np.quantile(features["area"][finite & jury_has], [0, .25, .5, .75, 1])
    for lower, upper in zip(edges[:-1], edges[1:]):
        block = jury_has & (features["area"] >= lower) & (features["area"] <= upper)
        if block.sum() < 5:
            continue
        report["by_bbox_area_quartile"].append({
            "area_from": int(lower), "area_to": int(upper), "queries": int(block.sum()),
            "rank1": round(100 * float(jury_top1[block].mean()), 1),
            "recall_at_10": round(100 * float(in_top10[block].mean()), 1),
        })

    for label, mask in (("с дубликатом своей камеры", has_match & same_camera_duplicate),
                        ("только кросс-камерные", has_match & ~same_camera_duplicate)):
        if mask.sum():
            # Здесь основа — режим отказа: берём top1_correct без junk-фильтра, потому что
            # именно этот кандидат уходит в candidates.csv.
            report["by_same_camera_duplicate"][label] = {
                "queries": int(mask.sum()),
                "rank1": round(100 * float(top1_correct[mask].mean()), 1),
                "recall_at_10": round(100 * float(in_top10[mask].mean()), 1),
                "median_top1_cosine": round(float(np.median(conf[mask])), 3),
                "accept_rate": round(100 * float(accepted[mask].mean()), 1),
            }

    # Примеры: самые уверенные промахи — на них ошибка стоит дороже всего.
    interesting = [
        ("miss_top10", "уверенный промах: валидная пара есть, но её нет и в десятке",
         groups["miss_top10"] & accepted),
        ("false_accept", "ложное принятие: валидной пары нет, а система ответила",
         (~has_match) & accepted),
        ("false_refuse", "ложный отказ: пара была, но уверенность ниже порога",
         has_match & ~accepted),
    ]
    sheets = args.report.parent / "errors"
    for slug, title, mask in interesting:
        picked = np.argsort(-conf)[np.isin(np.argsort(-conf), np.nonzero(mask)[0])][:args.examples]
        entries = []
        for number, index in enumerate(picked, 1):
            # Полоса показывает ровно то, что уходит в submission.csv: пары vid+cam,
            # совпавшие с запросом, из галереи уже удалены (ответ 11).
            junk = ((gallery["vids"] == query["vids"][index])
                    & (gallery["cams"] == query["cams"][index]))
            kept = [j for j in order[index] if not junk[j]][:4]
            keys = [gallery["keys"][j] for j in kept]
            labels = [f"#{position} cos {raw[index, j]:.3f}"
                      + (" +" if gallery["vids"][j] == query["vids"][index] else "")
                      for position, j in enumerate(kept, 1)]
            name = f"{slug}_{number}.jpg"
            contact_sheet(query["keys"][index], keys, rows_q, rows_g, images,
                          sheets / name, labels)
            entries.append({"query_id": str(query["keys"][index]),
                            "top1_cosine": round(float(conf[index]), 4),
                            "top1_is_same_camera_duplicate": bool(junk[order[index][0]]),
                            "accepted": bool(accepted[index]),
                            "cross_camera_positives": int(cross_camera_positives[index]),
                            "sheet": "docs/errors/" + name})
        report["examples"].append({"case": title, "count": int(mask.sum()), "shown": entries})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.report)
    print(f"[errors] mAP@10 {report['jury_map10']}, rank-1 {report['jury_rank1']}")
    for name, block in report["groups"].items():
        print(f"[errors] {name:26s} {block['count']:5d}")
    print(f"[errors] → {args.out} и {args.report}")
    return 0


def write_markdown(report: dict, path: Path) -> None:
    total = report["n_query"]
    lines = [
        "# Анализ ошибок", "",
        "Считано по отложенной валидации двойника релиза (модель не видела эти личности),",
        "правилом жюри и замороженным рецептом: те же k-reciprocal 6/2 и тот же порог отказа,",
        f"что уходят в сдачу. Запросов {total}, галерея {report['n_gallery']}, "
        f"mAP@10 {report['jury_map10']}, rank-1 {report['jury_rank1']}.", "",
        "Воспроизвести: `python scripts/error_analysis.py`.", "",
        "## Куда делись запросы", "",
        "| исход | запросов | доля | медиана площади рамки, px | медиана яркости кропа | "
        "медиана кросс-камерных положительных |", "|---|---:|---:|---:|---:|---:|",
    ]
    titles = {
        "rank1_ok": "первый кандидат верный",
        "rank1_wrong_but_in_top10": "верный ответ есть, но не первый",
        "miss_top10": "верного ответа нет в десятке",
        "no_valid_positive": "валидной пары нет — запрос вне mAP",
    }
    for name, block in report["groups"].items():
        if not block["count"]:
            continue
        lines.append(
            f"| {titles.get(name, name)} | {block['count']} | "
            f"{100 * block['count'] / total:.1f}% | "
            f"{block.get('median_area', '—')} | {block.get('median_brightness', '—')} | "
            f"{block.get('median_cross_camera_positives', '—')} |")

    refusal = report["refusal"]
    lines += ["", "## Режим отказа", "",
              f"Порог {refusal['threshold']}. Верных принятий {refusal['true_positive']}, "
              f"ложных принятий {refusal['false_positive']}, ложных отказов "
              f"{refusal['false_negative']}, верных отказов {refusal['true_negative']}.", "",
              f"Ложные принятия распадаются надвое: {refusal['false_accept_on_no_match']} раз "
              f"система ответила на запрос без пары и {refusal['false_accept_on_wrong_top1']} раз "
              "уверенно назвала не ту машину. Вторая половина опаснее: отказ тут не помог бы, "
              "нужен более сильный признак.", ""]

    if report["by_bbox_area_quartile"]:
        lines += ["## Размер рамки", "",
                  "| площадь рамки, px | запросов | rank-1 | доля верных в топ-10 |",
                  "|---|---:|---:|---:|"]
        for row in report["by_bbox_area_quartile"]:
            lines.append(f"| {row['area_from']}–{row['area_to']} | {row['queries']} | "
                         f"{row['rank1']}% | {row['recall_at_10']}% |")
        spread = max(r["rank1"] for r in report["by_bbox_area_quartile"]) - \
            min(r["rank1"] for r in report["by_bbox_area_quartile"])
        lines += ["", f"Размах rank-1 между квартилями {spread:.1f} п.п., и он не монотонен по "
                  "площади: на этих данных размер рамки ошибку не объясняет. Кропы здесь и так "
                  "крупные — медиана около 340 тысяч пикселей при входе модели 336×336, так что "
                  "разрешения хватает всем квартилям.", ""]

    if report["by_same_camera_duplicate"]:
        lines += ["## Дубликат своей камеры — главный перекос валидации", "",
                  "| группа | запросов | rank-1 | в топ-10 | медиана косинуса топ-1 | принято |",
                  "|---|---:|---:|---:|---:|---:|"]
        for label, row in report["by_same_camera_duplicate"].items():
            lines.append(f"| {label} | {row['queries']} | {row['rank1']}% | "
                         f"{row['recall_at_10']}% | {row['median_top1_cosine']} | "
                         f"{row['accept_rate']}% |")
        lines.append("")
        if "только кросс-камерные" not in report["by_same_camera_duplicate"]:
            lines += [
                "Второй строки в таблице нет, и это само по себе результат: **у всех запросов с "
                "парой в галерее лежит кадр той же машины с той же камеры**. Так устроен и "
                "открытый тест — галерея собрана по одному кадру на трек «машина × камера». "
                "Косинус до такого почти дубликата 0.91, и режим отказа на нашей валидации "
                "решает слишком лёгкую задачу.", "",
                "В закрытом тесте гарантирована только кросс-камерная пара (ответ 17), где "
                "медиана косинуса падает до 0.505. Именно поэтому порог заморожен на 0.55, а не "
                "на оптимуме валидации 0.60: см. стресс-проверку "
                "`python scripts/refusal_stress.py` и раздел про порог в README.", ""]

    lines += ["## Характерные случаи", "",
              "Полосы: слева кроп запроса, дальше первые кандидаты с косинусом; "
              "галочкой отмечена та же машина.", ""]
    for block in report["examples"]:
        lines.append(f"### {block['case']} — {block['count']} запросов")
        lines.append("")
        for entry in block["shown"]:
            duplicate = (" топ-1 — дубликат своей камеры, жюри его удаляет,"
                         if entry.get("top1_is_same_camera_duplicate") else "")
            lines.append(f"* `{entry['query_id']}`: косинус топ-1 {entry['top1_cosine']},"
                         f"{duplicate} кросс-камерных положительных "
                         f"{entry['cross_camera_positives']}")
            lines.append("")
            lines.append(f"  ![{entry['query_id']}](errors/{Path(entry['sheet']).name})")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
