"""Публикация релиза на GitHub: веса в Release, рецепт и манифест весов — коммитом.

Зачем отдельный скрипт. Ночная очередь собирает и проверяет релиз сама, и публиковать его
тоже должна сама: ждать утра ради одной команды незачем, а сессия, из которой очередь
запущена, может к тому моменту закончиться. Но публикация необратима, поэтому скрипт
проверяет две вещи и без них ничего не выкладывает:

  1. что новый двойник ЛУЧШЕ уже опубликованного по метрике жюри (mAP@10 с той
     постобработкой, которая стоит в рецепте) — иначе это не «лучшая модель», а просто
     другая, и заменять ей рабочий релиз нельзя;
  2. что весов в релизе ровно столько, сколько требует рецепт: model.pt всегда,
     reranker.pt — только если в рецепте включён каскад.

Рецепт и манифест весов коммитятся ОДНИМ коммитом с тегом: так recipe.json и веса,
которыми сняты числа, не могут разойтись — scripts/fetch_weights.py читает манифест из того
же коммита.

    python scripts/publish_release.py --tag v1.1 --val results/hack_ft_soup_r392_fit_val.json \\
        --title "v1.1 — вход 392"
    python scripts/publish_release.py ... --dry-run     # всё проверить и показать, не публикуя
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
MANIFEST = RELEASE / "weights.json"
GH = r"C:\Program Files\GitHub CLI\gh.exe"
# Автор коммитов в публичном репозитории — решение владельца: noreply-почта и без трейлера
# соавторства (история переписана под это 22.09, возвращать нельзя).
GIT_IDENTITY = ["-c", "user.name=Связанные одной цепью",
                "-c", "user.email=Pelmen535@users.noreply.github.com"]
DESCRIPTIONS = {
    "model.pt": "DINOv2 ViT-B/14, основная модель: её вектор идёт в embeddings.npy, "
                "её путь меряет жюри",
    "reranker.pt": "DINOv2 ViT-L/14, ре-ранкер второй ступени каскада",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def twin_score(val_path: Path, recipe: dict) -> tuple[float, float, str]:
    """mAP@10 и rank-1 двойника ровно при той постобработке, что стоит в рецепте.

    Приоритет - эталонному scorer организаторов (results/official_validation/comparison.json),
    если он считал ИМЕННО этого двойника. Val-отчёт считает k-reciprocal по всей матрице, а
    боевой путь - пачками с блокировкой кадров; на двойнике 392 это 77.60 против 77.41. Балл
    жюри считается по боевому пути, поэтому публиковать надо его число, а не промежуточное:
    v1.1 сначала вышел с 77.60, и прирост над v1.0 был завышен на 0.19."""
    official = ROOT / "results" / "official_validation" / "comparison.json"
    run_name = val_path.name.replace("hack_", "", 1).replace("_val.json", "")
    if official.is_file():
        data = json.loads(official.read_text(encoding="utf-8"))
        if Path(str(data.get("run", ""))).name == run_name:
            ranking = data["official"]["ranking"]
            return 100 * ranking["mAP@10"], 100 * ranking["Rank-1"], "эталонный scorer"
        print(f"[publish] эталонный scorer считал {Path(str(data.get('run'))).name}, а не "
              f"{run_name} - беру val-отчёт; число может разойтись с боевым путём на десятые")
    data = json.loads(val_path.read_text(encoding="utf-8"))
    rows = data["postprocess"]["rows"]
    if recipe.get("kr"):
        name = f"k-recip {recipe['k1']}/{recipe['k2']}"
        row = next((r for r in rows if r["name"] == name), None)
        if row is None:
            raise SystemExit(f"[publish] в {val_path} нет строки «{name}» — сравнивать не с чем")
    else:
        name, row = "base", data["jury"]
    return 100 * row["mAP"], 100 * row["rank1"], name


def run(cmd: list[str], dry: bool) -> str:
    print("[publish] $ " + " ".join(cmd))
    if dry:
        return ""
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if done.returncode != 0:
        print(done.stdout, done.stderr)
        raise SystemExit(f"[publish] команда упала с кодом {done.returncode}")
    return done.stdout


def notes(tag: str, recipe: dict, twin: tuple, prev: dict | None, files: dict) -> str:
    bench = {}
    bench_path = ROOT / "results" / "bench_full.json"
    if bench_path.is_file():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
    m, r1, pp = twin
    lines = [
        f"Веса для запуска решения, релиз {tag}. В репозитории их нет намеренно: бесплатной квоты "
        f"Git LFS хватило бы на одно клонирование.",
        "",
        "## Как получить",
        "",
        "```bash",
        "python scripts/fetch_weights.py",
        "```",
        "",
        "Скрипт берёт список файлов и их SHA-256 из `release/weights.json` того же коммита, что "
        "и `recipe.json`, кладёт веса в `release/` и сверяет суммы.",
        "",
        "## Файлы",
        "",
        "| файл | размер | SHA-256 | что это |",
        "|---|---:|---|---|",
    ]
    for name, spec in files.items():
        lines.append(f"| `{name}` | {spec['size'] / 2**20:.0f} МиБ | `{spec['sha256']}` | {spec['what']} |")
    lines += ["", "## Точность", "",
              "Метрика жюри mAP@10 на отложенной валидации — модель-двойник, не видевшая эти "
              "личности, суп трёх сидов, без отбора эпохи по валидации.", "",
              "| | mAP@10 | rank-1 |", "|---|---:|---:|"]
    if prev and prev.get("twin_map10"):
        lines.append(f"| предыдущий релиз {prev.get('tag')} | {prev['twin_map10']:.2f} | "
                     f"{prev.get('twin_rank1', float('nan')):.2f} |")
    lines.append(f"| **{tag}** ({pp}) | **{m:.2f}** | **{r1:.2f}** |")
    if bench:
        lines += ["", "## Производительность", "",
                  "По методике организаторов: латентность — медиана 300 прогонов batch=1 после 50 "
                  "прогревов, пропускная — устойчивый прогон не короче 10 секунд.", "",
                  "| | значение | балл |", "|---|---:|---:|",
                  f"| латентность batch=1 | {bench['latency_ms_median']:.1f} мс | "
                  f"{10 * bench['latency_score']:.1f} / 10 |",
                  f"| лучший FPS (batch {bench.get('best_batch')}) | {bench['best_fps']:.1f} | "
                  f"{10 * bench['throughput_score']:.1f} / 10 |",
                  f"| веса | {bench.get('weights_mb', 0):.0f} МБ из 2000 | — |",
                  f"| два прогона подряд | {'бит в бит' if bench.get('determinism_bitwise') else 'РАСХОДЯТСЯ'} | — |"]
    lines += ["", "## Рецепт", "",
              f"порог отказа {recipe['threshold']}, k-reciprocal {recipe.get('k1')}/{recipe.get('k2')}/"
              f"{recipe.get('lam')}, каскад {'включён' if recipe.get('cascade') else 'выключен'}. "
              f"Полностью — `release/recipe.json` этого же коммита."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--val", type=Path, required=True,
                        help="val-отчёт ДВОЙНИКА этого релиза (results/hack_ft_<суп>_fit_val.json)")
    parser.add_argument("--title", default=None,
                        help="по умолчанию строится из рецепта: вход и включён ли каскад. Строится "
                             "здесь, а не передаётся из очереди: PowerShell 5.1 без BOM портит "
                             "кириллицу в аргументах командной строки")
    parser.add_argument("--img-size", type=int, default=None, help="вход модели, для названия")
    parser.add_argument("--distill-w", type=float, default=None,
                        help="вес дистилляции, для названия: v1.2 отличается от v1.1 только им")
    parser.add_argument("--distill-focus", type=float, default=None,
                        help="--distill-focus обучения, для названия: v1.3 отличается от v1.2 только им")
    parser.add_argument("--repo", default="Pelmen535/FalconTech")
    parser.add_argument("--min-gain", type=float, default=0.3,
                        help="на сколько пунктов mAP@10 двойник обязан обогнать опубликованный")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recipe = json.loads((RELEASE / "recipe.json").read_text(encoding="utf-8"))
    if args.title is None:
        parts = [f"вход {args.img_size}"] if args.img_size else []
        if args.distill_w:
            parts.append(f"дистилляция ×{args.distill_w:g}")
        if args.distill_focus:
            parts.append(f"фокус на трудных парах ×{1 + args.distill_focus:g}")
        parts.append("с каскадом" if recipe.get("cascade") else "без каскада")
        args.title = ", ".join(parts)
    prev = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else None
    twin = twin_score(ROOT / args.val if not args.val.is_absolute() else args.val, recipe)
    print(f"[publish] двойник {args.tag}: mAP@10 {twin[0]:.2f} ({twin[2]}), rank-1 {twin[1]:.2f}")

    # 1. Лучше ли опубликованного. Первый релиз, где число не записано, сравнивать не с чем.
    if prev and prev.get("twin_map10") is not None:
        gain = twin[0] - float(prev["twin_map10"])
        print(f"[publish] опубликован {prev['tag']}: {prev['twin_map10']:.2f} → прирост {gain:+.2f}")
        if gain < args.min_gain:
            print(f"[publish] прирост меньше {args.min_gain} — это не лучшая модель, "
                  f"опубликованный релиз не трогаю")
            return 2
    if prev and prev.get("tag") == args.tag:
        raise SystemExit(f"[publish] тег {args.tag} уже опубликован")

    # 2. Ровно те веса, которые требует рецепт
    needed = ["model.pt"] + (["reranker.pt"] if recipe.get("cascade") else [])
    files = {}
    for name in needed:
        path = RELEASE / name
        if not path.is_file():
            raise SystemExit(f"[publish] рецепт требует {name}, а его нет в release/")
        files[name] = {"sha256": sha256(path), "size": path.stat().st_size,
                       "what": DESCRIPTIONS.get(name, "")}
        print(f"[publish] {name}: {files[name]['size'] / 2**20:.0f} МиБ, sha256 {files[name]['sha256'][:16]}…")

    # twin_run пишется явно: по нему все отчёты (vreid.artifacts.release_twin_run) находят
    # двойника текущего релиза. Без него он выводится из имени val-отчёта - работает, но
    # держится на соглашении об именах.
    twin_run = "runs/hack/" + args.val.name.replace("hack_", "", 1).replace("_val.json", "")
    manifest = {"tag": args.tag, "repo": args.repo, "files": files, "twin_run": twin_run,
                "twin_map10": round(twin[0], 3), "twin_rank1": round(twin[1], 3),
                "twin_postprocess": twin[2], "twin_val": str(args.val).replace("\\", "/")}
    body = notes(args.tag, recipe, twin, prev, files)
    notes_path = ROOT / "logs" / f"release_notes_{args.tag}.md"

    if args.dry_run:
        print("[publish] --dry-run: ничего не пишу и не публикую. Заметки к релизу:\n")
        print(body)
        return 0

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes_path.write_text(body, encoding="utf-8")
    run(["git", "add", "-A"], False)
    message = (f"Релиз {args.tag}: {args.title}\n\n"
               f"Двойник релиза: mAP@10 {twin[0]:.2f} ({twin[2]}), rank-1 {twin[1]:.2f}"
               + (f", предыдущий {prev['tag']} — {prev['twin_map10']:.2f}" if prev and prev.get("twin_map10") else "")
               + ".\nВеса лежат файлами GitHub Release, их SHA-256 — в release/weights.json этого "
                 "коммита.\nОпубликовано скриптом scripts/publish_release.py после прохождения "
                 "всех проверок очереди.")
    run(["git", *GIT_IDENTITY, "commit", "-q", "-m", message], False)
    run(["git", "push", "-q", "origin", "HEAD"], False)
    run([GH, "release", "create", args.tag, *[str(RELEASE / n) for n in files],
         "--repo", args.repo, "--title", f"{args.tag} — {args.title}",
         "--notes-file", str(notes_path)], False)
    print(f"[publish] опубликовано: https://github.com/{args.repo}/releases/tag/{args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
