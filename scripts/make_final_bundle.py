# -*- coding: utf-8 -*-
"""Собрать итоговую папку на отправку: ядро, релиз, доказательства, письмо.

    python scripts/make_final_bundle.py --out send/vreid_2026-09-19 --zip

Чего НЕ кладём: обучающие кеши (runs/), промежуточные чекпойнты (weights/), сырые данные
организаторов. Всё, что кладём, перечислено явно — молчаливый rglob однажды уже превратил
пакет в 3.1 ГБ.

Каждый файл в MANIFEST-е получает sha256, размер и время. Отсутствующий файл не роняет
сборку, а печатается в конце списком: лучше увидеть дырку сейчас, чем у жюри.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (откуда, куда внутри пакета). Папки копируются целиком с фильтром по расширениям.
TREES = [
    ("vreid", "vreid", {".py"}),
    ("scripts", "scripts", {".py"}),
    ("tests", "tests", {".py"}),
    ("configs", "configs", {".yaml", ".yml"}),
    ("handoff/splits", "splits", {".csv", ".json", ".txt"}),
]
FILES = [
    ("Dockerfile", "Dockerfile"),
    ("docker-compose.yml", "docker-compose.yml"),
    ("requirements.txt", "requirements.txt"),
    ("requirements-infer.txt", "requirements-infer.txt"),
    ("pytest.ini", "pytest.ini"),
    ("README.md", "docs/README-dev.md"),
    ("PLAN.md", "docs/PLAN.md"),
    ("API.md", "docs/API.md"),
    ("handoff/00_ГЛАВНОЕ.md", "00_ГЛАВНОЕ.md"),
    ("handoff/README.md", "README.md"),
    ("handoff/COVER.md", "COVER.md"),
    ("handoff/MANIFEST.json", "MANIFEST.json"),
    ("release_soup/model.pt", "release/model.pt"),
    ("release_soup/recipe.json", "release/recipe.json"),
    ("results/threshold_transfer.json", "reports/threshold_transfer.json"),
    ("results/gallery_size_effect.json", "reports/gallery_size_effect.json"),
    ("results/refusal_stress.json", "reports/refusal_stress.json"),
    ("runs/hack/ft_soup_b336_val/refusal_audit.json", "reports/refusal_audit.json"),
    ("release_soup_fast/recipe.json", "reports/recipe_fast_decode_candidate.json"),
    ("results/bench_full.json", "reports/bench_full.json"),
    ("results/hack_ft_soup_b336_val_val.json", "reports/val_soup.json"),
    ("submission_v2/submission.csv", "example_output/submission.csv"),
    ("submission_v2/candidates.csv", "example_output/candidates.csv"),
    ("submission_v2/embeddings.npy", "example_output/embeddings.npy"),
    ("submission_v2/run_info.json", "example_output/run_info.json"),
]
# Логи шагов очереди: имя в logs/ → имя в reports/
LOGS = ["tests10", "ties", "stress", "predict10", "form10", "indep10", "indep10b",
        "cmp_v2", "build10", "docker10a", "docker10b", "cmp10_det", "docker10_noshm",
        "cmp10_noshm", "form10_docker", "scorecard10", "manifest10",
        "tests", "val_fast", "export_fast", "export_soup", "transfer", "predict_soup", "independence", "independence2",
        "check_form", "check_form_docker", "check_form_noshm", "pin_base", "build_release", "build_dev",
        "bench_native", "bench_container",
        "docker_run_a", "docker_run_b", "cmp_det", "docker_noshm", "cmp_noshm", "cmp_native",
        "abl_report", "gallery_size", "refusal_audit", "manifest", "scorecard", "package"]
# Артефакты контрольного прогона напарника (его дерево)
CONTROL = [
    ("competition/results/selection_baseline", "control/selection_baseline"),
    ("competition/release_calibrated_v1/recipe.json", "control/recipe.json"),
    ("competition/results/selection_baseline/report.json", "control/report.json"),
    ("1809/falcon_qa_2026-09-16.md", "control/falcon_qa_2026-09-16.md"),
    ("1809/IMPLEMENTATION_STATUS.md", "control/IMPLEMENTATION_STATUS.md"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path, missing: list) -> bool:
    if not src.exists():
        missing.append(str(src.relative_to(ROOT)) if ROOT in src.parents else str(src))
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="папка пакета; по умолчанию send/vreid_<дата>")
    ap.add_argument("--zip", action="store_true", help="дополнительно сложить в .zip рядом")
    ap.add_argument("--max-mb", type=float, default=600.0)
    a = ap.parse_args()

    out = Path(a.out) if a.out else ROOT / "send" / f"vreid_{time.strftime('%Y-%m-%d')}"
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        print(f"[bundle] {out} уже есть — дописываю поверх")
    out.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    n = 0
    for src_dir, dst_dir, exts in TREES:
        s = ROOT / src_dir
        if not s.is_dir():
            missing.append(src_dir); continue
        for f in sorted(s.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in exts or "__pycache__" in f.parts:
                continue
            n += copy_file(f, out / dst_dir / f.relative_to(s), missing)
    for src, dst in FILES:
        n += copy_file(ROOT / src, out / dst, missing)
    for name in LOGS:
        n += copy_file(ROOT / "logs" / f"{name}.log", out / "reports" / f"{name}.log", missing)
    for src, dst in CONTROL:
        s = ROOT / src
        if s.is_dir():
            for f in sorted(s.rglob("*")):
                if f.is_file() and f.suffix.lower() in {".json", ".csv", ".md", ".log", ".txt"}:
                    n += copy_file(f, out / dst / f.relative_to(s), missing)
        else:
            n += copy_file(s, out / dst, missing)

    manifest = {}
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.name != "BUNDLE.sha256.json":
            st = f.stat()
            manifest[f.relative_to(out).as_posix()] = {
                "sha256": sha256(f), "bytes": st.st_size,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))}
    (out / "BUNDLE.sha256.json").write_text(
        json.dumps({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "files": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    total_mb = sum(v["bytes"] for v in manifest.values()) / 1024 ** 2
    print(f"[bundle] {len(manifest)} файлов, {total_mb:.0f} МБ → {out}")
    if missing:
        print(f"[bundle] НЕ НАШЁЛ {len(missing)} шт. (проверь, что очередь отработала):")
        for m in missing:
            print(f"   - {m}")
    if total_mb > a.max_mb:
        print(f"[bundle] ВНИМАНИЕ: {total_mb:.0f} МБ > лимита {a.max_mb:.0f} — что-то лишнее заехало")

    if a.zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        zp = Path(z)
        print(f"[bundle] zip {zp.stat().st_size / 1024 ** 2:.0f} МБ, sha256 {sha256(zp)}")
        print(f"[bundle] → {zp}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
