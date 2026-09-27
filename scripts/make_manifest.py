"""Манифест поставки: то, по чему сдачу можно проверить, не веря нам на слово.

    python scripts/make_manifest.py --release release --submission submission \
        --run runs/hack/ft_soup_r392dw40f4t2_fit --out reports/MANIFEST.json

Складывает в один файл: хэши весов, рецепта и файлов сдачи; состав разбиения по vehicle_id;
версии пакетов и команды, которыми всё получено; ключевые измерения со ссылками на отчёты.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_of(path: Path) -> list[int]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return sorted({int(r["vid"]) for r in csv.DictReader(f) if r.get("vid") not in (None, "", "-1")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="reports/MANIFEST.json")
    a = ap.parse_args()

    rel, sub, run = Path(a.release), Path(a.submission), Path(a.run)
    m = {"собрано": __import__("datetime").datetime.now().isoformat(timespec="seconds")}

    m["хэши"] = {}
    for p in list(rel.glob("*")) + list(sub.glob("*")):
        if p.is_file():
            m["хэши"][f"{p.parent.name}/{p.name}"] = {"sha256": sha256(p), "байт": p.stat().st_size}

    m["рецепт"] = json.loads((rel / "recipe.json").read_text(encoding="utf-8"))

    m["разбиение"] = {
        "источник": str(run),
        "fit_ids": ids_of(run / "fit.csv"),
        "val_query_ids": ids_of(run / "val_query.csv"),
        "val_gallery_ids": ids_of(run / "val_gallery.csv"),
    }
    m["разбиение"]["пересечение_fit_и_val"] = sorted(
        set(m["разбиение"]["fit_ids"]) & set(m["разбиение"]["val_query_ids"]))

    m["окружение"] = {"python": sys.version.split()[0], "платформа": platform.platform()}
    for pkg in ("torch", "torchvision", "timm", "numpy", "PIL"):
        try:
            mod = __import__(pkg)
            m["окружение"][pkg] = getattr(mod, "__version__", "?")
        except Exception as e:
            m["окружение"][pkg] = f"нет ({type(e).__name__})"
    try:
        m["окружение"]["nvidia-smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        m["окружение"]["nvidia-smi"] = "недоступно"

    m["отчёты"] = {}
    for name, p in (("валидация", Path("results") / f"hack_{run.name}_val.json"),
                    ("производительность", Path("results/bench_full.json")),
                    ("аудит отказа", run / "refusal_audit.json"),
                    ("прогон", sub / "run_info.json")):
        if p.exists():
            m["отчёты"][name] = str(p)

    m["команды"] = {
        "учитель": "python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_l "
                   "--img-size 336 --epochs 20 --P 5 --K 4 --freeze-blocks 8 --cam-aware "
                   "--cross-cam-triplet --out weights/hack_dinov2_l_336_cam",
        "второй учитель (релиз v1.3)": "python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_b "
                                       "--img-size 392 --epochs 18 --P 8 --K 4 --cam-aware --cross-cam-triplet "
                                       "--distill weights/hack_dinov2_l_336_cam/best.pt --distill-w 40 "
                                       "--distill-focus 4 --val-frac 0 --seed <0|2|3> "
                                       "--out weights/all18_r392_dw40f4_s<seed>; затем суп трёх сидов "
                                       "в weights/soup_r392dw40f4_all/best.pt",
        "обучение": "python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_b "
                    "--img-size 392 --epochs 18 --P 8 --K 4 --cam-aware --cross-cam-triplet "
                    "--distill weights/hack_dinov2_l_336_cam/best.pt,weights/soup_r392dw40f4_all/best.pt "
                    "--distill-w 40 --distill-focus 4 --val-frac 0 --seed <0|2|3> "
                    "--out weights/all18_r392_dw40f4t2_s<seed>",
        "усреднение весов": "python scripts/model_soup.py weights/all18_r392_dw40f4t2_s0/best.pt "
                            "weights/all18_r392_dw40f4t2_s2/best.pt weights/all18_r392_dw40f4t2_s3/best.pt "
                            "--out weights/soup_r392dw40f4t2_all/best.pt",
        "порог отказа": "python scripts/choose_threshold.py --stress results/refusal_stress_r392dw40f4t2.json "
                        "--out results/threshold_choice_r392dw40f4t2.json",
        "сборка релиза": f"python scripts/export_release.py --weights weights/soup_r392dw40f4t2_all/best.pt "
                         f"--val results/hack_ft_soup_r392dw40f4t2_fit_val.json --confidence top1 "
                         f"--threshold {m['рецепт'].get('threshold')} "
                         f"--threshold-choice results/threshold_choice_r392dw40f4t2.json --kr "
                         f"--k1 {m['рецепт'].get('k1')} --k2 {m['рецепт'].get('k2')} "
                         f"--lam {m['рецепт'].get('lam')} --crop-pad 0.05 --out {rel}",
        "инференс": "docker run --rm --gpus all --shm-size=2g -v <данные>:/data:ro -v <выход>:/out vreid:1.0",
        "независимость запросов": f"python scripts/check_query_independence.py --submission {sub} --data <данные>",
        "аудит отказа": f"python scripts/refusal_audit.py --run {run} --recipe {rel}/recipe.json",
        "производительность": "docker run --rm --gpus all --shm-size=2g -v <данные>:/data:ro "
                              "-v <результаты>:/app/results --entrypoint python vreid-dev "
                              "-m vreid.bench_full --data /data --release /app/release --n 600 --workers 8 --fast-decode",
    }

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"[manifest] файлов с хэшами {len(m['хэши'])}, "
          f"fit id {len(m['разбиение']['fit_ids'])}, val id {len(m['разбиение']['val_query_ids'])}, "
          f"пересечение {len(m['разбиение']['пересечение_fit_и_val'])}")
    print(f"[manifest] → {out}")


if __name__ == "__main__":
    main()
