"""Замер производительности ровно по методике организаторов (ответы Q-31/Q-32/Q-34 от 16.09).

Меряется ПОЛНЫЙ цикл на одно ТС: чтение файла с диска, декодирование, кроп по bbox,
препроцессинг, forward, постобработка, L2-нормализация. Поиск по галерее не входит.

  latency   — при batch=1, медиана по выборке;
  throughput — лучший FPS среди батчей 1 / 8 / 16 / 32.

Балл (20% итоговой оценки) = 10% × latency_score + 10% × throughput_score:
  latency  ≤ 40 мс → 1.0, ≥ 80 мс → 0.0, между — линейно;
  throughput ≥ 100 FPS → 1.0, ≤ 50 FPS → 0.0, между — линейно.

    python -m vreid.bench_full --data Данные --release release --n 200 --workers 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .extract import crop_bbox, open_crop
from .hackathon_data import read_annotations
from .models import get_backbone


def latency_score(ms: float) -> float:
    return float(np.clip((80.0 - ms) / 40.0, 0.0, 1.0))


def throughput_score(fps: float) -> float:
    return float(np.clip((fps - 50.0) / 50.0, 0.0, 1.0))


class _Full:
    """Один элемент = полный цикл до входа в сеть: открыть файл, декодировать, кроп, препроцессинг."""

    def __init__(self, records, backbone, pad, fast_decode=False):
        self.recs, self.bb, self.pad = records, backbone, pad
        self.fast, self.size = fast_decode, int(getattr(backbone, "size", 336))

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        crop = open_crop(r.path, r.bbox, self.pad, self.fast, self.size)
        return self.bb.transform(crop)


def main():
    ap = argparse.ArgumentParser(description="полный цикл на одно ТС: латентность и пропускная способность")
    ap.add_argument("--data", default="Данные")
    ap.add_argument("--release", default="release")
    ap.add_argument("--n", type=int, default=300, help="сколько ТС прогнать (методика: не меньше 300)")
    ap.add_argument("--workers", type=int, default=8, help="воркеров загрузчика для пакетного режима")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32])
    ap.add_argument("--warmup", type=int, default=50, help="методика организаторов: 50 прогревочных")
    ap.add_argument("--min-seconds", type=float, default=10.0,
                    help="методика: устойчивый прогон не короче 10 с на каждый размер батча")
    ap.add_argument("--fast-decode", action="store_true",
                    help="декодировать JPEG сразу в уменьшенном масштабе (PIL draft), если bbox это позволяет")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    data, release = Path(a.data), Path(a.release)
    rec = json.loads((release / "recipe.json").read_text(encoding="utf-8")) if (release / "recipe.json").exists() else {}
    pad = float(rec.get("crop_pad", 0.05))
    t_load = time.perf_counter()
    bb = get_backbone("ft:" + str(release / "model.pt"), device=a.device)
    load_s = time.perf_counter() - t_load
    # Сумма ВСЕХ файлов весов — организаторы проверяют именно её против лимита 2 ГБ
    # и смотрят расширения .pt/.pth/.bin/.onnx/.engine/.plan/.safetensors/.ckpt/.trt/.pb/.tflite/.npz
    # (ответы 34/37). Превышение — недопуск к замеру, а не штраф, поэтому считаем сами.
    W_EXT = {".pt", ".pth", ".bin", ".onnx", ".engine", ".plan", ".safetensors", ".ckpt",
             ".trt", ".pb", ".tflite", ".npz"}
    w_files = [f for f in release.rglob("*") if f.is_file() and f.suffix.lower() in W_EXT]
    w_mb = sum(f.stat().st_size for f in w_files) / 1024 ** 2
    print(f"[bench] веса: {len(w_files)} файл(ов), {w_mb:.0f} МБ из 2048 лимита; "
          f"загрузка весов {load_s:.1f} с")
    recs = read_annotations(data / "test_query.csv", data / "images", None, "xywh")[: a.n]
    ds = _Full(recs, bb, pad, fast_decode=a.fast_decode)
    # явный --device cuda:0 раньше не попадал в сравнение со строкой "cuda" и выключал sync/pin_memory
    cuda = str(getattr(bb, "device", "cpu")).startswith("cuda")
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    print(f"[bench] {len(ds)} ТС, вход {getattr(bb, 'size', '?')}, устройство {getattr(bb, 'device', '?')}, "
          f"TTA flip={rec.get('tta_flip', False)}, быстрый декод={a.fast_decode}")

    # сколько стоит сам ввод-вывод: чтение + декодирование + кроп + препроцессинг, без сети
    t0 = time.perf_counter()
    for i in range(min(50, len(ds))):
        ds[i]
    io_ms = (time.perf_counter() - t0) / min(50, len(ds)) * 1000
    print(f"[bench] из них чтение+декод+кроп+препроцессинг: {io_ms:.1f} мс/ТС (одним потоком)")

    # --- латентность batch=1: строго последовательно, один процесс, как у жюри ---
    for i in range(min(a.warmup, len(ds))):
        bb.embed_tensors(ds[i].unsqueeze(0))
    sync()
    lat = []
    for i in range(len(ds)):
        sync()                                       # методика: CUDA sync ДО и ПОСЛЕ каждого прогона
        t0 = time.perf_counter()
        x = ds[i].unsqueeze(0)                       # чтение + декод + кроп + препроцессинг
        e = bb.embed_tensors(x)                      # forward (+ L2 внутри embed_tensors)
        if rec.get("tta_flip"):
            e = e + bb.embed_tensors(torch.flip(x, dims=[3]))
            e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
        sync()
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    if len(lat) < 300:
        print(f"[bench] ВНИМАНИЕ: прогонов {len(lat)} < 300, методика требует медиану по 300")
    ms = float(np.median(lat))
    print(f"[bench] латентность batch=1: медиана {ms:.1f} мс, среднее {lat.mean():.1f}, "
          f"90% {np.quantile(lat, .9):.1f} → балл {latency_score(ms) * 100:.0f}%")

    # --- пропускная способность: лучший FPS среди батчей ---
    # ВАЖНО: воркеры поднимаются заново на каждый новый загрузчик (на Windows это spawn, секунды).
    # Поэтому persistent_workers=True, первый проход прогревочный и не засекается, второй — замер.
    best_fps, best_b = 0.0, 0
    rows = []
    for b in a.batches:
        loader = DataLoader(ds, batch_size=b, shuffle=False, num_workers=a.workers,
                            pin_memory=cuda, persistent_workers=(a.workers > 0))
        for x in loader:                              # прогрев целиком: поднимает воркеров и кэш ФС
            bb.embed_tensors(x)
        sync()
        # Методика: устойчивый прогон не короче 10 с. Один проход по 300 кропам при 175 FPS
        # длится полторы секунды — это не замер устойчивой скорости, а замер переходного режима,
        # поэтому повторяем проходы, пока не наберётся min_seconds, и делим на ВСЕ обработанные кропы.
        t0 = time.perf_counter()
        done, passes = 0, 0
        while True:
            for x in loader:
                e = bb.embed_tensors(x)
                if rec.get("tta_flip"):
                    bb.embed_tensors(torch.flip(x, dims=[3]))
                done += len(x)
            passes += 1
            sync()
            if time.perf_counter() - t0 >= a.min_seconds:
                break
        dt = time.perf_counter() - t0
        del loader
        fps = done / dt
        rows.append({"batch": b, "fps": round(fps, 1), "ms_per_item": round(dt / done * 1000, 2),
                     "items": done, "seconds": round(dt, 1), "passes": passes})
        print(f"[bench] batch={b:>2}: {fps:6.1f} кроп/с ({dt / done * 1000:5.1f} мс/ТС), "
              f"{done} кропов за {dt:.1f} с в {passes} прохода(х)")
        if fps > best_fps:
            best_fps, best_b = fps, b
    print(f"[bench] лучший FPS {best_fps:.1f} при batch={best_b} → балл {throughput_score(best_fps) * 100:.0f}%")

    # Детерминизм двух прогонов: организаторы фиксируют его отдельно (ответ 32).
    n_det = min(64, len(ds))
    xb = torch.stack([ds[i] for i in range(n_det)])
    e1 = bb.embed_tensors(xb); e2 = bb.embed_tensors(xb)
    det_max = float(np.abs(e1 - e2).max())
    det_bit = bool(np.array_equal(e1, e2))
    det_txt = "бит в бит" if det_bit else "max|Δ|=%.2e" % det_max
    print(f"[bench] детерминизм двух прогонов на {n_det} кропах: {det_txt}")
    vram_mb = None
    if cuda:
        vram_mb = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
        print(f"[bench] пик VRAM {vram_mb:.0f} МБ из 24576 на A5000")

    total = 10 * latency_score(ms) + 10 * throughput_score(best_fps)
    print(f"[bench] ИТОГО за производительность: {total:.1f} из 20 баллов "
          f"(латентность {10 * latency_score(ms):.1f}/10, пропускная {10 * throughput_score(best_fps):.1f}/10)")
    print(f"[bench] воркеров {a.workers}; у жюри 2×Xeon Gold 6338 (64 ядра) — там пакетный режим будет быстрее")

    out = Path(rec.get("results_dir", "results")); out.mkdir(parents=True, exist_ok=True)
    # Хэши модели и рецепта пишутся рядом с числами. Без них замер нельзя приписать конкретному
    # релизу, и scorecard такую строку справедливо не засчитывает: один раз это уже случилось.
    import hashlib
    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    release_dir = Path(a.release)
    with open(out / "bench_full.json", "w", encoding="utf-8") as f:
        json.dump({"n": len(ds), "tta_flip": bool(rec.get("tta_flip")), "workers": a.workers,
                   "model_sha256": digest(release_dir / "model.pt"),
                   "recipe_sha256": digest(release_dir / "recipe.json"),
                   "device": str(getattr(bb, "device", a.device or "cuda")),
                   "warmup": a.warmup, "min_seconds": a.min_seconds,
                   "weights_files": len(w_files), "weights_mb": round(w_mb, 1),
                   "weights_load_seconds": round(load_s, 2), "peak_vram_mb": vram_mb,
                   "determinism_bitwise": det_bit, "determinism_max_abs_diff": det_max,
                   "fast_decode": bool(a.fast_decode), "io_ms_per_item": round(io_ms, 2),
                   "latency_ms_median": ms, "latency_ms_mean": float(lat.mean()),
                   "latency_score": latency_score(ms), "throughput": rows,
                   "best_fps": best_fps, "best_batch": best_b,
                   "throughput_score": throughput_score(best_fps), "score_of_20": total},
                  f, ensure_ascii=False, indent=2)
    print(f"[bench] → {out / 'bench_full.json'}")


if __name__ == "__main__":
    main()
