"""Сколько разрешения можно купить на запас производительности, не выходя из полного балла.

Зачем. Шкала производительности опубликована и однозначна (ответ 34): латентность до 40 мс —
полный балл, пропускная от 100 FPS — полный балл. Релиз сейчас стоит на 28.5 мс и 195.7 FPS,
то есть часть бюджета не потрачена. Тратить её надо внутри ЗАМЕРЯЕМОГО пути: там выигрыш
достаётся без единого допущения о том, как жюри посчитает ре-ранжирование.

Что меряется. Чистый форвард ViT-B на входах 336/364/392/420/448 при batch 1 и batch 8,
той же методикой, что в vreid/bench_full.py: fp16 autocast, CUDA sync до и после, медиана.
Веса не нужны — время форварда от их значений не зависит, только от формы; поэтому модель
создаётся без предобученных весов и прогон занимает минуты, а не часы обучения.

Дальше время подставляется в формулу балла вместе с ИЗМЕРЕННЫМ вводом-выводом текущего
релиза: латентность = ввод-вывод + форвард, пропускная = 1000 / форвард на кроп при batch 8
(ввод-вывод при восьми воркерах прячется за GPU — это видно из того, что добавление второго
форварда роняет пропускную ровно на его стоимость).

    python scripts/resolution_budget.py --out results/resolution_budget.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vreid.bench_full import latency_score, throughput_score   # noqa: E402
from vreid.train import build_model                            # noqa: E402

SIZES = (336, 364, 392, 420, 448)


def forward_ms(model, size: int, batch: int, device: str, repeats: int, warmup: int) -> float:
    import torch
    x = torch.randn(batch, 3, size, size, device=device)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                model(x)
        sync()
        taken = []
        for _ in range(repeats):
            sync()
            t0 = time.perf_counter()
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                model(x)
            sync()
            taken.append((time.perf_counter() - t0) * 1000)
    return float(np.median(np.array(taken)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--bench", type=Path, default=ROOT / "results/bench_full.json",
                        help="откуда взять измеренный ввод-вывод текущего релиза")
    parser.add_argument("--compile", action="store_true",
                        help="прогнать backbone через torch.compile: в Linux-контейнере это "
                             "обычно 20-50% к скорости форварда, а значит и к запасу по баллу")
    parser.add_argument("--out", type=Path, default=ROOT / "results/resolution_budget.json")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench = json.loads(args.bench.read_text(encoding="utf-8")) if args.bench.is_file() else {}
    io_ms = float(bench.get("io_ms_per_item", 24.0))
    base_fwd = float(bench.get("forward_ms_main", 0) or 0)

    rows = []
    for size in args.sizes:
        backbone, _, _ = build_model(args.model, size, pretrained=False)
        backbone = backbone.to(device).eval()
        if args.compile:
            backbone = torch.compile(backbone)
        one = forward_ms(backbone, size, 1, device, args.repeats, args.warmup)
        many = forward_ms(backbone, size, args.batch, device, args.repeats, args.warmup) / args.batch
        latency = io_ms + one
        fps = 1000.0 / many
        rows.append({"size": size, "tokens": (size // 14) ** 2,
                     "forward_ms_b1": round(one, 2),
                     "forward_ms_per_item_b8": round(many, 2),
                     "latency_ms": round(latency, 2),
                     "latency_score": round(latency_score(latency), 4),
                     "fps": round(fps, 1),
                     "throughput_score": round(throughput_score(fps), 4),
                     "score_of_20": round(10 * latency_score(latency) + 10 * throughput_score(fps), 2)})
        print(f"[budget] {size}: форвард {one:5.1f} мс (b1), {many:5.2f} мс/кроп (b{args.batch}) → "
              f"латентность {latency:5.1f} мс, {fps:6.1f} FPS → "
              f"{rows[-1]['score_of_20']:5.2f} / 20", flush=True)
        del backbone
        if device == "cuda":
            torch.cuda.empty_cache()

    full = [r for r in rows if r["score_of_20"] >= 20.0]
    report = {"device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
              "compiled": bool(args.compile),
              "io_ms_per_item": io_ms,
              "release_forward_ms_measured": base_fwd,
              "note": ("время форварда не зависит от значений весов, только от формы входа; "
                       "поэтому модель создана без предобученных весов"),
              "max_size_with_full_score": (max(r["size"] for r in full) if full else None),
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[budget] полный балл держится до входа {report['max_size_with_full_score']} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
