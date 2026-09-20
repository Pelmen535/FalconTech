"""Усреднение весов нескольких прогонов одного рецепта (model soup, Wortsman et al., 2022).

    python scripts/model_soup.py --out weights/soup_b336/best.pt \
        weights/hack_dinov2_b_336_distill20/best.pt weights/hack_dinov2_b_336_seed2/best.pt

Стоимость инференса остаётся стоимостью одной модели с той же архитектурой.
Улучшение качества не гарантировано: сравнить soup с лучшим участником на одном
identity-disjoint validation. Совпадение имён и форм тензоров не доказывает
совместимое происхождение обучения; оно проверяется по журналам прогонов.

Усредняются только вещественные тензоры; целочисленные буферы (счётчики BatchNorm и подобное)
берутся из первого чекпойнта.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", help="два и более weights/<run>/best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="+", default=["backbone", "bnneck"],
                    help="какие части чекпойнта усреднять")
    a = ap.parse_args()

    import torch

    if len(a.checkpoints) < 2:
        raise SystemExit("нужно минимум два чекпойнта")
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in a.checkpoints]
    base = cks[0]
    import hashlib
    from datetime import datetime, timezone
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for ck in cks[1:]:
        for k in ('model_name','img_size','mean','std'):
            if ck.get(k)!=base.get(k):raise ValueError('Incompatible model metadata: '+k)
    sources=[dict(path=str(p),sha256=digest(p)) for p in a.checkpoints]
    for part in a.parts:
        if part not in base:
            print(f"[soup] в чекпойнте нет части «{part}» — пропускаю")
            continue
        keys = set(base[part])
        for c in cks[1:]:
            if set(c[part]) != keys:
                raise SystemExit(f"наборы ключей в «{part}» не совпадают — это разные архитектуры")
        merged, n_avg, n_copy = {}, 0, 0
        for k in base[part]:
            t = base[part][k]
            for ck in cks:
                u=ck[part][k]
                if u.shape!=t.shape or u.dtype!=t.dtype:raise ValueError('Incompatible tensor: '+part+'.'+k)
                if u.is_floating_point() and not torch.isfinite(u).all():raise ValueError('Nonfinite tensor: '+k)
            if t.is_floating_point():
                acc = t.float().clone()
                for c in cks[1:]:
                    acc += c[part][k].float()
                merged[k] = (acc / len(cks)).to(t.dtype)
                n_avg += 1
            else:
                merged[k] = t
                n_copy += 1
        base[part] = merged
        print(f"[soup] «{part}»: усреднено {n_avg} тензоров, скопировано без изменений {n_copy}")
    out = Path(a.out)
    # Если передана папка (путь без .pt), кладём внутрь best.pt — именно так их ищет
    # get_backbone("ft:weights/<run>/best.pt"). Раньше создавался файл без расширения,
    # и следующий шаг падал с FileNotFoundError уже ПОСЛЕ того, как суп посчитан.
    if out.suffix.lower() not in ('.pt', '.pth'):
        out = out / 'best.pt'
    out.parent.mkdir(parents=True, exist_ok=True)
    base['soup_sources']=sources
    base['soup_buffer_policy']='Floating buffers averaged; integer buffers copied from first member. Validate, do not assume improvement.'
    base['created_at']=datetime.now(timezone.utc).isoformat()
    torch.save(base, out)
    mb = out.stat().st_size / 1024 ** 2
    print(f"[soup] {len(cks)} моделей -> {out} ({mb:.0f} МБ)")
    for p in a.checkpoints:
        print(f"        {p}")


if __name__ == "__main__":
    main()
