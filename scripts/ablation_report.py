"""Ablation report. Aggregates are descriptive; paired rows are needed for uncertainty.
No automatic claim that license-plate dependence is absent, and no worst-control selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(results: Path, model: str, tag: str | None):
    name = f"hack_{model}_val" + (f"_mask-{tag}" if tag else "") + ".json"
    p = results / name
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ft_hack_dinov2_b_336_distill20")
    ap.add_argument("--results", default="results")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="во сколько пунктов опора на номер считается существенной")
    ap.add_argument('--paired-json', help='Aligned per-query AP, identities and predeclared control groups')
    ap.add_argument('--out', default='results/ablation_paired.json')
    a = ap.parse_args()
    if a.paired_json:
        import sys, hashlib
        import numpy as np
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from vreid.paired import paired_drop
        source=Path(a.paired_json);data=json.loads(source.read_text('utf-8-sig'))
        for k in ('model_sha256','recipe_sha256','split_sha256','mask_design_sha256'):
            if not data.get(k):raise ValueError('Missing evidence field: '+k)
        if not data.get('predeclared_controls'):raise ValueError('Declare control groups before comparing outcomes')
        if len(set(data['query_ids']))!=len(data['query_ids']):raise ValueError('Duplicate query IDs')
        base=data['baseline_ap10'];ids=data['vehicle_ids']
        if len(base)!=len(data['query_ids']):raise ValueError('Row alignment mismatch')
        report={k:data[k] for k in ('model_sha256','recipe_sha256','split_sha256','mask_design_sha256')}
        report['groups']=[]
        for group in data['control_groups']:
            name=group['mask'];mask=data['variants'][name]
            row={'mask':name,'effect':paired_drop(base,mask['ap10'],ids),'controls':{}}
            if not group['controls']:raise ValueError('Control group is empty')
            for control in group['controls']:
                c=data['variants'][control]
                if not np.allclose(mask['area_fraction'],c['area_fraction'],rtol=0,atol=1e-6):
                    raise ValueError('Per-query mask areas differ from the declared matched control')
                row['controls'][control]=paired_drop(base,c['ap10'],ids)
                row['controls'][control]['plate_minus_control']=paired_drop(c['ap10'],mask['ap10'],ids)
            report['groups'].append(row)
        report['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        report['conclusion']='Sensitivity and paired uncertainty only; no assertion of license-plate independence.'
        out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
        return 0

    results = Path(a.results)
    runs = {tag or "base": _load(results, a.model, tag)
            for tag in (None, "det", "up", "rand", "band")}
    base = runs["base"]
    if base is None:
        raise SystemExit(f"нет базового прогона для {a.model} в {results}")

    # Метрика жюри — это удаление только пар vid+cam. Ключ "cross_camera" в старых файлах
    # хранит строгий режим, который завышает результат; берём "jury", если он есть.
    key = "jury" if "jury" in base else "standard"
    b = base[key]["mAP"] * 100
    print(f"модель {a.model}, запросов {base['n_query']}, галерея {base['n_gallery']}")
    print(f"{'вариант':<34}{'mAP@10':>8}{'rank1':>8}{'Δ к base':>10}")
    print("-" * 60)
    drops = {}
    for tag, d in runs.items():
        if d is None:
            print(f"{tag:<34}{'—':>8}   нет прогона")
            continue
        k = "jury" if "jury" in d else "standard"
        m = d[k]["mAP"] * 100
        r1 = d[k]["rank1"] * 100
        drops[tag] = b - m
        delta = "" if tag == "base" else f"{m - b:+.1f}"
        note = {"det": "  закрыт номер", "up": "  контроль: выше по кузову",
                "rand": "  контроль: случайное место", "band": "  старый грубый прямоугольник"}.get(tag, "")
        print(f"{tag:<34}{m:8.1f}{r1:8.1f}{delta:>10}{note}")

    print('Aggregate values alone do not establish license-plate independence.')
    print('Supply --paired-json with predeclared matched-area controls and a large solid-mask condition.')
    print('No training decision is made automatically from the worst/best control.')
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
