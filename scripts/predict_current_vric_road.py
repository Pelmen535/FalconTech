"""Validate the global correctness verifier with the preserved expanded CLI runtime."""
from pathlib import Path
import argparse,json,runpy,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        print('Usage: python scripts/predict_current_vric_road.py (--manifest MANIFEST | --query CSV --gallery CSV --images DIR) --out NEW_DIR [--batch-size {1,8,16,32}]')
        print('Recall-oriented vric-road-verifier-v1/current-query-v1: one exposed-development-selected global threshold, no gallery-style routing. Full acceptance is recorded separately.')
        raise SystemExit(0)
    # Capture the unmodified baseline inside the pair module before replacing
    # the CLI's process-local class binding. No source files are modified.
    from vreid.refusal_policy_vric_road import VricRoadRefusalPolicy,MANIFEST
    import vreid.refusal_policy_expanded as expanded
    old=expanded.ExpandedRefusalPolicy
    expanded.ExpandedRefusalPolicy=VricRoadRefusalPolicy
    try:runpy.run_path(str(ROOT/'scripts/predict_current_expanded.py'),run_name='__main__')
    finally:expanded.ExpandedRefusalPolicy=old
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--out',type=Path,required=True);args,_=parser.parse_known_args()
    from vreid.refusal_policy_bagged import digest
    path=args.out/'run_info.json';info=json.loads(path.read_text(encoding='utf-8'));info['vric_road_refusal']={'profile':'vric-road-verifier-v1','selection_SHA':digest(ROOT/MANIFEST),'global_threshold':True,'gallery_style_routing':False}
    path.write_text(json.dumps(info,indent=2),encoding='utf-8')
