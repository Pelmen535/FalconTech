"""Current-query ranking with the FIT-trained, CAL-frozen contrast verifier."""
from pathlib import Path
import argparse,hashlib,json,os,sys,runpy,time
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'var/general-refusal-final-20261017'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
if '--help' in sys.argv or '-h' in sys.argv:
    print('Usage: python scripts/predict_current_expanded.py (--manifest MANIFEST | --query CSV --gallery CSV --images DIR) --out NEW_DIR [--batch-size {1,8,16,32}]')
    print('Frozen encoders + current-query-v1 + expanded-source-v1. Local Windows profile; new Linux/full-speed acceptance still required. Outputs: embeddings.npy/submission.csv/candidates.csv.')
    raise SystemExit(0)
decision=json.loads((P/'decision.json').read_text(encoding='utf-8-sig'))
if decision['status'] not in ['Q_DOMINANT_PENDING_NEW52_AND_NATIVE','READY_LOCAL_WINDOWS_Q_GAIN_WITH_TNR_TRADEOFF']:
    raise RuntimeError('Practical Q candidate is not qualified for validation')
if hashlib.sha256((P/'selection.json').read_bytes()).hexdigest()!=decision['selection_SHA']:raise RuntimeError('Frozen selection checksum changed')
selection=json.loads((P/'selection.json').read_text(encoding='utf-8-sig'))
for path,field in [(ROOT/'vreid/raw_refusal.py','feature_code_SHA'),(ROOT/'vreid/refusal_policy_raw.py','policy_code_SHA'),(P/'predict.py','CLI_code_SHA')]:
    if hashlib.sha256(path.read_bytes()).hexdigest()!=selection[field]:raise RuntimeError('Frozen implementation checksum changed: '+field)
profile=json.loads((ROOT/'var/rerank-repair-20261017/runtime-selection.json').read_text(encoding='utf-8'))
for filename,expected in profile['files'].items():
    if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=expected:raise RuntimeError('Rerank profile code changed: '+filename)
sys.path.insert(0,str(ROOT/'var/teacher-projection-20261011'));import resources
resources.restrict_current_process();resources.snapshot();time.sleep(.2);snap=resources.snapshot();why=[r for r in resources.start_reasons(snap,'gpu') if r!='GPU free below9GiB or unknown']
if snap.get('GPU_free_GiB') is None or snap['GPU_free_GiB']<7:why.append('GPU free below7GiB')
if snap['RAM_free_GiB']<4.35:why.append('Insufficient measured native RAM headroom')
m=resources.Mem();m.length=resources.ctypes.sizeof(m);assert resources.ctypes.windll.kernel32.GlobalMemoryStatusEx(resources.ctypes.byref(m))
if m.commit_available/2**30<7:why.append('Insufficient measured native commit headroom')
if why:print('WAITING_RESOURCES',why);raise SystemExit(77)
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'var/cached-feature-post-20261010'))
from vreid.refusal_policy_raw import GeneralRefusalPolicy
from vreid.refusal_policy_expanded import ExpandedRefusalPolicy as LearnedContrastRefusalPolicy
policy=LearnedContrastRefusalPolicy.load(ROOT)
import context
from cached_post import install
install();original=context.confidence
def confidence(result,recipe):
    legacy=original(result,recipe);return policy.predict(result,legacy,context.bc.BASE['threshold'])['confidence']
import fusion_evaluate as fusion
original_route=fusion.route
from vreid.current_query_profile import route as selected_route
def route(q,g,cfg):
    result=selected_route(q,g,cfg)
    result['_query_features']={k:q[k] for k in ['emb','specialist']}
    result['_gallery_features']={k:g[k] for k in ['emb','specialist']}
    result['_gallery_keys']=g['keys']
    return result
fusion.route=route
context.confidence=confidence
original_args=sys.argv[1:];sys.argv=[str(ROOT/'var/cached-feature-post-20261010/predict_cached.py'),*original_args]
try:runpy.run_path(sys.argv[0],run_name='__main__')
finally:
    context.confidence=original
    fusion.route=original_route
parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--out',type=Path,required=True);args,_=parser.parse_known_args(original_args)
info=json.loads((args.out/'run_info.json').read_text(encoding='utf-8'));info['expanded_refusal']={'profile':'expanded-source-v1','selection_SHA':hashlib.sha256((ROOT/'var/systematic-ranking-20261017/expanded-runtime-selection.json').read_bytes()).hexdigest()}
info['general_refusal']={'selection_SHA':decision['selection_SHA'],'model_SHA':json.loads((P/'selection.json').read_text())['winner']['model_SHA'],'ranking_unchanged':False,'rerank_profile':'current-query-v1','policy_file_SHA':hashlib.sha256((ROOT/'vreid/refusal_policy_raw.py').read_bytes()).hexdigest()};(args.out/'run_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
