from context import *
from cached_post import install
cache_receipt=install()
from joint_runtime import JointFeatures
from runtime_features import Features as FullFeatures
from runtime_fast_dual import FeaturesFast
from fusion_evaluate import route
from d3_runtime import write_outputs
import argparse
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path);p.add_argument('--query',type=Path);p.add_argument('--gallery',type=Path);p.add_argument('--images',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--batch-size',type=int,choices=[1,8,16,32],default=16);a=p.parse_args();assert not a.out.exists();sel=json.loads((W/'fusion-selection.json').read_text());assert all(digest(W/n)==h for n,h in sel['files'].items());assert json.loads((J/'batch-decision.json').read_text())['status']=='PASS_DISTINCT_BATCH_QUALITY'
assert json.loads((J/'transfer-results.json').read_text())['_decision']['status']=='PROCEED_SPEED'
if a.manifest:
 if any([a.query,a.gallery,a.images]):p.error('Supply manifest OR query/gallery/images')
 m=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
else:
 if not all([a.query,a.gallery,a.images]):p.error('Supply manifest OR all query/gallery/images')
 from vreid.hackathon_data import read_annotations
 m={side:[{'key':r.key,'path':str(Path(r.path).resolve()),'bbox':list(r.bbox)} for r in read_annotations(path,a.images,bbox_format='xywh',strict=True)] for side,path in [('query',a.query),('gallery',a.gallery)]}
# Runtime consumes images, bbox and keys only, discard identity/camera metadata even if supplied.
m={side:[{k:r[k] for k in ['key','path','bbox'] if k in r} for r in m[side]] for side in ['query','gallery']}
for side in m:
 rows=m[side];keys=[r['key'] for r in rows];assert rows and len(set(keys))==len(keys) and all(isinstance(k,str) and k.strip()==k and k for k in keys)
 for r in rows:
  assert Path(r['path']).is_file();box=r.get('bbox');assert box is None or (len(box)==4 and np.isfinite(box).all() and box[2]>0 and box[3]>0)
logs=a.out.with_name(a.out.name+'-logs');logs.mkdir();rt=JointFeatures(logs,a.batch_size)
try:q,qt=rt.extract(m['query']);g,gt=rt.extract(m['gallery'])
finally:rt.close()
r=route(q,g,sel['config']);r['probability']=confidence(r,sel['config']['recipe']);r['accepted']=r['probability']>=bc.BASE['threshold'];write_outputs(a.out,r,q['keys'].tolist(),g['keys'].tolist(),bc.BASE['threshold']);(a.out/'run_info.json').write_text(json.dumps({'status':'COMPLETE','constant_cache':cache_receipt,'cached_post_SHA':digest(M/'cached_post.py'),'selection_SHA':digest(W/'fusion-selection.json'),'engine_SHA':digest(B/'export/model.engine'),'CNN_SHA':digest(W/'specialist-cnn.pt'),'CNN_engine_SHA':digest(E/'model.engine'),'runtime_SHA':digest(J/'joint_runtime.py'),'decoder':'padded352','batch':a.batch_size,'query':qt,'gallery':gt,'numeric_equivalence':'not claimed; distinct-batch CAL/VAL admission','A5000_verified':False},indent=2),encoding='utf-8');print('COMPLETE',a.out,flush=True)
