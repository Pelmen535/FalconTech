from env import *
from learning import Residual
from collections import defaultdict
import joblib
z=np.load(N/'night_soft-transform.npz');domain=joblib.load(N/'domain.joblib');oldnight=joblib.load(N/'night-head.joblib');heads={}
def pack_side(side,kind='base',alpha=None):
 assert side in ['cal','val'];data=np.load(W/'backend-cal-val'/(side+'-features.npz'));p=bc.pack(records[side],data['emb']);p.update({k:data[k] for k in data.files if k!='emb'});p.setdefault('parts',np.zeros((len(p['keys']),1),np.float16));p.setdefault('cnn',np.zeros((len(p['keys']),1),np.float16));return p

def episodes(p,side):
 out=[];vid=p['vids'];org=bc.sub(p,vid<1000000);night=bc.sub(p,(vid>=1000000)&(vid<2000000));dn=bc.sub(p,vid>=2000000)
 for name,rows in [('original',org),('night',night)]:
  if not len(rows['vids']):continue
  ids=rows['vids'][::4]
  if name=='night':
   original=json.loads((B/'records.json').read_text());ids=np.array(original['night_folds']['calibration' if side=='cal' else 'validation']);assert set(ids)==set(rows['vids'])
  assert len(ids)==len(set(ids))
  for keep in [.8,.4]:out.append((name+('-natural' if keep==.8 else '-removed'),*bc.episode(rows,ids,keep)))
 if len(dn['vids']):
  ids=sorted(set(dn['vids']));by=defaultdict(lambda:defaultdict(list))
  for i,(v,c) in enumerate(zip(dn['vids'],dn['cams'])):by[int(v)][int(c)].append(i)
  for qc,gc,direction in [(0,1,'day-night'),(1,0,'night-day')]:
   for keep in [.8,.4]:
    known=ids[:int(len(ids)*.8)] if keep==.8 else ids[int(len(ids)*.4):int(len(ids)*.8)];qi=[by[v][qc][0] for v in ids];gi=[i for v in known for i in by[v][gc][:3]];out.append(('dn-'+direction+('-natural' if keep==.8 else '-removed'),bc.sub(dn,qi),bc.sub(dn,gi)))
 return out

def head_features(x,base,path):
 if path not in heads:
  ck=torch.load(W/path,map_location='cpu',weights_only=True);head=Residual(ck.get('dim',2048));head.load_state_dict(ck['head']);head.eval();heads[path]=head
 with torch.inference_mode():return heads[path](torch.tensor(np.asarray(x,dtype=np.float32)),torch.tensor(base.copy())).numpy()

def route(q,g,cfg):
 import evaluate as single
 alpha=cfg['fusion'];r0=single.route(q,g,{});r1=single.route(q,g,{'cnn':1.,'cnn_head':'specialist-cnn.pt'})
 if alpha==0:return r0
 if alpha==1:return r1
 raw=(1-alpha)*r0['raw']+alpha*r1['raw'];rank=(1-alpha)*r0['rank']+alpha*r1['rank'];r={**r1,'raw':raw,'rank':rank};order=np.argsort(-rank,axis=1,kind='stable');top=order[:,0];r['top']=top;conf=raw[np.arange(len(raw)),top];top5=np.sort(raw,axis=1)[:,-5:];other=raw.copy();other[np.arange(len(raw)),top]=-np.inf
 gg=(1-alpha)*(r0['gallery']@r0['gallery'].T)+alpha*(r1['gallery']@r1['gallery'].T);np.fill_diagonal(gg,-np.inf);dens=np.partition(gg,-5,axis=1)[:,-5:].mean(1)[top];r['X']=np.column_stack([conf,conf-np.max(other,1),top5[:,-1]-top5[:,-2],top5.mean(1),top5.std(1),rank[np.arange(len(raw)),top]-rank[np.arange(len(raw)),order[:,1]],dens,conf-dens]).astype(np.float64)
 oq=bc.transform(q,bc.OLDZ['transform'],bc.OLDZ['mean']);og=bc.transform(g,bc.OLDZ['transform'],bc.OLDZ['mean']);aux=br.rank_infer(oq,og);sim=aux['raw'][np.arange(len(top)),top];best=np.max(aux['raw'],1);gg=og['emb']@og['emb'].T;np.fill_diagonal(gg,-np.inf);density=np.partition(gg,-5,axis=1)[:,-5:].mean(1)[top];r['X20']=np.column_stack([r['X'],aux['X'],top==aux['top'],sim,sim-best,sim-density]);return r

def score(m):return .45*m['mAP']+.1*m['Q']
def summarize(metrics):
 avg=lambda prefix:float(np.mean([score(v) for k,v in metrics.items() if k.startswith(prefix)])) if any(k.startswith(prefix) for k in metrics) else 0.;return avg('original')+.1*avg('night')+.1*avg('dn')
def assess(p,side,cfg,tune=False):
 eps=episodes(p,side);outputs=[route(q,g,cfg) for _,q,g in eps];recipe=cfg.get('recipe',choice['recipe']).copy()
 if tune:
  truth=[bc.truth(q,g,r) for (_,q,g),r in zip(eps,outputs)]
  # Each domain threshold calibrated on original and NiVe episodes only. DN is diagnostic/training, not threshold tuning.
  from recipe import tune as tune_threshold
  for field,head,selected in [('source_threshold',bc.BASEH,[0,1]),('night_threshold',oldnight,[2,3])]:recipe[field]=tune_threshold([bc.odds(head,outputs[i]['X20']) for i in selected],[truth[i] for i in selected])
 cfg={**cfg,'recipe':recipe};metrics={label:bc.metrics(q,g,r,confidence(r,recipe),bc.BASE['threshold']) for (label,q,g),r in zip(eps,outputs)};return cfg,metrics
