"""Optional matched metric/refusal supplement; never revokes an original acceptance."""
from pathlib import Path
from collections import OrderedDict
import os,json,time
import joblib
import numpy as np
from scipy.special import expit,logit
from .recall_rescue import digest
from .native_tail import LinearMetric,TailGainFeatures
from .refusal_policy_vric_road import VricRoadRefusalPolicy as OriginalPolicy
from .raw_refusal import raw_features

class MatchedState:
 def __init__(self,spec,head,pcas,weight):
  if spec.get('status') not in ['FROZEN_FOR_VALIDATION','READY_LOCAL_WINDOWS_PENDING_LINUX']:raise ValueError('Unqualified matched rescue')
  if getattr(head,'n_features_in_',None) not in [479,736] or list(head.classes_)!=[0,1,2]:raise ValueError('Expected native479/metric736D joint head')
  c=spec['calibration']
  if not (0<spec['threshold']<1 and c['T']>0 and np.isfinite([c['T'],c['b0'],c['b1']]).all()):raise ValueError('Invalid calibrated rule')
  if len(pcas)!=3 or any(p.components_.shape!=(64,d) for p,d in zip(pcas,[1024,1024,2048])):raise ValueError('Invalid native projections')
  self.spec=spec;self.head=head;self.pcas=pcas;self.metric=LinearMetric(weight);self.cache=OrderedDict();self.owner=None;self.gallery=None
 def project(self,pack):
  # Only content determines descriptors. IDs, camera labels and gallery composition do not.
  channels=TailGainFeatures.channels(pack);native=np.empty((len(channels[0]),192),np.float32);metric=np.empty((len(channels[0]),128),np.float64);tokens=[];missing=[]
  import hashlib
  for i,(a,b) in enumerate(zip(pack['emb'],pack['specialist'])):
   h=hashlib.sha256()
   for x in [a,b]:x=np.asarray(x);h.update(x.dtype.str.encode());h.update(x.tobytes())
   token=h.digest();tokens.append(token)
   if token in self.cache:native[i],metric[i]=self.cache[token];self.cache.move_to_end(token)
   else:missing.append(i)
  if missing:
   values=[]
   for x,p in zip(channels,self.pcas):
    v=(x[missing]-p.mean_.astype(np.float64))@p.components_.astype(np.float64).T;v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-10);values.append(v)
   native[missing]=np.concatenate(values,1).astype(np.float32);metric[missing]=self.metric.project({k:np.asarray(pack[k])[missing] for k in ['emb','specialist']})
   for i in missing:self.cache[tokens[i]]=(native[i].copy(),metric[i].copy())
  while len(self.cache)>4096:self.cache.popitem(last=False)
  return channels,native,metric
 def predict(self,result,old,boundary):
  accepted=np.asarray(old['accepted'],bool).copy();confidence=np.asarray(old['confidence']).copy();indices=np.flatnonzero(~accepted);added=np.zeros(len(accepted),bool);score=np.full(len(accepted),np.nan)
  if not len(indices):return {**old,'matched_rescue_score':score,'matched_rescue_added':added}
  q={k:result['_query_features'][k][indices] for k in ['emb','specialist']};g=result['_gallery_features'];owner=result.get('_matched_rescue_gallery_snapshot')
  if owner is not None and owner is self.owner:gs,gd,gm=self.gallery
  else:
   gs,gd,gm=self.project(g)
   if owner is not None:self.owner=owner;self.gallery=(gs,gd,gm)
  qs,qd,qm=self.project(q);base=result['rank'][indices];mask=~np.isfinite(result['raw'][indices]);sim=qm@gm.T;sim[mask]=-np.inf;rank=.65/(20+self.metric.ordinal(base))+.35/(20+self.metric.ordinal(sim));rank[mask]=-np.inf;top=np.argsort(-rank,axis=1,kind='stable')[:,0]
  proposed={**result,'rank':rank,'top':top,'raw':result['raw'][indices],'query':result['query'][indices],'query_night':result['query_night'][indices],'_query_features':q};raw=raw_features(proposed)[:,[i for i in range(93) if i!=28]];b=gd[top];cos=np.column_stack([np.einsum('ij,ij->i',a,x[top]) for a,x in zip(qs,gs)]).astype(np.float32);pair=np.c_[(qd-b)**2,qd*b,cos];x=np.c_[raw,pair]
  if self.head.n_features_in_==736:
   b=gm[top];extra=np.c_[(qm-b)**2,qm*b,np.einsum('ij,ij->i',qm,b)].astype(np.float32);x=np.c_[x,extra]
  p=np.log(np.clip(self.head.predict_proba(x),1e-12,1));c=self.spec['calibration'];value=expit((p[:,2]-p[:,0])/c['T']-c['b0']);score[indices]=value;cut=self.spec['threshold'];added[indices]=value>=cut
  confidence[added]=np.maximum(boundary,expit(logit(np.clip(score[added],1e-12,1-1e-12))-logit(cut)+logit(boundary)));accepted|=added
  # The original result arrays are replaced only for newly accepted refusals.
  changed_rank=result['rank'].copy();changed_top=result['top'].copy();chosen=value>=cut;changed_rank[indices[chosen]]=rank[chosen];changed_top[indices[chosen]]=top[chosen];result['rank']=changed_rank;result['top']=changed_top
  assert np.array_equal(accepted,confidence>=boundary) and np.array_equal(confidence[old['accepted']],old['confidence'][old['accepted']])
  return {**old,'accepted':accepted,'confidence':confidence,'matched_rescue_score':score,'matched_rescue_added':added}

_STATES={}
def load_state(root):
 root=Path(root).resolve();path=Path(os.environ.get('VREID_MATCHED_RESCUE_SELECTION',''))
 if not path.is_absolute():raise ValueError('Explicit absolute matched rescue selection required')
 s=json.loads(path.read_text(encoding='utf-8'));key=(str(root),digest(path))
 if digest(root/'var/recall-road-fit-vric-20261018/runtime-selection.json')!=s['base_selection_SHA256']:raise ValueError('Base selection changed')
 for f,want in s['files'].items():
  if digest(f)!=want:raise ValueError('Matched rescue asset changed: '+f)
 if key not in _STATES:_STATES[key]=MatchedState(s,joblib.load(s['head']),joblib.load(s['projections']),np.load(s['metric']))
 return _STATES[key]

def install_runtime(root):
 state=load_state(root)
 import context
 from cached_post import install
 install()
 from joint_runtime import JointFeatures
 existing=getattr(JointFeatures.extract,'_matched_rescue_state',None)
 if existing is not None and existing is not state:raise ValueError('Restart worker to switch matched rescue selection')
 if existing is None:
  original=JointFeatures.extract
  def extract(self,records):
   features,timing=original(self,records);start=time.perf_counter();state.project(features);extra=time.perf_counter()-start;timing=dict(timing);timing['seconds']=float(timing['seconds'])+extra;timing['FPS']=len(records)/timing['seconds'];timing['matched_rescue_descriptor_seconds']=extra;return features,timing
  extract._matched_rescue_state=state;JointFeatures.extract=extract
 from . import current_query_profile
 if not getattr(current_query_profile.route,'_matched_rescue',False):
  original_route=current_query_profile.route
  def route(q,g,cfg):
   r=original_route(q,g,cfg);r['_matched_rescue_gallery_snapshot']=g;return r
  route._matched_rescue=True;current_query_profile.route=route
 return state

class MatchedRescuePolicy:
 def __init__(self,base,state):
  if state.spec.get('shared_primary_features'):
   from .shared_refusal import SharedVricRoadPolicy
   base=SharedVricRoadPolicy(base)
  self.baseline=base;self.state=state
 @classmethod
 def load(cls,root):return cls(OriginalPolicy.load(root),install_runtime(root))
 def predict(self,result,legacy_confidence,boundary):
  # Frozen original input allows repeat calls on the same result without accumulated reranking.
  if '_matched_rescue_original' not in result:result['_matched_rescue_original']=dict(result)
  original=result['_matched_rescue_original'];work=dict(original);old=self.baseline.predict(original,legacy_confidence,boundary);out=self.state.predict(work,old,boundary);result['rank']=work['rank'];result['top']=work['top'];return out
