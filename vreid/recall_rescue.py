"""Explicit experimental refusal supplement; original accepted answers are immutable."""
from collections import OrderedDict
from pathlib import Path
import hashlib,json,os,time
import joblib
import numpy as np
from scipy.special import expit,logit
from .refusal_policy_vric_road import VricRoadRefusalPolicy as OriginalPolicy
from .raw_refusal import raw_features

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class RescueState:
 def __init__(self,spec,head,pcas):
  if spec.get('status') not in ['FROZEN_FOR_VALIDATION','READY_LOCAL_WINDOWS_PENDING_LINUX']:raise ValueError('Unqualified rescue selection')
  if getattr(head,'n_features_in_',None) not in [479,480] or list(head.classes_)!=[0,1,2]:raise ValueError('Expected native conditional 479/480D three-class head')
  self.spec=spec;self.head=head;self.pcas=pcas;self.cache=OrderedDict();self.owner=None;self.gallery=None
  self.input_precision=spec.get('input_precision','float32-all')
  if self.input_precision not in ['float32-all','float32-native-confidence64','float64-native']:raise ValueError('Unknown trained input precision')
  c=spec['calibration'];cut=spec['threshold']
  if not (0<cut<1 and np.isfinite([c['T'],c['b0'],c['b1']]).all() and c['T']>0):raise ValueError('Invalid calibration or cut')
  if len(pcas)!=3 or any(p.components_.shape!=(64,d) for p,d in zip(pcas,[1024,1024,2048])):raise ValueError('Invalid native projections')
 def project(self,pack):
  emb=np.asarray(pack['emb']);cnn=np.asarray(pack['specialist'])
  if emb.ndim!=2 or emb.shape[1]!=2048 or cnn.shape!=emb.shape or not np.isfinite(emb).all() or not np.isfinite(cnn).all():raise ValueError('Invalid native feature pack')
  channels=[]
  for x in [emb[:,:1024],emb[:,1024:],cnn]:
   x=np.asarray(x,np.float64);channels.append(x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-10))
  out=np.empty((len(emb),192),np.float32)
  for i,(a,b) in enumerate(zip(emb,cnn)):
   h=hashlib.sha256()
   for x in [a,b]:h.update(x.dtype.str.encode());h.update(x.tobytes())
   token=h.digest()
   if token in self.cache:out[i]=self.cache[token];self.cache.move_to_end(token);continue
   values=[]
   for x,p in zip(channels,self.pcas):
    y=(x[i]-p.mean_.astype(np.float64))@p.components_.astype(np.float64).T;y/=max(np.linalg.norm(y),1e-10);values.append(y)
   out[i]=np.concatenate(values);self.cache[token]=out[i].copy()
  while len(self.cache)>4096:self.cache.popitem(last=False)
  return channels,out
 def predict(self,result,old,boundary):
  if not 0<boundary<1:raise ValueError('Invalid boundary')
  accepted=np.asarray(old['accepted'],bool).copy();confidence=np.asarray(old['confidence']).copy();indices=np.flatnonzero(~accepted);score=np.full(len(accepted),np.nan)
  if not len(indices):return {**old,'recall_rescue_score':score,'recall_rescue_added':np.zeros(len(accepted),bool)}
  q=result['_query_features'];g=result['_gallery_features'];owner=result.get('_recall_rescue_gallery_snapshot')
  if owner is not None and owner is self.owner:gs,gd=self.gallery
  else:
   gs,gd=self.project(g)
   if owner is not None:self.owner=owner;self.gallery=(gs,gd)
  qs,qd=self.project({k:q[k][indices] for k in ['emb','specialist']});top=result['top'][indices];b=gd[top];cos=np.column_stack([np.einsum('ij,ij->i',a,x[top]) for a,x in zip(qs,gs)]).astype(np.float32);pair=np.c_[(qd-b)**2,qd*b,cos];x=np.c_[raw_features(result)[indices][:,[i for i in range(93) if i!=28]],pair]
  if self.input_precision!='float64-native':x=x.astype(np.float32)
  if self.head.n_features_in_==480:x=np.c_[x,old['confidence'][indices]]
  if self.input_precision=='float32-all':x=x.astype(np.float32)
  raw=np.log(np.clip(self.head.predict_proba(x),1e-12,1));c=self.spec['calibration'];value=expit((raw[:,2]-raw[:,0])/c['T']-c['b0']);score[indices]=value;cut=self.spec['threshold'];add=np.zeros(len(accepted),bool);add[indices]=value>=cut;confidence[add]=np.maximum(boundary,expit(logit(np.clip(score[add],1e-12,1-1e-12))-logit(cut)+logit(boundary)));accepted|=add
  assert np.array_equal(accepted,confidence>=boundary) and np.array_equal(confidence[old['accepted']],old['confidence'][old['accepted']])
  return {**old,'accepted':accepted,'confidence':confidence,'recall_rescue_score':score,'recall_rescue_added':add}

_STATES={}
def load_state(root):
 root=Path(root).resolve();path=Path(os.environ.get('VREID_RECALL_RESCUE_SELECTION',''))
 if not path.is_absolute():raise ValueError('Explicit absolute rescue selection required')
 spec=json.loads(path.read_text(encoding='utf-8'));key=(str(root),digest(path))
 if digest(root/'var/recall-road-fit-vric-20261018/runtime-selection.json')!=spec['base_selection_SHA256']:raise ValueError('Base selection changed')
 for name,want in spec['files'].items():
  if digest(name)!=want:raise ValueError('Rescue asset checksum mismatch: '+name)
 if key not in _STATES:_STATES[key]=RescueState(spec,joblib.load(spec['head']),joblib.load(spec['projections']))
 return _STATES[key]

def install_runtime(root):
 state=load_state(root)
 import context
 from cached_post import install
 install()
 from joint_runtime import JointFeatures
 if not getattr(JointFeatures.extract,'_recall_rescue',False):
  original=JointFeatures.extract
  def extract(self,records):
   features,timing=original(self,records);start=time.perf_counter();state.project(features);extra=time.perf_counter()-start;timing=dict(timing);timing['seconds']=float(timing['seconds'])+extra;timing['FPS']=len(records)/timing['seconds'];timing['recall_rescue_descriptor_seconds']=extra;return features,timing
  extract._recall_rescue=True;JointFeatures.extract=extract
 from . import current_query_profile
 if not getattr(current_query_profile.route,'_recall_rescue',False):
  original_route=current_query_profile.route
  def route(q,g,cfg):
   result=original_route(q,g,cfg);result['_recall_rescue_gallery_snapshot']=g;return result
  route._recall_rescue=True;current_query_profile.route=route
 return state

class RecallRescuePolicy:
 def __init__(self,base,state):
  if state.spec.get('shared_primary_features'):
   from .shared_refusal import SharedVricRoadPolicy
   base=SharedVricRoadPolicy(base)
  self.baseline=base;self.state=state
 @classmethod
 def load(cls,root):return cls(OriginalPolicy.load(root),install_runtime(root))
 def predict(self,result,legacy_confidence,boundary):return self.state.predict(result,self.baseline.predict(result,legacy_confidence,boundary),boundary)
