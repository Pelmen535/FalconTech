"""Conditional local-patch verifier; original accepted candidates stay immutable."""
from collections import OrderedDict
from pathlib import Path
import os,json
import numpy as np
import joblib
from scipy.special import expit,logit
from .recall_rescue import RescueState,RecallRescuePolicy,digest
from .raw_refusal import raw_features
from .refusal_policy_vric_road import VricRoadRefusalPolicy as OriginalPolicy

class LocalState(RescueState):
 def __init__(self,spec,head,pcas):
  if spec.get('status')!='FROZEN_FOR_VALIDATION' or head.n_features_in_!=724 or list(head.classes_)!=[0,1,2]:raise ValueError('Expected frozen three-class native480+local244 model')
  if not 0<spec['threshold']<1 or spec['calibration']['T']<=0:raise ValueError('Invalid local calibration')
  if not np.isfinite([spec['threshold'],*spec['calibration'].values()]).all():raise ValueError('Nonfinite local calibration')
  if len(pcas)!=3 or any(p.components_.shape!=(64,d) for p,d in zip(pcas,[1024,1024,2048])):raise ValueError('Invalid fixed native projections')
  self.spec=spec;self.head=head;self.pcas=pcas;self.cache=OrderedDict();self.owner=None;self.gallery=None
 def predict(self,result,old,boundary):
  accepted=np.asarray(old['accepted'],bool).copy();confidence=np.asarray(old['confidence']).copy();ix=np.flatnonzero(~accepted);score=np.full(len(accepted),np.nan);added=np.zeros(len(accepted),bool)
  if not len(ix):return {**old,'local_recall_score':score,'local_recall_added':added}
  from .local_patch import features
  import torch
  torch.set_num_threads(1)
  q=result['_query_features'];g=result['_gallery_features'];ql=np.asarray(result['_query_local36']);gl=np.asarray(result['_gallery_local36'])
  if ql.shape!=(len(accepted),36,1024) or gl.shape!=(len(result['gallery']),36,1024) or not np.isfinite(ql).all() or not np.isfinite(gl).all():raise ValueError('Missing actual native local36 descriptors; rebuild gallery')
  owner=result.get('_local_recall_gallery_snapshot')
  if owner is not None and owner is self.owner:gs,gd=self.gallery
  else:
   gs,gd=self.project(g)
   if owner is not None:self.owner=owner;self.gallery=(gs,gd)
  qs,qd=self.project({k:q[k][ix] for k in ['emb','specialist']});top=result['top'][ix];other=gd[top];cos=np.column_stack([np.einsum('ij,ij->i',a,b[top]) for a,b in zip(qs,gs)]).astype(np.float32);pair=np.c_[(qd-other)**2,qd*other,cos];native=np.c_[raw_features(result)[ix][:,[i for i in range(93) if i!=28]],pair,old['confidence'][ix]].astype(np.float32);local=[]
  with torch.inference_mode():
   for i,j in zip(ix,top):local.append(features(torch.from_numpy(np.asarray(ql[i:i+1],np.float32)),torch.from_numpy(np.asarray(gl[j:j+1],np.float32))).numpy()[0])
  x=np.c_[native,np.stack(local)].astype(np.float32);p=np.log(np.clip(self.head.predict_proba(x),1e-12,1));c=self.spec['calibration'];score[ix]=expit((p[:,2]-p[:,0])/c['T']-c['b0']);cut=self.spec['threshold'];added[ix]=score[ix]>=cut;confidence[added]=np.maximum(boundary,expit(logit(np.clip(score[added],1e-12,1-1e-12))-logit(cut)+logit(boundary)));accepted|=added
  assert np.array_equal(accepted,confidence>=boundary) and np.array_equal(confidence[old['accepted']],old['confidence'][old['accepted']])
  return {**old,'accepted':accepted,'confidence':confidence,'local_recall_score':score,'local_recall_added':added}

_STATES={}
def verified_selection(root):
 root=Path(root).resolve();path=Path(os.environ.get('VREID_LOCAL_RECALL_SELECTION',''))
 if not path.is_absolute():raise ValueError('Explicit absolute local verifier selection required')
 spec=json.loads(path.read_text(encoding='utf-8'))
 if spec.get('profile')!='native-local-recall-v1' or spec.get('status')!='FROZEN_FOR_VALIDATION':raise ValueError('Unqualified local profile')
 if digest(root/'var/recall-road-fit-vric-20261018/runtime-selection.json')!=spec['base_selection_SHA256']:raise ValueError('Base selection changed')
 if digest(root/'var/native-tail-20261019/runtime-selection.json')!=spec['tail_selection_SHA256']:raise ValueError('Tail selection changed')
 for f,want in spec['files'].items():
  if digest(f)!=want:raise ValueError('Local asset changed: '+f)
 return spec
def load_state(root):
 spec=verified_selection(root);key=digest(os.environ['VREID_LOCAL_RECALL_SELECTION'])
 if key not in _STATES:_STATES[key]=LocalState(spec,joblib.load(spec['head']),joblib.load(spec['projections']))
 return _STATES[key]
def install_runtime(root):
 state=load_state(root)
 import context
 from cached_post import install
 install()
 from joint_runtime import JointFeatures
 from .local_runtime import install as attach
 attach(JointFeatures,state)
 from . import current_query_profile
 existing=getattr(current_query_profile.route,'_local_recall_state',None)
 if existing is not None and existing is not state:raise ValueError('Restart worker to switch local route')
 if existing is None:
  original=current_query_profile.route
  def route(q,g,cfg):
   r=original(q,g,cfg);r.update(_query_local36=q['local36'],_gallery_local36=g['local36'],_local_recall_gallery_snapshot=g);return r
  route._local_recall_state=state;current_query_profile.route=route
 return state
class LocalRecallPolicy(RecallRescuePolicy):
 @classmethod
 def load(cls,root):return cls(OriginalPolicy.load(root),install_runtime(root))

class LocalTailPolicy:
 def __init__(self,policy):self.policy=policy
 @classmethod
 def load(cls,root):
  from .native_tail import NativeTailPolicy,install_runtime as install_tail
  local=LocalRecallPolicy.load(root);return cls(NativeTailPolicy(local,install_tail(root)))
 @classmethod
 def for_cached_base(cls,base,root):
  from .native_tail import NativeTailPolicy,get_state
  return cls(NativeTailPolicy(LocalRecallPolicy(base,load_state(root)),get_state(root)))
 def predict(self,result,legacy_confidence,boundary):return self.policy.predict(result,legacy_confidence,boundary)
