"""Optional FIT-trained quality570 refusal supplement; baseline accepted outputs stay exact."""
from pathlib import Path
from collections import OrderedDict
import os,json
import joblib
import numpy as np
from scipy.special import expit,logit
from .recall_rescue import RescueState,RecallRescuePolicy,digest
from .raw_refusal import raw_features
from .refusal_policy_vric_road import VricRoadRefusalPolicy as OriginalPolicy
from .quality_runtime import QualityBank

class QualityState(RescueState):
 def __init__(self,spec,head,pcas):
  if spec.get('status') not in ['FROZEN_FOR_VALIDATION','READY_LOCAL_WINDOWS_PENDING_LINUX']:raise ValueError('Unqualified quality selection')
  if getattr(head,'n_features_in_',None)!=570 or list(head.classes_)!=[0,1,2]:raise ValueError('Expected quality570 three-class head')
  c=spec['calibration']
  if not (0<spec['threshold']<1 and c['T']>0 and np.isfinite([c['T'],c['b0'],c['b1']]).all()):raise ValueError('Invalid quality calibration')
  if len(pcas)!=3 or any(p.components_.shape!=(64,d) for p,d in zip(pcas,[1024,1024,2048])):raise ValueError('Invalid native projections')
  self.spec=spec;self.head=head;self.pcas=pcas;self.cache=OrderedDict();self.owner=None;self.gallery=None;self.quality_bank=QualityBank()
 def predict(self,result,old,boundary):
  a=np.asarray(old['accepted'],bool).copy();confidence=np.asarray(old['confidence']).copy();ix=np.flatnonzero(~a);score=np.full(len(a),np.nan);add=np.zeros(len(a),bool)
  if not len(ix):return {**old,'quality_recall_score':score,'quality_recall_added':add}
  qp=np.asarray(result['_query_quality30'],np.float32);gp=np.asarray(result['_gallery_quality30'],np.float32)
  if qp.shape!=(len(a),30) or gp.shape!=(len(result['gallery']),30) or not np.isfinite(qp).all() or not np.isfinite(gp).all():raise ValueError('Expected actual query/gallery crop quality30')
  owner=result.get('_quality_recall_gallery_snapshot');g=result['_gallery_features']
  if owner is not None and owner is self.owner:gs,gd=self.gallery
  else:
   gs,gd=self.project(g)
   if owner is not None:self.owner=owner;self.gallery=(gs,gd)
  qs,qd=self.project({k:result['_query_features'][k][ix] for k in ['emb','specialist']});top=result['top'][ix];b=gd[top];cos=np.column_stack([np.einsum('ij,ij->i',q,v[top]) for q,v in zip(qs,gs)]).astype(np.float32);pair=np.c_[(qd-b)**2,qd*b,cos];native=np.c_[raw_features(result)[ix][:,[i for i in range(93) if i!=28]],pair].astype(np.float32);qphoto,gphoto=qp[ix],gp[top];x=np.c_[native,old['confidence'][ix],qphoto,gphoto,np.abs(qphoto-gphoto)];p=np.log(np.clip(self.head.predict_proba(x),1e-12,1));c=self.spec['calibration'];v=expit((p[:,2]-p[:,0])/c['T']-c['b0']);score[ix]=v;cut=self.spec['threshold'];add[ix]=v>=cut;confidence[add]=np.maximum(boundary,expit(logit(np.clip(score[add],1e-12,1-1e-12))-logit(cut)+logit(boundary)));a|=add;assert np.array_equal(a,confidence>=boundary) and np.array_equal(confidence[old['accepted']],old['confidence'][old['accepted']]);return {**old,'accepted':a,'confidence':confidence,'quality_recall_score':score,'quality_recall_added':add}

_STATES={}
def load_state(root):
 root=Path(root).resolve();path=Path(os.environ.get('VREID_QUALITY_RECALL_SELECTION',''))
 if not path.is_absolute():raise ValueError('Explicit absolute quality selection required')
 s=json.loads(path.read_text(encoding='utf-8'));key=(str(root),digest(path))
 if s.get('profile')!='quality-recall-rescue-v1':raise ValueError('Wrong quality profile')
 if digest(root/'var/recall-road-fit-vric-20261018/runtime-selection.json')!=s['base_selection_SHA256']:raise ValueError('Base selection changed')
 for f,want in s['files'].items():
  if digest(f)!=want:raise ValueError('Quality profile asset changed: '+f)
 if key not in _STATES:_STATES[key]=QualityState(s,joblib.load(s['head']),joblib.load(s['projections']))
 return _STATES[key]
def install_runtime(root):
 state=load_state(root)
 import context
 from cached_post import install
 install()
 from joint_runtime import JointFeatures
 from .quality_runtime import install as attach
 attach(JointFeatures,state)
 from . import current_query_profile
 existing=getattr(current_query_profile.route,'_quality_recall_state',None)
 if existing is not None and existing is not state:raise ValueError('Restart worker to switch quality route')
 if existing is None:
  original=current_query_profile.route
  def route(q,g,cfg):
   r=original(q,g,cfg);r['_query_quality30']=q['pixel_quality30'];r['_gallery_quality30']=g['pixel_quality30'];r['_quality_recall_gallery_snapshot']=g;return r
  route._quality_recall_state=state;current_query_profile.route=route
 return state
class QualityRecallPolicy(RecallRescuePolicy):
 @classmethod
 def load(cls,root):return cls(OriginalPolicy.load(root),install_runtime(root))
