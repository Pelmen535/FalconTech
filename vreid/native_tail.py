"""Optional native-tail-gain-v1: preserve original candidates/confidence, rerank tails.

Only explicit new launchers install this profile. Existing profiles stay unchanged.
"""
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict
import hashlib,json,time
import numpy as np
import joblib
from .refusal_policy_vric_road import VricRoadRefusalPolicy as _OriginalVric
from .raw_refusal import raw_features
PROFILE='native-tail-gain-v1'
MANIFEST='var/native-tail-20261019/runtime-selection.json'
@dataclass(frozen=True)
class PreparedMetricGallery:
 keys:tuple
 projection:np.ndarray
 weight_digest:str
 source_digest:str

class LinearMetric:
 def __init__(self,weight):
  self.weight=np.array(weight,dtype=np.float64,copy=True);assert self.weight.shape==(128,4096) and np.isfinite(self.weight).all();self.weight.flags.writeable=False
  self.weight_digest=hashlib.sha256(self.weight.tobytes()).hexdigest()
 def prepare_gallery(self,pack):
  # Explicit immutable snapshot, not a cache keyed by filenames/IDs alone.
  keys=tuple(map(str,pack['keys']));f=self.project(pack).copy();f.flags.writeable=False;h=hashlib.sha256()
  for value in [np.asarray(keys,dtype='U'),np.asarray(pack['emb']),np.asarray(pack['specialist'])]:h.update(str(value.shape).encode());h.update(value.dtype.str.encode());h.update(value.tobytes())
  return PreparedMetricGallery(keys,f,self.weight_digest,h.hexdigest())
 def rank_cached(self,base,q,gallery,mask,gallery_keys,alpha=.35):
  assert gallery.weight_digest==self.weight_digest,'Metric weights changed: rebuild gallery'
  assert tuple(map(str,gallery_keys))==gallery.keys,'Gallery column order changed: rebuild snapshot'
  sim=self.project(q)@gallery.projection.T;assert sim.shape==base.shape==mask.shape;sim[mask]=-np.inf;rank=(1-alpha)/(20+self.ordinal(base))+alpha/(20+self.ordinal(sim));rank[mask]=-np.inf;return rank
 def project(self,pack):
  if np.asarray(pack['emb']).ndim!=2 or np.asarray(pack['emb']).shape[1]!=2048 or np.asarray(pack['specialist']).shape!=(len(pack['emb']),2048) or not np.isfinite(pack['emb']).all() or not np.isfinite(pack['specialist']).all():raise ValueError('Expected finite native DINO2048/CNN2048')
  xs=[]
  for a,w in zip([pack['emb'][:,:1024],pack['emb'][:,1024:],pack['specialist']],[.25,.25,.5]):
   a=np.asarray(a,dtype=np.float64);xs.append(a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-10)*np.sqrt(w))
  x=np.concatenate(xs,1);f=np.einsum('ni,ji->nj',x,self.weight,optimize=False);return f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-10)
 @staticmethod
 def ordinal(rank):
  order=np.argsort(-rank,axis=1,kind='stable');rr=np.empty_like(order);np.put_along_axis(rr,order,np.arange(rank.shape[1])[None,:],axis=1);return rr
 def rank(self,base,q,g,mask,alpha=.35):
  sim=self.project(q)@self.project(g).T;sim[mask]=-np.inf;rank=(1-alpha)/(20+self.ordinal(base))+alpha/(20+self.ordinal(sim));rank[mask]=-np.inf;return rank

class TailGainFeatures:
 def __init__(self,metric):self.metric=metric;self.cols=[i for i in range(93) if i!=28]
 @staticmethod
 def channels(pack):
  values=[pack['emb'][:,:1024],pack['emb'][:,1024:],pack['specialist']];return [np.asarray(x,dtype=np.float64)/np.maximum(np.linalg.norm(np.asarray(x,dtype=np.float64),axis=1,keepdims=True),1e-10) for x in values]
 def prepare(self,pack):return self.channels(pack)+[self.metric.project(pack)]
 @staticmethod
 def from_prepared(q,g,original,proposal,raw93):
  out=[raw93[:,[i for i in range(93) if i!=28]],(original[:,0]==proposal[:,0])[:,None],np.array([len(set(a)&set(b))/len(a) for a,b in zip(original,proposal)])[:,None]]
  for anchor in ['query','old-candidate']:
   for order in [original,proposal]:
    for qq,gg in zip(q,g):
     a=qq if anchor=='query' else gg[original[:,0]];out.append(np.einsum('ij,ikj->ik',a,gg[order],optimize=False))
  x=np.concatenate(out,1);assert x.shape==(len(original),254) and np.isfinite(x).all();return x
 def features(self,q,g,original,proposal,raw93):return self.from_prepared(self.prepare(q),self.prepare(g),original,proposal,raw93)

def preserve_prefix(original,proposal,prefix):
 original=np.asarray(original);proposal=np.asarray(proposal);assert original.shape==proposal.shape and original.ndim==2 and 1<=prefix<=original.shape[1]
 out=np.empty_like(original)
 for i,(a,b) in enumerate(zip(original,proposal)):
  protected=list(a[:prefix]);rest=[v for v in b if v not in protected];rest.extend(v for v in a[prefix:] if v not in protected and v not in rest);out[i]=protected+rest[:len(a)-prefix]
 assert np.array_equal(out[:,:prefix],original[:,:prefix])
 assert all(len(set(row))==len(row) for row in out)
 return out

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verified_selection(root):
 root=Path(root);s=json.loads((root/MANIFEST).read_text(encoding='utf-8'))
 if s.get('profile')!=PROFILE or s.get('status') not in ('FROZEN_FOR_VALIDATION','READY_LOCAL_WINDOWS_PENDING_LINUX'):
  raise ValueError('Native tail profile is not qualified for this explicit validation launcher')
 if s.get('proposal_mode','global') not in ('global','within-original-top10'):raise ValueError('Invalid native-tail proposal mode')
 if not 1<=s.get('prefix',0)<10 or s.get('alpha')!=.35 or not np.isfinite(s.get('gain_threshold',np.nan)):
  raise ValueError('Invalid native-tail selection')
 if digest(root/'var/recall-road-fit-vric-20261018/runtime-selection.json')!=s['base_selection_SHA256']:
  raise ValueError('Preserved base selection changed')
 for f,want in s['files'].items():
  if digest(root/f)!=want:raise ValueError('Native-tail checksum mismatch: '+f)
 return s

class TailState:
 """One immutable model per worker; caches deterministic per-image descriptors only."""
 def __init__(self,selection,weight,gate):
  if getattr(gate,'n_features_in_',None)!=254:raise ValueError('Expected the frozen 254D AP-gain model')
  self.selection=selection;self.metric=LinearMetric(weight);self.view=TailGainFeatures(self.metric);self.gate=gate
  self.descriptors=OrderedDict();self.gallery_owner=None;self.gallery_prepared=None
  self.descriptor_hits=self.gallery_hits=0
 def token(self,emb,specialist):
  h=hashlib.sha256()
  for a in [emb,specialist]:
   a=np.asarray(a);h.update(a.dtype.str.encode());h.update(a.tobytes(order='C'))
  return h.digest()
 def record(self,pack,projection):
  for a,b,f in zip(pack['emb'],pack['specialist'],projection):
   key=self.token(a,b);self.descriptors[key]=f.copy();self.descriptors.move_to_end(key)
  while len(self.descriptors)>4096:self.descriptors.popitem(last=False)
 def project(self,pack):
  tokens=[self.token(a,b) for a,b in zip(pack['emb'],pack['specialist'])];out=np.empty((len(tokens),128),np.float64);missing=[]
  for i,key in enumerate(tokens):
   if key in self.descriptors:out[i]=self.descriptors[key];self.descriptors.move_to_end(key);self.descriptor_hits+=1
   else:missing.append(i)
  if missing:
   subset={k:pack[k][missing] for k in ['emb','specialist']};fresh=self.metric.project(subset);self.record(subset,fresh);out[missing]=fresh
  return out
 def prepared(self,pack):return self.view.channels(pack)+[self.project(pack)]
 def apply(self,result):
  original=np.argsort(-result['rank'],axis=1,kind='stable');q=result['_query_features'];g=result['_gallery_features'];owner=result.get('_native_tail_gallery_snapshot')
  if owner is not None and owner is self.gallery_owner:
   gp=self.gallery_prepared;self.gallery_hits+=1
  else:
   gp=self.prepared(g)
   if owner is not None:self.gallery_owner=owner;self.gallery_prepared=gp
  qp=self.prepared(q);mask=~np.isfinite(result['raw'])
  if self.selection.get('proposal_mode','global')=='within-original-top10':
   top=original[:,:10];sim=np.einsum('ij,ikj->ik',qp[-1],gp[-1][top],optimize=False);fused=.65/(20+np.arange(10)[None,:])+.35/(20+self.metric.ordinal(sim));proposal=original.copy();proposal[:,:10]=np.take_along_axis(top,np.argsort(-fused,axis=1,kind='stable'),axis=1)
  else:
   sim=qp[-1]@gp[-1].T;sim[mask]=-np.inf;fused=.65/(20+self.metric.ordinal(result['rank']))+.35/(20+self.metric.ordinal(sim));fused[mask]=-np.inf;proposal=np.argsort(-fused,axis=1,kind='stable')
  prefix=self.selection['prefix'];desired=preserve_prefix(original[:,:10],proposal[:,:10],prefix)
  # If the public top10 is already identical, no score or output needs changing.
  different=np.any(desired!=original[:,:10],axis=1);score=np.full(len(original),-np.inf);chosen=np.zeros(len(original),bool)
  if different.any():
   x=self.view.from_prepared([a[different] for a in qp],gp,original[different,:10],proposal[different,:10],raw_features(result)[different]);score[different]=self.gate.predict(x);assert np.isfinite(score[different]).all();chosen=different&(score>=self.selection['gain_threshold'])
  rank=result['rank'].copy()
  for i in np.flatnonzero(chosen):
   protected=original[i,:prefix];tail=[j for j in proposal[i] if j not in protected];order=np.r_[protected,tail];floor=float(result['rank'][i,protected[-1]])
   # The protected scores remain byte-identical, including HTTP candidate metadata.
   # Tail scores are ordering priorities, not graph-distance probabilities.
   rank[i,order[prefix:]]=floor-1-np.arange(len(tail))/len(order)
   rank[i,protected]=result['rank'][i,protected];rank[i,mask[i]]=-np.inf
   assert np.array_equal(np.argsort(-rank[i],kind='stable')[:10],desired[i])
  return rank,score,chosen

_STATES={}

def get_state(root):
 root=Path(root).resolve();s=verified_selection(root);key=(str(root),digest(root/MANIFEST))
 if key not in _STATES:_STATES[key]=TailState(s,np.load(root/s['projection']),joblib.load(root/s['gate']))
 return _STATES[key]

def install_extractor(cls,state):
 existing=getattr(cls.extract,'_native_tail_state',None)
 if existing is not None:
  if existing is not state:raise ValueError('Restart worker to switch native-tail weights')
  return
 original=cls.extract
 def extract(self,records):
  features,timing=original(self,records);start=time.perf_counter();projection=state.metric.project(features);state.record(features,projection);extra=time.perf_counter()-start;features['native_tail_descriptor']=projection
  timing=dict(timing);timing['seconds']=float(timing['seconds'])+extra;timing['FPS']=len(records)/timing['seconds'];timing['native_tail_descriptor_seconds']=extra;timing['scope']=timing.get('scope','')+'; includes native-tail per-image CPU descriptor and transfer cache; baseline remaining projections keep their original separate timing scope'
  return features,timing
 extract._native_tail_state=state;cls.extract=extract

def install_runtime(root):
 state=get_state(root)
 # Existing core bootstrap establishes the same native paths/backend as base CLI.
 import context
 from cached_post import install
 install()
 from joint_runtime import JointFeatures
 install_extractor(JointFeatures,state)
 from . import current_query_profile
 if not getattr(current_query_profile.route,'_native_tail_snapshot',False):
  original=current_query_profile.route
  def route(q,g,cfg):
   result=original(q,g,cfg);result['_native_tail_gallery_snapshot']=g;result['_gallery_keys']=g['keys'];return result
  route._native_tail_snapshot=True;current_query_profile.route=route
 return state

class NativeTailPolicy:
 def __init__(self,baseline,state):self.baseline=baseline;self.state=state
 @classmethod
 def load(cls,root):
  baseline=_OriginalVric.load(root);state=install_runtime(root);return cls(baseline,state)
 def predict(self,result,legacy_confidence,boundary):
  baseline=result.get('_native_tail_base_result')
  if baseline is None:
   baseline=dict(result);result['_native_tail_base_result']=baseline
  out=self.baseline.predict(baseline,legacy_confidence,boundary)
  if '_native_tail_rank' not in result:
   rank,score,chosen=self.state.apply(baseline);result['_native_tail_rank']=rank;result['_native_tail_gain_score']=score;result['_native_tail_chosen']=chosen
  result['rank']=result['_native_tail_rank']
  assert np.array_equal(np.argsort(-result['rank'],axis=1,kind='stable')[:,:self.state.selection['prefix']],np.argsort(-baseline['rank'],axis=1,kind='stable')[:,:self.state.selection['prefix']])
  return {**out,'native_tail_gain_score':result['_native_tail_gain_score'],'native_tail_applied':result['_native_tail_chosen']}
