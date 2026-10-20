"""Transport actual local36 in raw gallery NPZ; query metadata stays in memory."""
from collections import OrderedDict
import time
import numpy as np
from .quality_runtime import QualityBank

class LocalBank:
 key=staticmethod(QualityBank.key)
 def __init__(self):self.recent=OrderedDict();self.loaded={};self.in_extract=False
 def remember(self,emb,cnn,value,loaded=False):
  key=self.key(emb,cnn);value=np.asarray(value,np.float16)
  if value.shape!=(36,1024) or not np.isfinite(value).all():raise ValueError('Invalid actual local36 metadata')
  target=self.loaded if loaded else self.recent
  for cache in [self.loaded,self.recent]:
   if key in cache and not np.array_equal(cache[key],value):raise ValueError('Native/local descriptor collision; rebuild gallery or use a consistent numeric batch regime')
  target[key]=value.copy()
  if not loaded:
   self.recent.move_to_end(key)
   while len(self.recent)>4096:self.recent.popitem(last=False)
 def lookup(self,emb,cnn):
  key=self.key(emb,cnn)
  if key in self.recent:self.recent.move_to_end(key);return self.recent[key]
  if key in self.loaded:return self.loaded[key]
  raise ValueError('Local36 metadata missing; rebuild gallery with this profile')

def install(cls,state):
 existing=getattr(cls.extract,'_local_recall_state',None)
 if existing is not None:
  if existing is not state:raise ValueError('Restart worker to switch local model')
  return
 if getattr(cls.extract,'_quality_recall_state',None) is not None:raise ValueError('Separate worker required for local and quality models')
 import joint_runtime
 from service.local_native_runtime import core_class
 ordinary_core=joint_runtime.Core;ordinary_init=cls.__init__;ordinary_extract=cls.extract;ordinary_pack=cls.pack;ordinary_load=np.load;ordinary_savez=np.savez;bank=LocalBank();state.local_bank=bank;joint_runtime.Core=core_class(ordinary_core,state.spec)
 def init(self,*args,**kwargs):
  ordinary_init(self,*args,**kwargs);self.engine=state.spec['local_engine']
 def pack(self,records,features):
  out=ordinary_pack(self,records,features)
  if bank.in_extract:return out
  if 'local36' not in out:out['local36']=np.stack([bank.lookup(a,b) for a,b in zip(out['emb'],out['specialist'])])
  if out['local36'].shape!=(len(records),36,1024) or not np.isfinite(out['local36']).all():raise ValueError('Invalid packed local descriptors')
  bank.loaded.clear();return out
 def extract(self,records):
  bank.loaded.clear();bank.in_extract=True
  try:features,timing=ordinary_extract(self,records)
  finally:bank.in_extract=False
  start=time.perf_counter()
  for a,b,v in zip(features['emb'],features['specialist'],features['local36']):bank.remember(a,b,v)
  state.project(features);extra=time.perf_counter()-start;timing=dict(timing);timing['seconds']+=extra;timing['FPS']=len(records)/timing['seconds'];timing['native_local_PCA_seconds']=extra;timing['scope']=timing.get('scope','')+'; includes additional native DINO local36 pass and fixed native192 projection';return features,timing
 def savez(file,*args,**kwargs):
  if not args and set(kwargs)=={'emb','specialist','base'}:kwargs={**kwargs,'local36':bank.lookup(kwargs['emb'],kwargs['specialist']).copy()}
  return ordinary_savez(file,*args,**kwargs)
 def load(file,*args,**kwargs):
  value=ordinary_load(file,*args,**kwargs)
  if hasattr(value,'files') and {'emb','specialist','base','local36'}<=set(value.files):
   try:bank.remember(value['emb'],value['specialist'],value['local36'],loaded=True)
   except Exception:value.close();raise
  return value
 extract._local_recall_state=state;cls.__init__=init;cls.extract=extract;cls.pack=pack;np.load=load;np.savez=savez
 def restore():joint_runtime.Core=ordinary_core;cls.__init__=ordinary_init;cls.extract=ordinary_extract;cls.pack=ordinary_pack;np.load=ordinary_load;np.savez=ordinary_savez
 state.restore_local_hooks=restore
