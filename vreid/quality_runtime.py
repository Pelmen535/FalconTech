"""Process-local pixel metadata transport for the optional quality verifier.

Gallery metadata lives in its existing atomic raw-feature NPZ. Query metadata is
memory-only. Neither filenames nor previous query results determine the features.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib,time
import numpy as np
from PIL import Image
from .pixel_quality import quality

def image_quality30(record):
 from .extract import open_crop
 if record.get('bbox') is None:
  with Image.open(record['path']) as f:return quality(f)[:30]
 return quality(open_crop(record['path'],record['bbox'],pad=.05,fast_decode=True,target=384))[:30]

class QualityBank:
 def __init__(self):self.recent=OrderedDict();self.loaded={};self.in_extract=False
 @staticmethod
 def key(emb,cnn):
  h=hashlib.sha256()
  for a in [emb,cnn]:
   a=np.asarray(a,dtype='<f4')
   if a.shape!=(2048,) or not np.isfinite(a).all():raise ValueError('Invalid native quality key')
   h.update(a.tobytes())
  return h.digest()
 def remember(self,emb,cnn,value,loaded=False):
  key=self.key(emb,cnn);value=np.asarray(value,np.float32)
  if value.shape!=(30,) or not np.isfinite(value).all():raise ValueError('Invalid pixel quality metadata')
  for cache in [self.recent,self.loaded]:
   if key in cache and not np.array_equal(cache[key],value):raise ValueError('Native feature collision with different pixels/quality; rebuild gallery')
  if loaded:self.loaded[key]=value.copy()
  else:
   self.recent[key]=value.copy();self.recent.move_to_end(key)
   while len(self.recent)>4096:self.recent.popitem(last=False)
 def lookup(self,emb,cnn):
  key=self.key(emb,cnn)
  if key in self.recent:self.recent.move_to_end(key);return self.recent[key]
  if key in self.loaded:return self.loaded[key]
  raise ValueError('Pixel quality metadata missing; rebuild gallery with this profile')

def install(cls,state):
 existing=getattr(cls.extract,'_quality_recall_state',None)
 if existing is not None:
  if existing is not state:raise ValueError('Restart worker to switch quality model')
  return
 bank=state.quality_bank;original_extract=cls.extract;original_pack=cls.pack;original_load=np.load;original_savez=np.savez
 def pack(self,records,features):
  out=original_pack(self,records,features)
  if bank.in_extract:return out
  if 'pixel_quality30' not in out:out['pixel_quality30']=np.stack([bank.lookup(a,b) for a,b in zip(out['emb'],out['specialist'])])
  if out['pixel_quality30'].shape!=(len(records),30) or not np.isfinite(out['pixel_quality30']).all():raise ValueError('Invalid packed pixel quality')
  bank.loaded.clear();return out
 def extract(self,records):
  bank.loaded.clear();bank.in_extract=True
  try:features,timing=original_extract(self,records)
  finally:bank.in_extract=False
  start=time.perf_counter()
  with ThreadPoolExecutor(max_workers=2) as pool:photo=np.stack(list(pool.map(image_quality30,records)))
  for a,b,v in zip(features['emb'],features['specialist'],photo):bank.remember(a,b,v)
  state.project(features);features['pixel_quality30']=photo;extra=time.perf_counter()-start;timing=dict(timing);timing['seconds']=float(timing['seconds'])+extra;timing['FPS']=len(records)/timing['seconds'];timing['quality_and_native_descriptor_seconds']=extra;timing['scope']=timing.get('scope','')+'; includes actual crop pixel quality30 and nativePCA CPU formation';return features,timing
 def savez(file,*args,**kwargs):
  if not args and set(kwargs)=={'emb','specialist','base'}:
   kwargs={**kwargs,'pixel_quality30':bank.lookup(kwargs['emb'],kwargs['specialist']).copy()}
  return original_savez(file,*args,**kwargs)
 def load(file,*args,**kwargs):
  value=original_load(file,*args,**kwargs)
  if hasattr(value,'files') and {'emb','specialist','base','pixel_quality30'}<=set(value.files):
   try:bank.remember(value['emb'],value['specialist'],value['pixel_quality30'],loaded=True)
   except Exception:value.close();raise
  return value
 extract._quality_recall_state=state;cls.extract=extract;cls.pack=pack;np.savez=savez;np.load=load
 # Test-only restoration; worker lifetime normally ends all process-local hooks.
 def restore():cls.extract=original_extract;cls.pack=original_pack;np.load=original_load;np.savez=original_savez
 state.restore_quality_hooks=restore
