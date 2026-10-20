"""Both fixed image descriptors in one gallery NPZ, all formation cost counted."""
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from .quality_runtime import image_quality30

def attach(cls,local,quality):
 oldextract=cls.extract;oldpack=cls.pack;oldsavez=np.savez;oldload=np.load;bank=quality.quality_bank
 def pack(self,records,features):
  out=oldpack(self,records,features)
  if bank.in_extract:return out
  if 'pixel_quality30' not in out:out['pixel_quality30']=np.stack([bank.lookup(a,b) for a,b in zip(out['emb'],out['specialist'])])
  if out['pixel_quality30'].shape!=(len(records),30) or not np.isfinite(out['pixel_quality30']).all():raise ValueError('Invalid hybrid pixel metadata')
  bank.loaded.clear();return out
 def extract(self,records):
  bank.loaded.clear();bank.in_extract=True
  try:features,timing=oldextract(self,records)
  finally:bank.in_extract=False
  start=time.perf_counter()
  with ThreadPoolExecutor(max_workers=2) as pool:photo=np.stack(list(pool.map(image_quality30,records)))
  for a,b,v in zip(features['emb'],features['specialist'],photo):bank.remember(a,b,v)
  quality.project(features);features['pixel_quality30']=photo;extra=time.perf_counter()-start;timing=dict(timing);timing['seconds']+=extra;timing['FPS']=len(records)/timing['seconds'];timing['hybrid_photo_and_PCA_seconds']=extra;timing['scope']=timing.get('scope','')+'; includes exact crop quality30/native192 as well as all three native encoder passes';return features,timing
 def savez(file,*args,**kwargs):
  if not args and set(kwargs)=={'emb','specialist','base'}:kwargs={**kwargs,'local36':local.local_bank.lookup(kwargs['emb'],kwargs['specialist']).copy(),'pixel_quality30':bank.lookup(kwargs['emb'],kwargs['specialist']).copy()}
  return oldsavez(file,*args,**kwargs)
 def load(file,*args,**kwargs):
  value=oldload(file,*args,**kwargs)
  if hasattr(value,'files') and {'emb','specialist','base','local36','pixel_quality30'}<=set(value.files):
   try:bank.remember(value['emb'],value['specialist'],value['pixel_quality30'],loaded=True)
   except Exception:value.close();raise
  return value
 extract._hybrid_quality=quality;cls.extract=extract;cls.pack=pack;np.savez=savez;np.load=load
 def restore():cls.extract=oldextract;cls.pack=oldpack;np.savez=oldsavez;np.load=oldload;local.restore_local_hooks()
 quality.restore_hybrid_hooks=restore
