"""Optional scheduling change: overlap unchanged pixel descriptors with encoding."""
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np


def enable():
 from . import quality_runtime as transport
 if getattr(transport.install,'_overlap_quality',False):return
 ordinary_install=transport.install
 def install(cls,state):
  if getattr(cls.extract,'_quality_recall_state',None) is not None:
   if getattr(cls.extract,'_quality_recall_state',None) is state and getattr(cls.extract,'_quality_overlap',False):return
   raise ValueError('Restart worker before installing overlapping quality extraction')
  original_extract=cls.extract
  ordinary_install(cls,state)
  serial_extract=cls.extract
  bank=state.quality_bank
  def extract(self,records):
   if len(records)!=1:return serial_extract(self,records)
   bank.loaded.clear();bank.in_extract=True;start=time.perf_counter()
   try:
    with ThreadPoolExecutor(max_workers=2) as pool:
     pending=[pool.submit(transport.image_quality30,row) for row in records]
     features,timing=original_extract(self,records)
     photo=np.stack([future.result() for future in pending])
   finally:bank.in_extract=False
   for a,b,v in zip(features['emb'],features['specialist'],photo):bank.remember(a,b,v)
   state.project(features);features['pixel_quality30']=photo
   timing=dict(timing);timing['seconds']=time.perf_counter()-start;timing['FPS']=len(records)/timing['seconds'];timing['quality_schedule']='same per-image photo30, overlap with original encoding, max2 photo workers, unchanged native descriptors';timing['scope']=timing.get('scope','')+'; includes overlapping actual crop quality30 and nativePCA CPU formation'
   return features,timing
  extract._quality_recall_state=state;extract._quality_overlap=True;cls.extract=extract
 install._overlap_quality=True;transport.install=install
