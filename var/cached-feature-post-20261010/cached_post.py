from env import *
import evaluate as e
import recipe
SOURCE={k:bc.Z[k] for k in bc.Z.files} if hasattr(bc.Z,'files') else bc.Z
AUX={k:bc.OLDZ[k] for k in bc.OLDZ.files} if hasattr(bc.OLDZ,'files') else bc.OLDZ
NIGHT={k:e.z[k] for k in e.z.files} if hasattr(e.z,'files') else e.z
DOMAIN=e.domain
for data in [SOURCE,AUX,NIGHT]:
 for v in data.values():
  if isinstance(v,np.ndarray):v.setflags(write=False)
def teacher_cached(raw):
 base=bc.transform({'emb':raw},SOURCE['transform'],SOURCE['mean'])['emb'];mask=DOMAIN.predict_proba(base)[:,1]>=.95
 if mask.any():base[mask]=bc.transform({'emb':raw[mask]},NIGHT['transform'],NIGHT['mean'])['emb']
 return base

def install():
 bc.Z=SOURCE;bc.OLDZ=AUX;e.z=NIGHT;recipe.z=NIGHT
 import fusion_evaluate as f
 f.z=NIGHT
 return {'source_bytes':sum(v.nbytes for v in SOURCE.values() if isinstance(v,np.ndarray)),'night_bytes':sum(v.nbytes for v in NIGHT.values() if isinstance(v,np.ndarray)),'aux_bytes':sum(v.nbytes for v in AUX.values() if isinstance(v,np.ndarray)),'scope':'immutable matrices/domain resident before warmup, no values/weights changed'}
