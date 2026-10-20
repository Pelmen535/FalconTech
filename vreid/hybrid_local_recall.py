"""Frozen complementary quality then local rescue, before unchanged D29 tail."""
from pathlib import Path
import os,json
import numpy as np
from .recall_rescue import digest
from .local_recall import load_state as local_state,install_runtime as install_local
from .quality_recall import load_state as quality_state
from .refusal_policy_vric_road import VricRoadRefusalPolicy as OriginalPolicy

def verified_selection(root):
 root=Path(root).resolve();path=Path(os.environ.get('VREID_HYBRID_LOCAL_SELECTION',''))
 if not path.is_absolute():raise ValueError('Explicit absolute hybrid selection required')
 s=json.loads(path.read_text(encoding='utf-8'))
 if s.get('profile')!='quality-local-recall-v1' or s.get('status')!='FROZEN_FOR_VALIDATION':raise ValueError('Unqualified hybrid profile')
 for name in ['local_selection','quality_selection']:
  if digest(s[name])!=s[name+'_SHA256']:raise ValueError('Hybrid component changed: '+name)
 for f,want in s['files'].items():
  if digest(f)!=want:raise ValueError('Hybrid source changed: '+f)
 os.environ['VREID_LOCAL_RECALL_SELECTION']=s['local_selection'];os.environ['VREID_QUALITY_RECALL_SELECTION']=s['quality_selection'];return s

def install_runtime(root):
 local=install_local(root);quality=quality_state(root)
 import joint_runtime
 cls=joint_runtime.JointFeatures
 if getattr(cls.extract,'_hybrid_quality',None) is not None:
  if cls.extract._hybrid_quality is not quality:raise ValueError('Restart worker to switch hybrid quality state')
  return quality,local
 from .hybrid_local_runtime import attach
 attach(cls,local,quality)
 from . import current_query_profile
 original=current_query_profile.route
 def route(q,g,cfg):
  r=original(q,g,cfg);r.update(_query_quality30=q['pixel_quality30'],_gallery_quality30=g['pixel_quality30'],_quality_recall_gallery_snapshot=g);return r
 current_query_profile.route=route
 return quality,local

class HybridPolicy:
 def __init__(self,base,quality,local):self.base=base;self.quality=quality;self.local=local
 def predict(self,result,legacy,boundary):
  original=self.base.predict(result,legacy,boundary);quality=self.quality.predict(result,original,boundary);out=self.local.predict(result,quality,boundary)
  assert np.array_equal(out['confidence'][original['accepted']],original['confidence'][original['accepted']])
  return out

class HybridTailPolicy:
 def __init__(self,policy):self.policy=policy
 @classmethod
 def load(cls,root):
  verified_selection(root)
  if os.name!='nt':raise ValueError('Hybrid local profile awaits Linux qualification')
  base=OriginalPolicy.load(root);quality,local=install_runtime(root)
  from .native_tail import NativeTailPolicy,install_runtime as install_tail
  return cls(NativeTailPolicy(HybridPolicy(base,quality,local),install_tail(root)))
 @classmethod
 def for_cached_base(cls,base,root):
  verified_selection(root)
  from .native_tail import NativeTailPolicy,get_state
  return cls(NativeTailPolicy(HybridPolicy(base,quality_state(root),local_state(root)),get_state(root)))
 def predict(self,result,legacy_confidence,boundary):return self.policy.predict(result,legacy_confidence,boundary)
