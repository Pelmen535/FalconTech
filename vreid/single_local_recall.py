"""Explicit distinct global backend; preserve its own base decisions, not old ones."""
from pathlib import Path
import json,os
from .recall_rescue import digest
from .local_recall import load_state,LocalRecallPolicy,install_runtime as install_local
from .refusal_policy_vric_road import VricRoadRefusalPolicy as OriginalPolicy

def verified_selection(root):
 path=Path(os.environ.get('VREID_SINGLE_LOCAL_SELECTION',''))
 if not path.is_absolute():raise ValueError('Explicit absolute single local selection required')
 s=json.loads(path.read_text(encoding='utf-8'))
 if s.get('profile')!='single-local-tail-v1' or s.get('status')!='FROZEN_FOR_VALIDATION':raise ValueError('Unqualified distinct single-local backend')
 if digest(s['local_selection'])!=s['local_selection_SHA256']:raise ValueError('Single local component changed')
 for f,want in s['files'].items():
  if digest(f)!=want:raise ValueError('Single local source changed: '+f)
 os.environ['VREID_LOCAL_RECALL_SELECTION']=s['local_selection'];return s

def install_runtime(root):
 from service import local_native_runtime as transport
 from service.single_local_native import core_class
 old=transport.core_class;transport.core_class=core_class
 try:return install_local(root)
 finally:transport.core_class=old

class SingleTailPolicy:
 def __init__(self,policy):self.policy=policy
 @classmethod
 def load(cls,root):
  verified_selection(root)
  if os.name!='nt':raise ValueError('Single local backend requires separate Linux qualification')
  base=OriginalPolicy.load(root);state=install_runtime(root)
  from .native_tail import NativeTailPolicy,install_runtime as install_tail
  return cls(NativeTailPolicy(LocalRecallPolicy(base,state),install_tail(root)))
 @classmethod
 def for_cached_base(cls,base,root):
  verified_selection(root)
  from .native_tail import NativeTailPolicy,get_state
  return cls(NativeTailPolicy(LocalRecallPolicy(base,load_state(root)),get_state(root)))
 def predict(self,result,legacy_confidence,boundary):return self.policy.predict(result,legacy_confidence,boundary)
