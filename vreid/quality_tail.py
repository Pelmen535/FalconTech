"""Frozen quality refusal BEFORE the existing prefix-two tail ranking profile."""
from pathlib import Path
import json,os
from .quality_recall import QualityRecallPolicy,load_state as quality_state
from .native_tail import NativeTailPolicy,get_state as tail_state,install_runtime as install_tail
from .recall_rescue import digest
_LOADED={}

def verified_selection(root):
 root=Path(root).resolve();path=Path(os.environ.get('VREID_QUALITY_TAIL_SELECTION',''))
 if not path.is_absolute():raise ValueError('Explicit absolute composite selection required')
 s=json.loads(path.read_text(encoding='utf-8'))
 if s.get('profile')!='quality-recall-tail-v1' or s.get('status') not in ['FROZEN_FOR_VALIDATION','READY_LOCAL_WINDOWS_PENDING_LINUX']:raise ValueError('Unqualified quality-tail profile')
 if digest(s['quality_selection'])!=s['quality_selection_SHA256']:raise ValueError('Quality component selection changed')
 if digest(root/'var/native-tail-20261019/runtime-selection.json')!=s['tail_selection_SHA256']:raise ValueError('Tail component selection changed')
 for name,want in s['files'].items():
  if digest(name)!=want:raise ValueError('Composite source changed: '+name)
 os.environ['VREID_QUALITY_RECALL_SELECTION']=s['quality_selection'];return s

class QualityTailPolicy:
 def __init__(self,policy,selection):self.policy=policy;self.selection=selection
 @classmethod
 def load(cls,root):
  s=verified_selection(root);key=(str(Path(root).resolve()),digest(os.environ['VREID_QUALITY_TAIL_SELECTION']))
  if key in _LOADED:return _LOADED[key]
  if _LOADED:raise ValueError('Restart worker to switch composite selection')
  quality=QualityRecallPolicy.load(root);tail=install_tail(root);value=cls(NativeTailPolicy(quality,tail),s);_LOADED[key]=value;return value
 @classmethod
 def for_cached_base(cls,base,root):
  s=verified_selection(root);return cls(NativeTailPolicy(QualityRecallPolicy(base,quality_state(root)),tail_state(root)),s)
 def predict(self,result,legacy_confidence,boundary):return self.policy.predict(result,legacy_confidence,boundary)
