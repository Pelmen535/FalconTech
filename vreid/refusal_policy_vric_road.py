"""Global road-fit verifier; reuse the frozen query-independent blend rule."""
from pathlib import Path
import json,joblib
from .refusal_policy_road_blend import RoadBlendRefusalPolicy as _Rule,blend
from .refusal_policy_expanded import ExpandedRefusalPolicy as _Expanded
from .refusal_policy_bagged import digest
PROFILE='vric-road-verifier-v1'
MANIFEST='var/recall-road-fit-vric-20261018/runtime-selection.json'

def verified_selection(root):
 root=Path(root);s=json.loads((root/MANIFEST).read_text(encoding='utf-8'))
 if s['profile']!=PROFILE or s['status'] not in ['FROZEN_FOR_VALIDATION','FRESH_CONFIRMED_PENDING_NATIVE','READY_LOCAL_WINDOWS_WITH_TRADEOFF']:raise ValueError('Global verifier not qualified')
 if not 0<s['threshold']<1 or not 0<s['alpha']<=1:raise ValueError('Invalid verifier calibration')
 for name,want in s['files'].items():
  if digest(root/name)!=want:raise ValueError('Global verifier checksum mismatch: '+name)
 return s

class VricRoadRefusalPolicy(_Rule):
 @classmethod
 def load(cls,root):
  root=Path(root);s=verified_selection(root)
  return cls(_Expanded.load(root),joblib.load(root/s['head']),s['threshold'],s['alpha'],s['remove_columns'])
