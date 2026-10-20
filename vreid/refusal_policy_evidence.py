"""Single global smooth match-evidence correction, no gallery-style routing."""
from pathlib import Path
import json
import numpy as np
from .refusal_policy_expanded import ExpandedRefusalPolicy as _Expanded
from .refusal_policy_bagged import BaggedRefusalPolicy,digest
from .refusal_heuristics import RAW_NAMES
from .raw_refusal import raw_features
PROFILE="soft-evidence-v1"
MANIFEST="var/recall-refinement-20261018/runtime-selection.json"
def verified_selection(root):
 root=Path(root);s=json.loads((root/MANIFEST).read_text(encoding="utf-8"))
 if s["profile"]!=PROFILE or s["feature"]!="patch_minus_cnn" or s["status"] not in ["FROZEN_FOR_VALIDATION","FRESH_CONFIRMED_PENDING_NATIVE","READY_LOCAL_WINDOWS_WITH_TRADEOFF"]:raise ValueError("Evidence profile not qualified")
 if not 0<s["threshold"]<1 or not np.isfinite(s["alpha"]):raise ValueError("Invalid evidence calibration")
 for name,want in s["files"].items():
  if digest(root/name)!=want:raise ValueError("Evidence checksum mismatch: "+name)
 return s

def corrected_score(x,p,alpha):
 x=np.asarray(x,dtype=np.float32)
 if x.ndim!=2 or x.shape[1]!=93 or not np.isfinite(x).all():raise ValueError("Expected finite raw93")
 a=x[:,RAW_NAMES.index("DINO_PATCH_MEAN_selected_cosine")].astype(np.float64)
 b=x[:,RAW_NAMES.index("CNN_RAW_selected_cosine")].astype(np.float64)
 p=np.clip(p,1e-10,1-1e-10);z=np.log(p)-np.log1p(-p)+alpha*4*(a-b)
 return 1/(1+np.exp(-np.clip(z,-35,35)))

class EvidenceRefusalPolicy:
 def __init__(self,baseline,threshold,alpha):self.baseline=baseline;self.threshold=float(threshold);self.alpha=float(alpha)
 @classmethod
 def load(cls,root):
  s=verified_selection(root);return cls(_Expanded.load(root),s["threshold"],s["alpha"])
 def predict(self,result,legacy_confidence,boundary):
  previous=self.baseline.predict(result,legacy_confidence,boundary);rows=np.flatnonzero(~previous["accepted"])
  if not len(rows):return previous
  sub={k:np.asarray(result[k])[rows] for k in ["raw","rank","query","query_night","top"]}
  sub.update(gallery=result["gallery"],_gallery_features=result["_gallery_features"],_query_features={k:v[rows] for k,v in result["_query_features"].items()})
  score=np.asarray(previous["expanded_score"]).copy();score[rows]=corrected_score(raw_features(sub),score[rows],self.alpha)
  got=BaggedRefusalPolicy.supplement(self,previous,score,np.zeros(len(score),bool),boundary)
  return {**got,"evidence_score":score}
