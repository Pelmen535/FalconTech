"""One global correctness head trained on diverse known failures."""
from pathlib import Path
import json
import numpy as np
import joblib
from .refusal_policy_expanded import ExpandedRefusalPolicy as _Expanded
from .refusal_policy_bagged import BaggedRefusalPolicy,digest
from .raw_refusal import raw_features
PROFILE="road-blend-v1"
MANIFEST="var/recall-road-blend-20261018/runtime-selection.json"
def verified_selection(root):
 root=Path(root);s=json.loads((root/MANIFEST).read_text(encoding="utf-8"))
 if s["profile"]!=PROFILE or s["status"] not in ["FROZEN_FOR_VALIDATION","FRESH_CONFIRMED_PENDING_NATIVE","READY_LOCAL_WINDOWS_WITH_TRADEOFF"]:raise ValueError("Global verifier not qualified")
 if not 0<s["threshold"]<1 or not 0<s["alpha"]<=1:raise ValueError("Invalid verifier calibration")
 for name,want in s["files"].items():
  if digest(root/name)!=want:raise ValueError("Global verifier checksum mismatch: "+name)
 return s

def blend(old,new,alpha):
 def logit(p):
  p=np.clip(p,1e-10,1-1e-10);return np.log(p)-np.log1p(-p)
 return 1/(1+np.exp(-np.clip((1-alpha)*logit(old)+alpha*logit(new),-35,35)))

class RoadBlendRefusalPolicy:
 def __init__(self,baseline,model,threshold,alpha,remove_columns):
  if list(model.classes_)!=[0,1] or model.n_features_in_!=93 or not 0<threshold<1 or not 0<alpha<=1:raise ValueError("Invalid global verifier model")
  self.baseline=baseline;self.model=model;self.threshold=float(threshold);self.alpha=float(alpha);self.remove_columns=tuple(remove_columns)
  if self.remove_columns not in [(28,),(28,29)]:raise ValueError("Invalid neutral columns")
 @classmethod
 def load(cls,root):
  root=Path(root);s=verified_selection(root);return cls(_Expanded.load(root),joblib.load(root/s["head"]),s["threshold"],s["alpha"],s["remove_columns"])
 def predict(self,result,legacy_confidence,boundary):
  previous=self.baseline.predict(result,legacy_confidence,boundary);rows=np.flatnonzero(~previous["accepted"])
  if not len(rows):return previous
  sub={k:np.asarray(result[k])[rows] for k in ["raw","rank","query","query_night","top"]}
  sub.update(gallery=result["gallery"],_gallery_features=result["_gallery_features"],_query_features={k:v[rows] for k,v in result["_query_features"].items()})
  features=raw_features(sub).astype(np.float32);features[:,self.remove_columns]=0;score=np.zeros(len(legacy_confidence))
  score[rows]=blend(np.asarray(previous["expanded_score"])[rows],self.model.predict_proba(features)[:,1],self.alpha)
  got=BaggedRefusalPolicy.supplement(self,previous,score,np.zeros(len(score),bool),boundary)
  return {**got,"global_verifier_score":score}
