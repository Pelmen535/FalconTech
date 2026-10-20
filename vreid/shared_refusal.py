"""Optional evaluation of the frozen VricRoad chain with shared primary raw93."""
import numpy as np
from .raw_refusal import raw_features
from .refusal_policy_bagged import BaggedRefusalPolicy
from .refusal_policy_road_blend import blend

class SharedVricRoadPolicy:
 """No new model/cut. Qualification must compare real outputs to original.predict."""
 def __init__(self,original):
  self.original=original;self.expanded=original.baseline
  if not hasattr(self.expanded,'previous_features'):raise ValueError('Expected frozen Expanded/VricRoad chain')
 def predict(self,result,legacy_confidence,boundary):
  features=raw_features(result);expanded=self.expanded;previous=expanded.previous_features(result,features,legacy_confidence,boundary);rows=np.flatnonzero(~previous['accepted'])
  if not len(rows):return previous
  score=np.zeros(len(features),float);score[rows]=expanded.model.predict_proba(features[rows].astype(np.float32))[:,1];previous={**BaggedRefusalPolicy.supplement(expanded,previous,score,np.zeros(len(score),bool),boundary),'expanded_score':score};rows=np.flatnonzero(~previous['accepted'])
  if not len(rows):return previous
  xx=features[rows].astype(np.float32);xx[:,self.original.remove_columns]=0;score=np.zeros(len(features));score[rows]=blend(np.asarray(previous['expanded_score'])[rows],self.original.model.predict_proba(xx)[:,1],self.original.alpha);out=BaggedRefusalPolicy.supplement(self.original,previous,score,np.zeros(len(score),bool),boundary);return {**out,'global_verifier_score':score}
