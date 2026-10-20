"""Conservative source supplement using learned local-gallery verification."""
import numpy as np
from .refusal_policy import GeneralRefusalPolicy as BasePolicy
from .local_refusal import local_features


class GeneralRefusalPolicy(BasePolicy):
    def __init__(self,model,kind,threshold,mode):
        if kind!='local' or mode!='supplement_source_only':raise ValueError('Invalid local policy')
        if not 0<=threshold<=1 or list(model.classes_)!=[0,1]:raise ValueError('Invalid local model')
        self.model=model;self.kind=kind;self.threshold=float(threshold);self.mode=mode

    def predict(self,result,legacy_confidence,boundary):
        legacy=np.asarray(legacy_confidence,dtype=float)
        if legacy.shape!=(len(result['top']),) or not np.isfinite(legacy).all() or not 0<boundary<1:raise ValueError('Invalid legacy confidence')
        score=self.model.predict_proba(local_features(result))[:,1]
        if not np.isfinite(score).all() or np.any((score<0)|(score>1)):raise ValueError('Invalid local score')
        accepted=(score>=self.threshold)&~np.asarray(result['query_night'],dtype=bool)
        def logit(p):
            p=np.clip(p,1e-12,1-1e-12);return np.log(p/(1-p))
        shifted=1/(1+np.exp(-np.clip(logit(score)-logit(self.threshold)+logit(boundary),-50,50)))
        shifted=np.where(accepted,np.maximum(shifted,boundary),np.minimum(shifted,np.nextafter(boundary,-np.inf)))
        confidence=np.where(accepted&(legacy<boundary),shifted,legacy);final=legacy>=boundary
        final|=accepted;assert np.array_equal(final,confidence>=boundary)
        return {'raw_score':score,'confidence':confidence,'accepted':final}
