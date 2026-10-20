"""Frozen learned acceptance, independent of labels and other query rows."""
from pathlib import Path
import hashlib,json
import numpy as np
from .general_refusal import model_features


class GeneralRefusalPolicy:
    def __init__(self,model,kind,threshold,mode):
        if kind not in ('relative','mixed') or mode not in ('supplement','replace'):
            raise ValueError('Invalid refusal policy')
        if not 0<=threshold<=1:raise ValueError('Invalid refusal threshold')
        if list(model.classes_)!=[0,1]:raise ValueError('Expected binary correct-versus-rest model')
        self.model=model;self.kind=kind;self.threshold=float(threshold);self.mode=mode

    @classmethod
    def load(cls,selection_path):
        import joblib
        path=Path(selection_path);selection=json.loads(path.read_text(encoding='utf-8-sig'));winner=selection['winner']
        if not winner:raise ValueError('No selected general refusal head')
        head=path.parent/(winner['name']+'.joblib')
        if hashlib.sha256(head.read_bytes()).hexdigest()!=winner['model_SHA']:raise ValueError('Head checksum mismatch')
        return cls(joblib.load(head),winner['kind'],winner['threshold'],winner['mode'])

    def predict(self,result,legacy_confidence,boundary):
        legacy=np.asarray(legacy_confidence,dtype=np.float64)
        if not 0<boundary<1 or legacy.shape!=(len(result['top']),) or not np.isfinite(legacy).all():raise ValueError('Invalid legacy confidence')
        score=self.model.predict_proba(model_features(result,self.kind))[:,1]
        if not np.isfinite(score).all() or np.any((score<0)|(score>1)):raise ValueError('Invalid correct-candidate score')
        accepted=score>=self.threshold
        def logit(x):
            x=np.clip(x,1e-12,1-1e-12);return np.log(x/(1-x))
        shifted=1/(1+np.exp(-np.clip(logit(score)-logit(self.threshold)+logit(boundary),-50,50)))
        shifted=np.where(accepted,np.maximum(shifted,boundary),np.minimum(shifted,np.nextafter(boundary,-np.inf)))
        if self.mode=='supplement':
            old=legacy>=boundary;accepted=old|accepted;shifted=np.where(old|~accepted,legacy,shifted)
        assert np.array_equal(accepted,shifted>=boundary)
        return {'raw_score':score,'confidence':shifted,'accepted':accepted}
