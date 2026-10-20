"""One global correctness rule using the query and its chosen candidate."""
from pathlib import Path
import json
import joblib
import numpy as np
from .refusal_policy_expanded import ExpandedRefusalPolicy as _Expanded
from .refusal_policy_bagged import BaggedRefusalPolicy, digest
from .raw_refusal import raw_features

PROFILE='pair-verifier-v1'
MANIFEST='var/refusal-generalization-20261018/balanced-runtime-selection.json'

def verified_selection(root):
    root=Path(root);spec=json.loads((root/MANIFEST).read_text(encoding='utf-8'))
    if spec['profile']!=PROFILE or spec['columns']!=285 or spec['rerank_profile']!='current-query-v1':
        raise ValueError('Pair verifier identity mismatch')
    if spec['status'] not in ['DEVELOPMENT_PASSED_PENDING_FRESH_NATIVE','READY_LOCAL_WINDOWS_WITH_DISCLOSED_LIMITS']:
        raise ValueError('Pair verifier is not qualified for this run')
    if not 0<spec['threshold']<1:raise ValueError('Invalid global pair threshold')
    for path,expected in spec['files'].items():
        if digest(root/path)!=expected:raise ValueError('Pair verifier checksum mismatch: '+path)
    return spec

def project(pack,projections):
    parts=[]
    for value,pc in zip([pack['emb'][:,:1024],pack['emb'][:,1024:],pack['specialist']],projections):
        a=np.asarray(value,dtype=np.float32);a=a/np.linalg.norm(a,axis=1,keepdims=True).clip(1e-8)
        parts.append(np.clip(pc.transform(a)/np.sqrt(pc.explained_variance_.clip(1e-7)),-8,8).astype(np.float32))
    return np.concatenate(parts,axis=1)

class PairRefusalPolicy:
    def __init__(self,baseline,model,projections,threshold):
        if list(model.classes_)!=[False,True] or model.n_features_in_!=285 or len(projections)!=3 or not 0<threshold<1:
            raise ValueError('Invalid global pair verifier')
        self.baseline=baseline;self.model=model;self.projections=projections;self.threshold=float(threshold)

    @classmethod
    def load(cls,root):
        spec=verified_selection(root);root=Path(root)
        return cls(_Expanded.load(root),joblib.load(root/spec['head']),joblib.load(root/spec['projections']),spec['threshold'])

    def predict(self,result,legacy_confidence,boundary):
        previous=self.baseline.predict(result,legacy_confidence,boundary)
        rows=np.flatnonzero(~previous['accepted'])
        if not len(rows):return previous
        q={k:v[rows] for k,v in result['_query_features'].items()};selected=np.asarray(result['top'])[rows]
        # Transform only the selected candidate for each request. No gallery
        # style inference, gallery-average descriptor or query/result cache.
        g={k:v[selected] for k,v in result['_gallery_features'].items()}
        qp=project(q,self.projections);gp=project(g,self.projections)
        x=np.concatenate([raw_features(result)[rows],(qp-gp)**2,qp*gp],axis=1).astype(np.float32)
        score=np.zeros(len(previous['accepted']),dtype=float);score[rows]=self.model.predict_proba(x)[:,1]
        got=BaggedRefusalPolicy.supplement(self,previous,score,np.zeros(len(score),bool),boundary)
        return {**got,'pair_score':score}
