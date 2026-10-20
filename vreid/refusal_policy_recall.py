"""One global Recall-oriented rescue rule; preserve the verified prior path."""
from pathlib import Path
import json
import numpy as np
from .refusal_policy_expanded import ExpandedRefusalPolicy as _Expanded
from .refusal_policy_bagged import BaggedRefusalPolicy,digest
from .refusal_heuristics import gate,candidate_rules
from .raw_refusal import raw_features

PROFILE='recall-global-v1'
MANIFEST='var/refusal-generalization-20261018/recall-runtime-selection.json'

def verified_selection(root):
    root=Path(root);spec=json.loads((root/MANIFEST).read_text(encoding='utf-8'))
    if spec['profile']!=PROFILE or spec['rerank_profile']!='current-query-v1' or spec['rule'] not in candidate_rules():
        raise ValueError('Recall profile identity mismatch')
    if spec['status'] not in ['FROZEN_DEVELOPMENT_FOR_VALIDATION','FRESH_CONFIRMED_PENDING_NATIVE','READY_LOCAL_WINDOWS_WITH_TRADEOFF']:
        raise ValueError('Recall profile is not qualified for this run')
    if not 0<spec['threshold']<1:raise ValueError('Invalid global Recall threshold')
    for path,expected in spec['files'].items():
        if digest(root/path)!=expected:raise ValueError('Recall profile checksum mismatch: '+path)
    return spec

class RecallRefusalPolicy:
    def __init__(self,baseline,threshold,rule):
        if not 0<threshold<1 or rule not in candidate_rules():raise ValueError('Invalid Recall rescue rule')
        self.baseline=baseline;self.threshold=float(threshold);self.rule=dict(rule)

    @classmethod
    def load(cls,root):
        spec=verified_selection(root)
        return cls(_Expanded.load(root),spec['threshold'],spec['rule'])

    def predict(self,result,legacy_confidence,boundary):
        previous=self.baseline.predict(result,legacy_confidence,boundary)
        rows=np.flatnonzero(~previous['accepted'])
        if not len(rows):return previous
        # Expanded already computed this exact float32-feature HGB probability.
        # Reuse it rather than evaluating another model or computing raw93 again.
        score=np.asarray(previous['expanded_score']).copy()
        if self.rule['kind']!='all':
            sub={k:np.asarray(result[k])[rows] for k in ['raw','rank','query','query_night','top']}
            sub.update(gallery=result['gallery'],_gallery_features=result['_gallery_features'],
                       _query_features={k:v[rows] for k,v in result['_query_features'].items()})
            score[rows]=np.where(gate(raw_features(sub),self.rule),score[rows],0.)
        got=BaggedRefusalPolicy.supplement(self,previous,score,np.zeros(len(score),bool),boundary)
        return {**got,'recall_score':score}
