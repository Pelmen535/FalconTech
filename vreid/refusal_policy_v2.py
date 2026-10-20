"""Conservative gallery-relative supplement; preserve specialized night decisions."""
import numpy as np
from .refusal_policy import GeneralRefusalPolicy as BasePolicy


class GeneralRefusalPolicy(BasePolicy):
    def __init__(self,model,kind,threshold,mode):
        if mode!='supplement_source_only' or kind!='relative':raise ValueError('Expected invariant source supplement')
        super().__init__(model,kind,threshold,'supplement')

    def predict(self,result,legacy_confidence,boundary):
        output=super().predict(result,legacy_confidence,boundary)
        night=np.asarray(result['query_night'],dtype=bool)
        legacy=np.asarray(legacy_confidence,dtype=float)
        output['confidence']=np.where(night,legacy,output['confidence'])
        output['accepted']=output['confidence']>=boundary
        return output
