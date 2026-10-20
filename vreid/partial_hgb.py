"""Conservative probability upper bound before computing expensive unknown fields."""
import numpy as np
from scipy.special import expit


class PartialHGB:
    """Bound each tree by its best leaf compatible with the known features.

    Unknown-feature conditions are ignored, enlarging the feasible leaf set.
    Summing tree maxima in prediction order bounds the complete binary HGB.
    This is only a rejection shortcut; it never adds an acceptance itself.
    """
    def __init__(self, model, known_columns=96):
        if list(model.classes_) != [0, 1] or model.n_features_in_ != 102:
            raise ValueError('Expected binary 102-column HGB')
        self.known_columns = known_columns
        self.baseline = float(model._baseline_prediction[0, 0])
        leaves=[];self.offsets=[]
        for stage in model._predictors:
            if len(stage) != 1:
                raise ValueError('Expected binary HGB stages')
            nodes=stage[0].nodes
            if np.any(nodes['is_categorical']):
                raise ValueError('Categorical HGB is unsupported')
            self.offsets.append(len(leaves));stack=[(0, [])]
            while stack:
                index, rules=stack.pop();node=nodes[index]
                if node['is_leaf']:
                    leaves.append((float(node['value']), rules));continue
                feature=int(node['feature_idx']);threshold=float(node['num_threshold'])
                for child, left in [(int(node['right']),False),(int(node['left']),True)]:
                    added=[(feature, threshold, left)] if feature<known_columns else []
                    stack.append((child,rules+added))
        width=max(len(rules) for _,rules in leaves)
        self.fields=np.zeros((len(leaves),width),dtype=np.int64)
        self.thresholds=np.full((len(leaves),width),np.inf)
        self.left=np.ones((len(leaves),width),dtype=bool)
        self.values=np.array([value for value,_ in leaves])
        for i,(_,rules) in enumerate(leaves):
            for j,(feature,threshold,left) in enumerate(rules):
                self.fields[i,j]=feature;self.thresholds[i,j]=threshold;self.left[i,j]=left

    def upper_probability(self, known):
        known=np.asarray(known,dtype=np.float64)
        if known.ndim!=2 or known.shape[1]!=self.known_columns or not np.isfinite(known).all():
            raise ValueError('Invalid known HGB inputs')
        outputs=[]
        for start in range(0,len(known),256):
            X=known[start:start+256]
            compatible=np.all((X[:,self.fields]<=self.thresholds)==self.left,axis=2)
            values=np.where(compatible,self.values,-np.inf)
            maxima=np.maximum.reduceat(values,self.offsets,axis=1)
            upper=np.full(len(X),self.baseline)
            for j in range(maxima.shape[1]):upper+=maxima[:,j]
            outputs.append(expit(upper))
        return np.concatenate(outputs) if outputs else np.empty(0)
