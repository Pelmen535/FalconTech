"""Paired identity bootstrap for ablations, not a proof of feature independence."""
import numpy as np

def paired_drop(baseline, masked, vehicle_ids, seed=0, repeats=2000):
    a,b=np.asarray(baseline,float),np.asarray(masked,float);ids=np.asarray(vehicle_ids)
    if a.shape!=b.shape or a.shape!=ids.shape or a.ndim!=1:
        raise ValueError('Paired rows and identity labels must align')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any((a<0)|(a>1)|(b<0)|(b>1)):
        raise ValueError('Finite AP@10 in [0,1] required')
    unique=np.unique(ids)
    if len(unique)<2 or repeats<100:raise ValueError('Insufficient identities/bootstrap repetitions')
    delta=a-b;groups=[delta[ids==v] for v in unique];rng=np.random.default_rng(seed)
    values=[float(np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]).mean()) for _ in range(repeats)]
    return dict(mean_drop=float(delta.mean()),ci95=list(map(float,np.quantile(values,[.025,.975]))),
                n_queries=len(delta),n_identities=len(groups),seed=seed,repeats=repeats)
