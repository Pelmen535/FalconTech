"""Identity-free local-gallery geometry for open-set confidence."""
import numpy as np
from .general_refusal import relative_features

LOCAL_NAMES=tuple(f'cluster{k}_{name}' for k in (3,5,8) for name in ('residual_ratio','center_distance_ratio','coefficient_l1','coefficient_max','coefficient_sum','query_cluster_cosine_mean'))


def local_features(result):
    relative=relative_features(result)
    q=np.asarray(result['query'],dtype=np.float64);g=np.asarray(result['gallery'],dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=g.shape[1] or not np.isfinite(q).all():raise ValueError('Invalid query embeddings')
    top=np.asarray(result['top'],dtype=np.int64);anchor=g[top];similarity=anchor@g.T;similarity[np.arange(len(q)),top]=-np.inf
    neighbors=np.argsort(-similarity,axis=1,kind='stable');features=[]
    for k in (3,5,8):
        width=min(k,len(g));rows=[]
        for i in range(len(q)):
            ix=np.r_[top[i],neighbors[i,:width-1]];cluster=g[ix];origin=cluster[0];delta=cluster[1:]-origin;query=q[i]-origin;gram=delta@delta.T
            radius=max(float(np.trace(gram)/max(1,len(delta))),1e-8)
            coef=np.linalg.solve(gram+np.eye(len(delta))*(radius*1e-3+1e-8),delta@query);residual=query-coef@delta;center=cluster.mean(0);center_radius=max(float(np.mean(np.sum((cluster-center)**2,axis=1))),1e-8)
            rows.append([float(residual@residual/radius),float(np.sum((q[i]-center)**2)/center_radius),float(np.abs(coef).sum()),float(np.abs(coef).max()),float(coef.sum()),float(np.mean(cluster@q[i]))])
        features.append(np.array(rows))
    out=np.column_stack([relative,*features])
    if out.shape!=(len(q),48) or not np.isfinite(out).all():raise ValueError('Invalid local geometry features')
    return out
