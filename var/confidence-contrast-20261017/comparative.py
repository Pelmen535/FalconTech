"""Counterfactual raw93 descriptors; no labels, cameras or other queries."""
import numpy as np
from vreid.raw_refusal import raw_features


def alternatives(result, gallery_keys, exclusions=(3,5)):
    raw=np.asarray(result['raw']);rank=np.asarray(result['rank']);top=np.asarray(result['top'],np.int64);g=np.asarray(result['gallery']);keys=np.asarray(gallery_keys)
    if raw.ndim!=2 or raw.shape!=rank.shape or len(keys)!=raw.shape[1] or len(set(keys.tolist()))!=len(keys):raise ValueError('Invalid candidate geometry/keys')
    canonical=np.argsort(keys,kind='stable');neighbor=g[top]@g.T
    # The anchor itself is always excluded, independently of exact similarity ties.
    neighbor[np.arange(len(top)),top]=np.inf
    order=canonical[np.argsort(-neighbor[:,canonical],axis=1,kind='stable')]
    columns=[];valid=[]
    for width in exclusions:
        allowed=np.isfinite(raw).copy();np.put_along_axis(allowed,order[:,:min(width,len(keys))],False,axis=1)
        scores=np.where(allowed,rank,-np.inf);found=np.isfinite(scores).any(1);best=canonical[np.argmax(scores[:,canonical],axis=1)];columns.append(np.where(found,best,top));valid.append(found)
    return np.column_stack(columns),np.column_stack(valid)


def descriptors(result, indices):
    indices=np.asarray(indices,np.int64);n,k=indices.shape;rows=np.repeat(np.arange(n),k)
    r={field:np.asarray(result[field])[rows] for field in ['raw','rank','query','query_night']}
    r.update(top=indices.reshape(-1),gallery=result['gallery'],
             _query_features={field:value[rows] for field,value in result['_query_features'].items()},
             _gallery_features=result['_gallery_features'])
    return raw_features(r).reshape(n,k,93)


def relative(selected, other, family):
    selected=np.clip(selected,1e-12,1-1e-12);other=np.clip(other,1e-12,1-1e-12)
    if family=='probability':return selected/(selected+other)
    if family!='odds':raise ValueError('Unknown relative evidence family')
    delta=np.log(selected/(1-selected))-np.log(other/(1-other))
    return np.clip(1/(1+np.exp(-np.clip(delta,-50,50))),1e-12,1-1e-12)
