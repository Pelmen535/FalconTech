"""Confidence verification with pre-projection DINO/CNN identity evidence."""
import numpy as np
from .local_refusal import local_features,LOCAL_NAMES

CHANNEL_FIELDS=('selected_cosine','selected_z','gap2_z','selected_percentile','query_similarity_std','cluster3_z_mean','cluster3_z_median','cluster3_z90','cluster3_z99','cluster3_zmax','cluster3_z3_fraction','cluster3_z6_fraction','cluster3_cosine_mean','cluster3_cosine_min','cluster3_cosine_max')
RAW_NAMES=LOCAL_NAMES+tuple(f'{channel}_{field}' for channel in ('DINO_CLS','DINO_PATCH_MEAN','CNN_RAW') for field in CHANNEL_FIELDS)


def raw_features(result):
    out=[local_features(result)];q=result['_query_features'];g=result['_gallery_features'];top=np.asarray(result['top'],dtype=np.int64);count=len(top)
    compressed=np.asarray(result['gallery'],dtype=np.float64);neighbor=compressed[top]@compressed.T;neighbor[np.arange(count),top]=-np.inf;nn=np.argsort(-neighbor,axis=1,kind='stable')[:,:2];cluster_indices=np.column_stack([top,nn])
    channels=[(q['emb'][:,:1024],g['emb'][:,:1024]),(q['emb'][:,1024:],g['emb'][:,1024:]),(q['specialist'],g['specialist'])]
    for query,gallery in channels:
        query=np.asarray(query,dtype=np.float64);gallery=np.asarray(gallery,dtype=np.float64)
        if query.ndim!=2 or len(query)!=count or not np.isfinite(query).all() or not np.isfinite(gallery).all():raise ValueError('Invalid raw descriptor')
        query=query/np.linalg.norm(query,axis=1,keepdims=True).clip(1e-9);gallery=gallery/np.linalg.norm(gallery,axis=1,keepdims=True).clip(1e-9);sims=query@gallery.T;selected=sims[np.arange(count),top];mean=sims.mean(1);std=sims.std(1).clip(1e-5);second=np.sort(sims,axis=1)[:,-2]
        cluster=gallery[cluster_indices];center=cluster.mean(1);scale=np.sqrt(cluster.var(1)+(1e-2/np.sqrt(gallery.shape[1]))**2);z=np.abs(query-center)/scale;cos=np.einsum('ijk,ik->ij',cluster,query)
        features=np.column_stack([selected,(selected-mean)/std,(selected-second)/std,np.mean(sims<=selected[:,None],axis=1),std,z.mean(1),np.median(z,axis=1),np.quantile(z,.9,axis=1),np.quantile(z,.99,axis=1),z.max(1),np.mean(z>3,axis=1),np.mean(z>6,axis=1),cos.mean(1),cos.min(1),cos.max(1)])
        out.append(features)
    result_features=np.column_stack(out)
    if result_features.shape!=(count,93) or not np.isfinite(result_features).all():raise ValueError('Invalid raw confidence features')
    return result_features
