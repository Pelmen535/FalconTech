"""Gallery-relative confidence features; receives no labels or camera metadata."""
import numpy as np

FEATURE_NAMES = (
    'selected_z','selected_robust_z','gap2_z','gap3_z','gap5_z','gap10_z',
    'above_median_z','above_q75_z','above_q90_z','above_q95_z',
    'top3_mean_z','top5_mean_z','top10_mean_z','selected_raw_percentile',
    'near_selected_05std_fraction','near_selected_10std_fraction',
    'positive_z2_fraction','positive_z3_fraction','rank_gap2','rank_gap5',
    'anchor_relative_mean_z','anchor_relative_max_z','anchor_relative_q90_z',
    'overlap3','overlap5','overlap10','anchor_neighbors_mean_query_z',
    'anchor_neighbors_max_query_z','log_gallery_size','legacy_night_route',
)


def relative_features(result):
    """Each row depends only on its query and the fixed gallery.

    Gallery-neighbor geometry uses the unchanged exported CNN-branch embedding;
    query similarity and final candidate use the original fused ranking.
    """
    raw=np.asarray(result['raw'],dtype=np.float64)
    rank=np.asarray(result['rank'],dtype=np.float64)
    top=np.asarray(result['top'],dtype=np.int64)
    gallery=np.asarray(result['gallery'],dtype=np.float64)
    if raw.ndim!=2 or raw.shape!=rank.shape or len(top)!=len(raw) or raw.shape[1]<5:
        raise ValueError('Expected query-by-gallery scores with at least five gallery items')
    if gallery.ndim!=2 or len(gallery)!=raw.shape[1] or not np.isfinite(gallery).all():
        raise ValueError('Invalid gallery embeddings')
    finite=np.isfinite(raw);count=finite.sum(1)
    if np.any(count<5) or not np.isfinite(raw[np.arange(len(raw)),top]).all():
        raise ValueError('Insufficient valid query-gallery scores')
    values=np.where(finite,raw,np.nan)
    mean=np.nanmean(values,axis=1);std=np.maximum(np.nanstd(values,axis=1),1e-5)
    quant=np.nanquantile(values,[.25,.5,.75,.9,.95],axis=1)
    robust=np.maximum((quant[2]-quant[0])/1.349,1e-5)
    selected=raw[np.arange(len(raw)),top]
    order=np.argsort(-raw,axis=1,kind='stable');sorted_raw=np.take_along_axis(raw,order,axis=1)
    columns=[(selected-mean)/std,(selected-quant[1])/robust]
    for k in [2,3,5,10]:columns.append((selected-sorted_raw[:,min(k,raw.shape[1])-1])/std)
    columns.extend((selected-q)/std for q in quant[1:])
    for k in [3,5,10]:columns.append((np.nanmean(np.where(np.isfinite(sorted_raw[:,:min(k,raw.shape[1])]),sorted_raw[:,:min(k,raw.shape[1])],np.nan),axis=1)-mean)/std)
    columns.append(np.sum(finite&(raw<=selected[:,None]),axis=1)/count)
    for gap in [.5,1.]:columns.append(np.sum(finite&(raw>=selected[:,None]-gap*std[:,None]),axis=1)/count)
    for z in [2.,3.]:columns.append(np.sum(finite&(raw>=mean[:,None]+z*std[:,None]),axis=1)/count)
    rank_order=np.argsort(-rank,axis=1,kind='stable');ranked=np.take_along_axis(rank,rank_order,axis=1)
    for k in [2,5]:columns.append(ranked[:,0]-ranked[:,k-1])
    # Only rows of the candidate anchors are needed: avoids an additional GxG allocation.
    anchor=gallery[top]@gallery.T
    anchor[np.arange(len(top)),top]=np.nan
    amean=np.nanmean(anchor,axis=1);astd=np.maximum(np.nanstd(anchor,axis=1),1e-5)
    columns.extend([(selected-amean)/astd,(selected-np.nanmax(anchor,axis=1))/astd,(selected-np.nanquantile(anchor,.9,axis=1))/astd])
    neighbors=np.argsort(-np.where(np.isfinite(anchor),anchor,-np.inf),axis=1,kind='stable')
    for k in [3,5,10]:
        width=min(k,raw.shape[1]-1);rows=[]
        for i in range(len(raw)):
            qn=[int(v) for v in order[i] if v!=top[i]][:width]
            rows.append(len(set(qn)&set(neighbors[i,:width]))/width)
        columns.append(np.array(rows))
    ns=np.take_along_axis(raw,neighbors[:,:min(5,raw.shape[1]-1)],axis=1)
    ns=np.where(np.isfinite(ns),ns,np.nan)
    columns.extend([(np.nanmean(ns,axis=1)-mean)/std,(np.nanmax(ns,axis=1)-mean)/std,np.full(len(raw),np.log1p(raw.shape[1])),np.asarray(result['query_night'],dtype=float)])
    out=np.column_stack(columns)
    if out.shape!=(len(raw),len(FEATURE_NAMES)) or not np.isfinite(out).all():
        raise ValueError('Nonfinite confidence features')
    return out


def model_features(result,kind):
    relative=relative_features(result)
    if kind=='relative':return relative
    if kind=='mixed':return np.column_stack([relative,np.asarray(result['X20'],dtype=np.float64)])
    raise ValueError('Unknown confidence feature set')
